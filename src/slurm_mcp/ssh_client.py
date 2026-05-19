"""SSH client for connecting to remote Slurm login nodes."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import asyncssh

from slurm_mcp.config import Settings
from slurm_mcp.models import CommandResult

logger = logging.getLogger(__name__)


def _quote_path(path: str) -> str:
    """Quote a path for safe use in shell commands.
    
    Uses double quotes to handle paths with spaces and most special characters.
    Escapes any existing double quotes, backticks, and dollar signs in the path.
    
    Args:
        path: The file path to quote.
        
    Returns:
        Quoted path safe for shell use.
    """
    # Escape characters that have special meaning inside double quotes
    escaped = path.replace('\\', '\\\\')  # Escape backslashes first
    escaped = escaped.replace('"', '\\"')  # Escape double quotes
    escaped = escaped.replace('`', '\\`')  # Escape backticks
    escaped = escaped.replace('$', '\\$')  # Escape dollar signs
    return f'"{escaped}"'


class SSHConnectionError(Exception):
    """Raised when SSH connection fails."""
    pass


class SSHCommandError(Exception):
    """Raised when SSH command execution fails."""
    pass


class SSHClient:
    """Manages SSH connections to the Slurm login node.
    
    This client handles connection lifecycle, command execution,
    and file operations over SSH using asyncssh.
    """
    
    def __init__(self, settings: Settings):
        """Initialize SSH client with settings.
        
        Args:
            settings: Configuration settings containing SSH connection details.
        """
        self.settings = settings
        self._connection: Optional[asyncssh.SSHClientConnection] = None
        self._lock = asyncio.Lock()
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connection is not None and not self._connection.is_closed()
    
    async def connect(self) -> None:
        """Establish SSH connection to the remote host.
        
        Raises:
            SSHConnectionError: If connection fails.
        """
        async with self._lock:
            if self.is_connected:
                return
            
            try:
                connect_kwargs: dict = {
                    "host": self.settings.ssh_host,
                    "port": self.settings.ssh_port,
                    "username": self.settings.ssh_user,
                }
                
                # Handle SSH key authentication
                if self.settings.ssh_key_path:
                    key_path = Path(self.settings.ssh_key_path).expanduser()
                    if key_path.exists():
                        connect_kwargs["client_keys"] = [str(key_path)]
                        if self.settings.ssh_password:
                            # Password is passphrase for the key
                            connect_kwargs["passphrase"] = self.settings.ssh_password
                    else:
                        logger.warning(f"SSH key not found at {key_path}, falling back to other auth methods")
                
                # Handle password authentication
                if self.settings.ssh_password and "client_keys" not in connect_kwargs:
                    connect_kwargs["password"] = self.settings.ssh_password
                
                # Handle known_hosts
                if self.settings.ssh_known_hosts:
                    known_hosts_path = Path(self.settings.ssh_known_hosts).expanduser()
                    if known_hosts_path.exists():
                        connect_kwargs["known_hosts"] = str(known_hosts_path)
                    else:
                        logger.warning(f"Known hosts file not found at {known_hosts_path}")
                        connect_kwargs["known_hosts"] = None  # Disable host key checking
                else:
                    # Default: try system known_hosts, disable if not available
                    connect_kwargs["known_hosts"] = None
                
                logger.info(f"Connecting to {self.settings.ssh_user}@{self.settings.ssh_host}:{self.settings.ssh_port}")
                self._connection = await asyncssh.connect(**connect_kwargs)
                logger.info("SSH connection established successfully")
                
            except asyncssh.Error as e:
                raise SSHConnectionError(f"Failed to connect to {self.settings.ssh_host}: {e}") from e
            except Exception as e:
                raise SSHConnectionError(f"Unexpected error connecting to {self.settings.ssh_host}: {e}") from e
    
    async def disconnect(self) -> None:
        """Close the SSH connection."""
        async with self._lock:
            if self._connection:
                self._connection.close()
                await self._connection.wait_closed()
                self._connection = None
                logger.info("SSH connection closed")
    
    async def ensure_connected(self) -> None:
        """Ensure connection is established, reconnecting if necessary."""
        if not self.is_connected:
            await self.connect()
    
    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        check: bool = False,
        working_directory: Optional[str] = None,
    ) -> CommandResult:
        """Execute a command on the remote host.
        
        Args:
            command: The command to execute.
            timeout: Command timeout in seconds (uses settings default if not specified).
            check: If True, raise exception on non-zero return code.
            working_directory: Directory to run command in.
            
        Returns:
            CommandResult with stdout, stderr, and return code.
            
        Raises:
            SSHConnectionError: If not connected and cannot connect.
            SSHCommandError: If check=True and command returns non-zero.
        """
        await self.ensure_connected()
        
        if timeout is None:
            timeout = self.settings.command_timeout
        
        # Wrap command with cd if working directory specified
        if working_directory:
            command = f"cd {working_directory} && {command}"
        
        try:
            logger.debug(f"Executing command: {command[:100]}...")
            
            result = await asyncio.wait_for(
                self._connection.run(command, check=False),
                timeout=timeout
            )
            
            cmd_result = CommandResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                return_code=result.exit_status or 0,
            )
            
            logger.debug(f"Command completed with return code {cmd_result.return_code}")
            
            if check and not cmd_result.success:
                raise SSHCommandError(
                    f"Command failed with return code {cmd_result.return_code}: {cmd_result.stderr}"
                )
            
            return cmd_result
            
        except asyncio.TimeoutError:
            raise SSHCommandError(f"Command timed out after {timeout} seconds: {command[:50]}...")
        except asyncssh.Error as e:
            # Connection might be broken, clear it
            self._connection = None
            raise SSHCommandError(f"SSH error executing command: {e}") from e
    
    async def execute_interactive(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
    ) -> CommandResult:
        """Execute a command with PTY (for interactive commands).
        
        This is useful for commands that require a terminal.
        
        Args:
            command: The command to execute.
            timeout: Command timeout in seconds.
            working_directory: Directory to run command in.
            
        Returns:
            CommandResult with combined output.
        """
        await self.ensure_connected()
        
        if timeout is None:
            timeout = self.settings.command_timeout
        
        if working_directory:
            command = f"cd {working_directory} && {command}"
        
        try:
            logger.debug(f"Executing interactive command: {command[:100]}...")
            
            async with self._connection.create_process(
                command,
                term_type="xterm",
            ) as process:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )
                    return_code = process.exit_status or 0
                except asyncio.TimeoutError:
                    process.terminate()
                    raise SSHCommandError(f"Interactive command timed out after {timeout} seconds")
            
            return CommandResult(
                stdout=stdout or "",
                stderr=stderr or "",
                return_code=return_code,
            )
            
        except asyncssh.Error as e:
            self._connection = None
            raise SSHCommandError(f"SSH error executing interactive command: {e}") from e
    
    async def write_remote_file(
        self,
        content: str,
        remote_path: str,
        mode: int = 0o644,
        make_dirs: bool = False,
    ) -> None:
        """Write content to a file on the remote host.
        
        Args:
            content: The content to write.
            remote_path: Path on the remote host.
            mode: File permissions (default 0o644).
            make_dirs: Create parent directories if they don't exist.
            
        Raises:
            SSHCommandError: If write fails.
        """
        await self.ensure_connected()
        
        try:
            if make_dirs:
                parent_dir = str(Path(remote_path).parent)
                quoted_parent = _quote_path(parent_dir)
                await self.execute(f"mkdir -p {quoted_parent}")
            
            async with self._connection.start_sftp_client() as sftp:
                async with sftp.open(remote_path, "w") as f:
                    await f.write(content)
                await sftp.chmod(remote_path, mode)
                
            logger.debug(f"Wrote {len(content)} bytes to {remote_path}")
            
        except asyncssh.Error as e:
            raise SSHCommandError(f"Failed to write to {remote_path}: {e}") from e
    
    async def read_remote_file(
        self,
        remote_path: str,
        encoding: str = "utf-8",
    ) -> str:
        """Read content from a file on the remote host.
        
        Args:
            remote_path: Path on the remote host.
            encoding: File encoding (default utf-8).
            
        Returns:
            File content as string.
            
        Raises:
            SSHCommandError: If read fails.
        """
        await self.ensure_connected()
        
        try:
            async with self._connection.start_sftp_client() as sftp:
                async with sftp.open(remote_path, "r") as f:
                    content = await f.read()
                    if isinstance(content, bytes):
                        content = content.decode(encoding)
                    return content
                    
        except asyncssh.Error as e:
            raise SSHCommandError(f"Failed to read {remote_path}: {e}") from e
    
    async def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the remote host.
        
        Args:
            remote_path: Path to check.
            
        Returns:
            True if file exists, False otherwise.
        """
        await self.ensure_connected()
        
        try:
            async with self._connection.start_sftp_client() as sftp:
                await sftp.stat(remote_path)
                return True
        except asyncssh.SFTPNoSuchFile:
            return False
        except asyncssh.Error:
            return False
    
    async def list_directory(
        self,
        remote_path: str,
        pattern: Optional[str] = None,
    ) -> list[dict]:
        """List contents of a directory on the remote host.
        
        Args:
            remote_path: Directory path.
            pattern: Optional glob pattern to filter results.
            
        Returns:
            List of file info dictionaries.
        """
        await self.ensure_connected()
        
        try:
            async with self._connection.start_sftp_client() as sftp:
                entries = []
                async for entry in sftp.scandir(remote_path):
                    if pattern:
                        import fnmatch
                        if not fnmatch.fnmatch(entry.filename, pattern):
                            continue
                    
                    entries.append({
                        "name": entry.filename,
                        "path": f"{remote_path}/{entry.filename}",
                        "size": entry.attrs.size or 0,
                        "is_dir": entry.attrs.type == asyncssh.FILEXFER_TYPE_DIRECTORY,
                        "is_link": entry.attrs.type == asyncssh.FILEXFER_TYPE_SYMLINK,
                        "permissions": entry.attrs.permissions,
                        "modified_time": entry.attrs.mtime,
                        "owner": entry.attrs.uid,
                        "group": entry.attrs.gid,
                    })
                
                return entries
                
        except asyncssh.Error as e:
            raise SSHCommandError(f"Failed to list directory {remote_path}: {e}") from e
    
    async def delete_file(self, remote_path: str) -> None:
        """Delete a file on the remote host.
        
        Args:
            remote_path: Path to the file.
            
        Raises:
            SSHCommandError: If deletion fails.
        """
        await self.ensure_connected()
        
        try:
            async with self._connection.start_sftp_client() as sftp:
                await sftp.remove(remote_path)
            logger.debug(f"Deleted file {remote_path}")
        except asyncssh.Error as e:
            raise SSHCommandError(f"Failed to delete {remote_path}: {e}") from e
    
    async def delete_directory(self, remote_path: str, recursive: bool = False) -> None:
        """Delete a directory on the remote host.
        
        Args:
            remote_path: Path to the directory.
            recursive: If True, delete contents recursively.
            
        Raises:
            SSHCommandError: If deletion fails.
        """
        await self.ensure_connected()
        
        if recursive:
            # Use rm -rf for recursive deletion (quote path for spaces/special chars)
            quoted_path = _quote_path(remote_path)
            result = await self.execute(f"rm -rf {quoted_path}")
            if not result.success:
                raise SSHCommandError(f"Failed to delete directory {remote_path}: {result.stderr}")
        else:
            try:
                async with self._connection.start_sftp_client() as sftp:
                    await sftp.rmdir(remote_path)
            except asyncssh.Error as e:
                raise SSHCommandError(f"Failed to delete directory {remote_path}: {e}") from e
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        recursive: bool = False,
    ) -> dict:
        """Download a file or directory from the remote host via SFTP.

        Args:
            remote_path: Path on the remote host.
            local_path: Destination path on the local filesystem.
            recursive: If True, download directories recursively.

        Returns:
            Dictionary with download stats: ``{"bytes": int, "files": int}``.

        Raises:
            SSHCommandError: If download fails.
        """
        await self.ensure_connected()

        try:
            async with self._connection.start_sftp_client() as sftp:
                attrs = await sftp.stat(remote_path)
                is_dir = attrs.type == asyncssh.FILEXFER_TYPE_DIRECTORY

                if is_dir and not recursive:
                    raise SSHCommandError(
                        f"{remote_path} is a directory; pass recursive=True to download it"
                    )

                local = Path(local_path).expanduser()
                local.parent.mkdir(parents=True, exist_ok=True)

                await sftp.get(remote_path, str(local), recurse=recursive, preserve=True)

            total_bytes = 0
            file_count = 0
            if local.is_dir():
                for p in local.rglob("*"):
                    if p.is_file():
                        total_bytes += p.stat().st_size
                        file_count += 1
            elif local.is_file():
                total_bytes = local.stat().st_size
                file_count = 1

            logger.debug(f"Downloaded {file_count} file(s) ({total_bytes} bytes) to {local}")
            return {"bytes": total_bytes, "files": file_count}

        except asyncssh.Error as e:
            raise SSHCommandError(f"Failed to download {remote_path}: {e}") from e

    async def get_file_info(self, remote_path: str) -> dict:
        """Get information about a file or directory.
        
        Args:
            remote_path: Path to the file/directory.
            
        Returns:
            Dictionary with file information.
            
        Raises:
            SSHCommandError: If stat fails.
        """
        await self.ensure_connected()
        
        try:
            async with self._connection.start_sftp_client() as sftp:
                attrs = await sftp.stat(remote_path)
                
                # Get owner/group names using shell command
                quoted_path = _quote_path(remote_path)
                result = await self.execute(f"stat -c '%U %G' {quoted_path}")
                owner, group = "unknown", "unknown"
                if result.success and result.stdout.strip():
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        owner, group = parts[0], parts[1]
                
                return {
                    "path": remote_path,
                    "name": Path(remote_path).name,
                    "size": attrs.size or 0,
                    "is_dir": attrs.type == asyncssh.FILEXFER_TYPE_DIRECTORY,
                    "is_link": attrs.type == asyncssh.FILEXFER_TYPE_SYMLINK,
                    "permissions": attrs.permissions,
                    "modified_time": attrs.mtime,
                    "owner": owner,
                    "group": group,
                }
                
        except asyncssh.Error as e:
            raise SSHCommandError(f"Failed to stat {remote_path}: {e}") from e
    
    async def __aenter__(self) -> "SSHClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# Global SSH client instance (lazy initialized)
_ssh_client: Optional[SSHClient] = None


def get_ssh_client(settings: Optional[Settings] = None) -> SSHClient:
    """Get or create the global SSH client instance.
    
    Args:
        settings: Optional settings to use. If not provided, uses default settings.
        
    Returns:
        SSHClient instance.
    """
    global _ssh_client
    
    if _ssh_client is None:
        if settings is None:
            from slurm_mcp.config import get_settings
            settings = get_settings()
        _ssh_client = SSHClient(settings)
    
    return _ssh_client


async def reset_ssh_client() -> None:
    """Reset the global SSH client (disconnect and clear)."""
    global _ssh_client
    
    if _ssh_client is not None:
        await _ssh_client.disconnect()
        _ssh_client = None
