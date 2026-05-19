"""Directory manager for cluster file operations."""

import fnmatch
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from slurm_mcp.config import Settings
from slurm_mcp.models import ClusterDirectories, DirectoryListing, FileInfo
from slurm_mcp.ssh_client import SSHClient, SSHCommandError

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


def _bytes_to_human(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def _parse_permissions(mode: int) -> str:
    """Convert numeric mode to permission string."""
    perms = ""
    for i in range(8, -1, -1):
        if mode & (1 << i):
            perms += "rwx"[(8 - i) % 3]
        else:
            perms += "-"
    return perms


class DirectoryManager:
    """Manages cluster directory structure and file operations."""
    
    # Mapping of directory types to their config attributes
    DIRECTORY_TYPES = {
        "datasets": "dir_datasets",
        "results": "dir_results",
        "models": "dir_models",
        "logs": "dir_logs",
        "projects": "dir_projects",
        "scratch": "dir_scratch",
        "home": "dir_home",
        "root": "dir_container_root",
        "images": "image_dir",
    }
    
    def __init__(self, ssh_client: SSHClient, settings: Settings):
        """Initialize directory manager.
        
        Args:
            ssh_client: SSH client for remote operations.
            settings: Configuration settings.
        """
        self.ssh = ssh_client
        self.settings = settings
    
    def get_cluster_directories(self) -> ClusterDirectories:
        """Get the configured cluster directory structure.
        
        Returns:
            ClusterDirectories object with all paths.
        """
        return ClusterDirectories(
            user_root=self.settings.user_root,
            datasets=self.settings.dir_datasets or "",
            results=self.settings.dir_results or "",
            models=self.settings.dir_models or "",
            logs=self.settings.dir_logs or "",
            projects=self.settings.dir_projects,
            scratch=self.settings.dir_scratch,
            home=self.settings.dir_home,
            container_root=self.settings.dir_container_root,
            gpfs_root=self.settings.gpfs_root,
            images=self.settings.image_dir,
        )
    
    def resolve_path(
        self,
        path: str,
        directory_type: Optional[str] = None,
    ) -> str:
        """Resolve a path, optionally relative to a directory type.
        
        Args:
            path: The path (can be relative or absolute).
            directory_type: Directory type to use as base.
            
        Returns:
            Resolved absolute path.
            
        Raises:
            ValueError: If directory_type is invalid or not configured.
        """
        # If path is absolute, validate and return
        if path.startswith('/'):
            return self._validate_path(path)
        
        # If no directory type, use user_root as base
        if not directory_type:
            base = self.settings.user_root
        else:
            # Get base directory from settings
            if directory_type not in self.DIRECTORY_TYPES:
                raise ValueError(f"Invalid directory type: {directory_type}")
            
            attr_name = self.DIRECTORY_TYPES[directory_type]
            base = getattr(self.settings, attr_name)
            
            if not base:
                raise ValueError(f"Directory type '{directory_type}' is not configured")
        
        # Join paths
        full_path = f"{base.rstrip('/')}/{path.lstrip('/')}"
        return self._validate_path(full_path)
    
    def _validate_path(self, path: str) -> str:
        """Validate path for security.
        
        Args:
            path: Path to validate.
            
        Returns:
            Validated path.
            
        Raises:
            ValueError: If path is potentially dangerous.
        """
        # Normalize the path
        normalized = str(Path(path).resolve()) if not path.startswith('/') else path
        
        # Check for path traversal attempts
        if '..' in path:
            raise ValueError("Path traversal not allowed")
        
        # Ensure path is within allowed directories
        allowed_roots = [
            self.settings.user_root,
            self.settings.gpfs_root,
            "/tmp",
        ]
        allowed_roots = [r for r in allowed_roots if r]
        
        is_allowed = any(
            normalized.startswith(root)
            for root in allowed_roots
        )
        
        if not is_allowed:
            raise ValueError(f"Path {path} is outside allowed directories")
        
        return normalized
    
    async def list_directory(
        self,
        path: str,
        directory_type: Optional[str] = None,
        pattern: Optional[str] = None,
        recursive: bool = False,
        max_depth: Optional[int] = None,
    ) -> DirectoryListing:
        """List contents of a directory.
        
        Args:
            path: Directory path.
            directory_type: Base directory type.
            pattern: Glob pattern to filter results.
            recursive: List recursively.
            max_depth: Maximum recursion depth.
            
        Returns:
            DirectoryListing object.
        """
        full_path = self.resolve_path(path, directory_type)
        quoted_path = _quote_path(full_path)
        
        # Build command
        if recursive:
            depth_arg = f"-maxdepth {max_depth}" if max_depth else ""
            cmd = f"find {quoted_path} {depth_arg} -printf '%y|%p|%s|%T@|%m|%u|%g\\n' 2>/dev/null"
        else:
            cmd = f"find {quoted_path} -maxdepth 1 -printf '%y|%p|%s|%T@|%m|%u|%g\\n' 2>/dev/null"
        
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            raise SSHCommandError(f"Failed to list directory {full_path}: {result.stderr}")
        
        files = []
        subdirs = []
        total_size = 0
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 7:
                continue
            
            file_type = parts[0]
            file_path = parts[1]
            
            # Skip the directory itself
            if file_path == full_path:
                continue
            
            size = int(parts[2]) if parts[2].isdigit() else 0
            mtime = float(parts[3]) if parts[3] else 0
            mode = int(parts[4], 8) if parts[4] else 0
            owner = parts[5]
            group = parts[6]
            
            name = Path(file_path).name
            
            # Apply pattern filter
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue
            
            is_dir = file_type == 'd'
            is_link = file_type == 'l'
            
            file_info = FileInfo(
                name=name,
                path=file_path,
                size_bytes=size,
                size_human=_bytes_to_human(size),
                modified_time=datetime.fromtimestamp(mtime),
                is_dir=is_dir,
                is_link=is_link,
                permissions=_parse_permissions(mode),
                owner=owner,
                group=group,
            )
            
            if is_dir:
                subdirs.append(file_info)
            else:
                files.append(file_info)
                total_size += size
        
        # Sort by name
        files.sort(key=lambda x: x.name)
        subdirs.sort(key=lambda x: x.name)
        
        return DirectoryListing(
            path=full_path,
            files=files,
            subdirs=subdirs,
            total_items=len(files) + len(subdirs),
            total_size_bytes=total_size,
            total_size_human=_bytes_to_human(total_size),
        )
    
    async def list_datasets(
        self,
        pattern: Optional[str] = None,
    ) -> list[FileInfo]:
        """List available datasets.
        
        Args:
            pattern: Filter pattern.
            
        Returns:
            List of dataset directories/files.
        """
        listing = await self.list_directory(
            "",
            directory_type="datasets",
            pattern=pattern,
        )
        return listing.subdirs + listing.files
    
    async def list_model_checkpoints(
        self,
        model_name: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[FileInfo]:
        """List model checkpoints.
        
        Args:
            model_name: Filter by model name/directory.
            pattern: Filter by pattern.
            
        Returns:
            List of checkpoint files/directories.
        """
        path = model_name if model_name else ""
        
        listing = await self.list_directory(
            path,
            directory_type="models",
            pattern=pattern or "checkpoint*",
            recursive=True,
            max_depth=3,
        )
        
        return listing.subdirs + listing.files
    
    async def list_job_logs(
        self,
        job_id: Optional[int] = None,
        job_name: Optional[str] = None,
        recent: Optional[int] = None,
    ) -> list[FileInfo]:
        """List job log files.
        
        Args:
            job_id: Filter by job ID.
            job_name: Filter by job name pattern.
            recent: Only show N most recent.
            
        Returns:
            List of log files.
        """
        if job_id:
            pattern = f"*{job_id}*"
        elif job_name:
            pattern = f"*{job_name}*"
        else:
            pattern = "*.out"
        
        listing = await self.list_directory(
            "",
            directory_type="logs",
            pattern=pattern,
        )
        
        # Sort by modification time (newest first)
        all_files = sorted(
            listing.files,
            key=lambda x: x.modified_time,
            reverse=True,
        )
        
        if recent:
            all_files = all_files[:recent]
        
        return all_files
    
    async def list_results(
        self,
        experiment_name: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[FileInfo]:
        """List experiment results.
        
        Args:
            experiment_name: Filter by experiment name.
            pattern: Filter by pattern.
            
        Returns:
            List of result files/directories.
        """
        path = experiment_name if experiment_name else ""
        
        listing = await self.list_directory(
            path,
            directory_type="results",
            pattern=pattern,
        )
        
        return listing.subdirs + listing.files
    
    async def read_file(
        self,
        path: str,
        directory_type: Optional[str] = None,
        tail_lines: Optional[int] = None,
        head_lines: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> str:
        """Read file contents.
        
        Args:
            path: File path.
            directory_type: Base directory type.
            tail_lines: Only read last N lines.
            head_lines: Only read first N lines.
            encoding: File encoding.
            
        Returns:
            File contents.
        """
        full_path = self.resolve_path(path, directory_type)
        quoted_path = _quote_path(full_path)
        
        if tail_lines:
            cmd = f"tail -n {tail_lines} {quoted_path}"
            result = await self.ssh.execute(cmd)
            if result.success:
                return result.stdout
            raise SSHCommandError(f"Failed to read file: {result.stderr}")
        
        if head_lines:
            cmd = f"head -n {head_lines} {quoted_path}"
            result = await self.ssh.execute(cmd)
            if result.success:
                return result.stdout
            raise SSHCommandError(f"Failed to read file: {result.stderr}")
        
        return await self.ssh.read_remote_file(full_path, encoding=encoding)
    
    async def write_file(
        self,
        path: str,
        content: str,
        directory_type: Optional[str] = None,
        append: bool = False,
        make_dirs: bool = True,
    ) -> None:
        """Write content to a file.
        
        Args:
            path: File path.
            content: Content to write.
            directory_type: Base directory type.
            append: Append instead of overwrite.
            make_dirs: Create parent directories.
        """
        full_path = self.resolve_path(path, directory_type)
        
        if append:
            # Use shell for append
            # Escape content for shell (single quotes)
            escaped = content.replace("'", "'\\''")
            quoted_path = _quote_path(full_path)
            cmd = f"echo '{escaped}' >> {quoted_path}"
            result = await self.ssh.execute(cmd)
            if not result.success:
                raise SSHCommandError(f"Failed to append to file: {result.stderr}")
        else:
            await self.ssh.write_remote_file(content, full_path, make_dirs=make_dirs)
    
    async def get_file_info(
        self,
        path: str,
        directory_type: Optional[str] = None,
    ) -> FileInfo:
        """Get information about a file or directory.
        
        Args:
            path: Path to file/directory.
            directory_type: Base directory type.
            
        Returns:
            FileInfo object.
        """
        full_path = self.resolve_path(path, directory_type)
        
        info = await self.ssh.get_file_info(full_path)
        
        return FileInfo(
            name=info["name"],
            path=info["path"],
            size_bytes=info["size"],
            size_human=_bytes_to_human(info["size"]),
            modified_time=datetime.fromtimestamp(info["modified_time"]) if info["modified_time"] else datetime.now(),
            is_dir=info["is_dir"],
            is_link=info["is_link"],
            permissions=_parse_permissions(info["permissions"]) if info["permissions"] else "unknown",
            owner=info["owner"],
            group=info["group"],
        )
    
    async def find_files(
        self,
        pattern: str,
        directory_type: Optional[str] = None,
        path: Optional[str] = None,
        file_type: Optional[str] = None,
        min_size: Optional[str] = None,
        max_age: Optional[str] = None,
    ) -> list[FileInfo]:
        """Search for files matching criteria.
        
        Args:
            pattern: Search pattern (glob).
            directory_type: Directory type to search in.
            path: Specific path to search in.
            file_type: Filter by type ('file', 'dir', 'link').
            min_size: Minimum size (e.g., '1G', '100M').
            max_age: Maximum age (e.g., '7d', '24h').
            
        Returns:
            List of matching files.
        """
        if path:
            search_path = self.resolve_path(path, directory_type)
        elif directory_type:
            search_path = self.resolve_path("", directory_type)
        else:
            search_path = self.settings.user_root
        
        # Build find command (quote path for spaces/special chars)
        quoted_path = _quote_path(search_path)
        cmd = f"find {quoted_path} -name '{pattern}'"
        
        if file_type:
            type_map = {"file": "f", "dir": "d", "link": "l"}
            if file_type in type_map:
                cmd += f" -type {type_map[file_type]}"
        
        if min_size:
            cmd += f" -size +{min_size}"
        
        if max_age:
            # Parse age string (e.g., "7d", "24h")
            match = re.match(r'^(\d+)([dhm])$', max_age)
            if match:
                num = match.group(1)
                unit = match.group(2)
                if unit == 'd':
                    cmd += f" -mtime -{num}"
                elif unit == 'h':
                    cmd += f" -mmin -{int(num) * 60}"
                elif unit == 'm':
                    cmd += f" -mmin -{num}"
        
        cmd += " -printf '%y|%p|%s|%T@|%m|%u|%g\\n' 2>/dev/null"
        
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            return []
        
        files = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 7:
                continue
            
            ftype = parts[0]
            fpath = parts[1]
            size = int(parts[2]) if parts[2].isdigit() else 0
            mtime = float(parts[3]) if parts[3] else 0
            mode = int(parts[4], 8) if parts[4] else 0
            owner = parts[5]
            group = parts[6]
            
            files.append(FileInfo(
                name=Path(fpath).name,
                path=fpath,
                size_bytes=size,
                size_human=_bytes_to_human(size),
                modified_time=datetime.fromtimestamp(mtime),
                is_dir=ftype == 'd',
                is_link=ftype == 'l',
                permissions=_parse_permissions(mode),
                owner=owner,
                group=group,
            ))
        
        return files
    
    async def delete_file(
        self,
        path: str,
        directory_type: Optional[str] = None,
        recursive: bool = False,
    ) -> None:
        """Delete a file or directory.
        
        Args:
            path: Path to delete.
            directory_type: Base directory type.
            recursive: Delete directories recursively.
        """
        full_path = self.resolve_path(path, directory_type)
        
        # Extra validation for destructive operation
        if full_path in [
            self.settings.user_root,
            self.settings.gpfs_root,
            self.settings.dir_datasets,
            self.settings.dir_models,
            self.settings.dir_results,
        ]:
            raise ValueError(f"Cannot delete root directory: {full_path}")
        
        info = await self.ssh.get_file_info(full_path)
        
        if info["is_dir"]:
            await self.ssh.delete_directory(full_path, recursive=recursive)
        else:
            await self.ssh.delete_file(full_path)
    
    async def download(
        self,
        path: str,
        directory_type: Optional[str] = None,
        local_path: Optional[str] = None,
        recursive: bool = False,
        overwrite: bool = False,
    ) -> dict:
        """Download a file or directory from the cluster to the local machine.

        Args:
            path: Remote path to download.
            directory_type: Base directory type for resolving ``path``.
            local_path: Local destination. If omitted, saves to ``~/Downloads/<name>``.
                If a directory, the remote item is placed inside it with its
                original name.
            recursive: Required when downloading a directory.
            overwrite: If False, refuse to overwrite an existing local file.

        Returns:
            Dictionary with: ``local_path``, ``remote_path``, ``bytes``, ``files``.
        """
        remote_full = self.resolve_path(path, directory_type)
        name = Path(remote_full).name

        if local_path:
            local = Path(local_path).expanduser()
            if local.is_dir():
                local = local / name
        else:
            local = Path.home() / "Downloads" / name

        if local.exists() and not overwrite:
            raise FileExistsError(
                f"Local path already exists: {local} (set overwrite=True to replace)"
            )

        if local.exists() and overwrite:
            if local.is_dir():
                import shutil
                shutil.rmtree(local)
            else:
                local.unlink()

        stats = await self.ssh.download_file(
            remote_path=remote_full,
            local_path=str(local),
            recursive=recursive,
        )

        return {
            "local_path": str(local),
            "remote_path": remote_full,
            "bytes": stats["bytes"],
            "files": stats["files"],
        }

    async def get_disk_usage(
        self,
        directory_type: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """Get disk usage information.
        
        Args:
            directory_type: Directory type to check.
            path: Specific path to check.
            
        Returns:
            Dictionary with disk usage info.
        """
        if path:
            check_path = self.resolve_path(path, directory_type)
            paths = {path: check_path}
        elif directory_type:
            check_path = self.resolve_path("", directory_type)
            paths = {directory_type: check_path}
        else:
            # Check all configured directories
            paths = {}
            for dtype, attr in self.DIRECTORY_TYPES.items():
                dir_path = getattr(self.settings, attr)
                if dir_path:
                    paths[dtype] = dir_path
        
        usage = {}
        
        for name, dir_path in paths.items():
            quoted_path = _quote_path(dir_path)
            cmd = f"du -sb {quoted_path} 2>/dev/null"
            result = await self.ssh.execute(cmd)
            
            if result.success and result.stdout.strip():
                parts = result.stdout.strip().split()
                if parts and parts[0].isdigit():
                    size = int(parts[0])
                    usage[name] = {
                        "path": dir_path,
                        "size_bytes": size,
                        "size_human": _bytes_to_human(size),
                    }
        
        # Get filesystem info
        if paths:
            first_path = list(paths.values())[0]
            quoted_first = _quote_path(first_path)
            cmd = f"df -B1 {quoted_first} 2>/dev/null"
            result = await self.ssh.execute(cmd)
            
            if result.success:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 4:
                        usage["filesystem"] = {
                            "total_bytes": int(parts[1]) if parts[1].isdigit() else 0,
                            "used_bytes": int(parts[2]) if parts[2].isdigit() else 0,
                            "available_bytes": int(parts[3]) if parts[3].isdigit() else 0,
                            "total_human": _bytes_to_human(int(parts[1])) if parts[1].isdigit() else "unknown",
                            "used_human": _bytes_to_human(int(parts[2])) if parts[2].isdigit() else "unknown",
                            "available_human": _bytes_to_human(int(parts[3])) if parts[3].isdigit() else "unknown",
                        }
        
        return usage


# Global directory manager instance
_directory_manager: Optional[DirectoryManager] = None


def get_directory_manager(
    ssh_client: Optional[SSHClient] = None,
    settings: Optional[Settings] = None,
) -> DirectoryManager:
    """Get or create the global directory manager instance.
    
    Args:
        ssh_client: SSH client (required on first call).
        settings: Settings (required on first call).
        
        Returns:
        DirectoryManager instance.
    """
    global _directory_manager
    
    if _directory_manager is None:
        if ssh_client is None or settings is None:
            raise ValueError("ssh_client and settings required on first call")
        _directory_manager = DirectoryManager(ssh_client, settings)
    
    return _directory_manager


def reset_directory_manager() -> None:
    """Reset the global directory manager."""
    global _directory_manager
    _directory_manager = None
