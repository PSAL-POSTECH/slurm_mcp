"""Wrapper for Slurm commands executed via SSH."""

import json
import logging
import re
from datetime import datetime
from typing import Optional

from slurm_mcp.config import Settings
from slurm_mcp.models import (
    CommandResult,
    ContainerImage,
    GPUInfo,
    JobInfo,
    JobSubmission,
    NodeInfo,
    PartitionInfo,
)
from slurm_mcp.ssh_client import SSHClient, SSHCommandError

logger = logging.getLogger(__name__)


def _escape_for_single_quotes(command: str) -> str:
    """Escape a command string for use inside single quotes in bash.
    
    Single quotes in bash don't allow any escaping inside them.
    The trick is to end the single-quoted string, add an escaped single quote,
    and start a new single-quoted string.
    
    Example:
        python -c 'print("hello")'
    becomes:
        python -c '\''print("hello")'\''
    
    Args:
        command: The command string to escape.
        
    Returns:
        Escaped command string safe for use inside single quotes.
    """
    # Replace ' with '\'' (end quote, escaped quote, start quote)
    return command.replace("'", "'\\''")


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


def _parse_size_to_bytes(size_str: str) -> int:
    """Parse human-readable size string to bytes."""
    size_str = size_str.strip().upper()
    
    multipliers = {
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 ** 2,
        'MB': 1024 ** 2,
        'G': 1024 ** 3,
        'GB': 1024 ** 3,
        'T': 1024 ** 4,
        'TB': 1024 ** 4,
    }
    
    # Extract number and unit
    match = re.match(r'^([\d.]+)\s*([A-Z]*)', size_str)
    if match:
        num = float(match.group(1))
        unit = match.group(2) or 'B'
        return int(num * multipliers.get(unit, 1))
    
    return 0


def _bytes_to_human(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def _parse_slurm_time(time_str: str) -> Optional[str]:
    """Parse Slurm time format and return normalized string."""
    if not time_str or time_str in ['UNLIMITED', 'INVALID', 'N/A', 'n/a']:
        return time_str if time_str else None
    return time_str


def _parse_gres(gres_str: str, features: str = "") -> list[GPUInfo]:
    """Parse GRES string to extract GPU information.
    
    Examples:
        "gpu:a100:4" -> [GPUInfo(gpu_type='a100', count=4)]
        "gpu:4" -> [GPUInfo(gpu_type='gpu', count=4)]
        "gpu:8(S:0-1)" with features="H100" -> [GPUInfo(gpu_type='h100', count=8)]
        "gpu:a100:2,gpu:v100:4" -> [GPUInfo(...), GPUInfo(...)]
    """
    gpus = []
    if not gres_str or gres_str in ['(null)', 'N/A', '']:
        return gpus
    
    # Try to extract GPU type from features if available
    gpu_type_from_features = None
    if features:
        # Look for known GPU types in features (e.g., "location=ap-tokyo-1,GPU,H100")
        known_gpus = ['h100', 'a100', 'v100', 'a10', 'l40', 't4', 'a6000', 'rtx']
        for feat in features.lower().replace(',', ' ').split():
            for known in known_gpus:
                if known in feat:
                    gpu_type_from_features = feat
                    break
            if gpu_type_from_features:
                break
    
    for part in gres_str.split(','):
        part = part.strip()
        if not part.startswith('gpu'):
            continue
        
        # Remove socket affinity info like (S:0-1)
        base_part = part.split('(')[0]
        parts = base_part.split(':')
        
        if len(parts) == 2:
            # Format: gpu:count or gpu:count(S:0-1)
            try:
                count = int(parts[1])
                gtype = gpu_type_from_features or 'gpu'
                gpus.append(GPUInfo(gpu_type=gtype, count=count))
            except ValueError:
                pass
        elif len(parts) >= 3:
            # Format: gpu:type:count
            try:
                gpu_type = parts[1]
                count = int(parts[2])
                gpus.append(GPUInfo(gpu_type=gpu_type, count=count))
            except (ValueError, IndexError):
                pass
    
    return gpus


class SlurmCommands:
    """Wrapper for Slurm commands executed via SSH."""
    
    def __init__(self, ssh_client: SSHClient, settings: Settings):
        """Initialize Slurm commands wrapper.
        
        Args:
            ssh_client: SSH client for remote execution.
            settings: Configuration settings.
        """
        self.ssh = ssh_client
        self.settings = settings
    
    # =========================================================================
    # Cluster Status Commands
    # =========================================================================
    
    async def sinfo(
        self,
        partition: Optional[str] = None,
        node: Optional[str] = None,
        format_str: Optional[str] = None,
    ) -> str:
        """Run sinfo command and return raw output.
        
        Args:
            partition: Filter by partition name.
            node: Filter by node name.
            format_str: Custom format string.
            
        Returns:
            Raw sinfo output.
        """
        cmd = "sinfo"
        
        if partition:
            cmd += f" -p {partition}"
        if node:
            cmd += f" -n {node}"
        if format_str:
            cmd += f" -o '{format_str}'"
        
        result = await self.ssh.execute(cmd)
        return result.stdout if result.success else result.stderr
    
    async def get_partitions(self) -> list[PartitionInfo]:
        """Get information about all partitions.
        
        Returns:
            List of PartitionInfo objects.
        """
        # Use custom format to get all needed fields including features
        format_str = "%P|%a|%l|%D|%C|%G|%F|%f"
        # %P=partition, %a=state, %l=timelimit, %D=nodes, %C=cpus(A/I/O/T), %G=gres, %F=nodes(A/I/O/T), %f=features
        
        cmd = f"sinfo -h -o '{format_str}'"
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            logger.error(f"sinfo failed: {result.stderr}")
            return []
        
        partitions = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 7:
                continue
            
            name = parts[0].rstrip('*')
            is_default = parts[0].endswith('*')
            state = parts[1]
            max_time = _parse_slurm_time(parts[2])
            total_nodes = int(parts[3]) if parts[3].isdigit() else 0
            
            # Parse CPU info (A/I/O/T format)
            cpu_parts = parts[4].split('/')
            if len(cpu_parts) == 4:
                cpus_allocated = int(cpu_parts[0]) if cpu_parts[0].isdigit() else 0
                cpus_idle = int(cpu_parts[1]) if cpu_parts[1].isdigit() else 0
                total_cpus = int(cpu_parts[3]) if cpu_parts[3].isdigit() else 0
                available_cpus = cpus_idle
            else:
                total_cpus = available_cpus = 0
            
            # Parse GRES for GPU info (with features for GPU type detection)
            gres = parts[5]
            features = parts[7] if len(parts) > 7 else ""
            gpus = _parse_gres(gres, features)
            has_gpus = len(gpus) > 0
            gpu_types = list(set(g.gpu_type for g in gpus if g.gpu_type != 'gpu'))
            total_gpus = sum(g.count for g in gpus)
            
            # Parse node state (A/I/O/T format)
            node_parts = parts[6].split('/')
            if len(node_parts) == 4:
                nodes_allocated = int(node_parts[0]) if node_parts[0].isdigit() else 0
                nodes_idle = int(node_parts[1]) if node_parts[1].isdigit() else 0
                available_nodes = nodes_idle
            else:
                available_nodes = 0
            
            # Merge with existing partition entry if exists
            if name in partitions:
                existing = partitions[name]
                existing.total_nodes += total_nodes
                existing.available_nodes += available_nodes
                existing.total_cpus += total_cpus
                existing.available_cpus += available_cpus
                existing.total_gpus += total_gpus
                if gpu_types:
                    existing.gpu_types = list(set(existing.gpu_types + gpu_types))
                existing.has_gpus = existing.has_gpus or has_gpus
            else:
                partitions[name] = PartitionInfo(
                    name=name,
                    state=state,
                    total_nodes=total_nodes,
                    available_nodes=available_nodes,
                    total_cpus=total_cpus,
                    available_cpus=available_cpus,
                    max_time=max_time,
                    default=is_default,
                    has_gpus=has_gpus,
                    gpu_types=gpu_types,
                    total_gpus=total_gpus,
                    available_gpus=0,  # Will be calculated separately if needed
                )
        
        return list(partitions.values())
    
    async def get_nodes(
        self,
        partition: Optional[str] = None,
        state: Optional[str] = None,
    ) -> list[NodeInfo]:
        """Get information about cluster nodes.
        
        Args:
            partition: Filter by partition.
            state: Filter by node state.
            
        Returns:
            List of NodeInfo objects.
        """
        # Format: NodeName|State|CPUsTotal|CPUsAlloc|Memory|AllocMem|Partitions|Gres|Features
        format_str = "%N|%T|%c|%C|%m|%e|%P|%G|%f"
        
        cmd = f"sinfo -N -h -o '{format_str}'"
        if partition:
            cmd += f" -p {partition}"
        if state:
            cmd += f" -t {state}"
        
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            logger.error(f"sinfo failed: {result.stderr}")
            return []
        
        nodes = {}
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 9:
                continue
            
            node_name = parts[0]
            
            # Skip if we already have this node (can appear multiple times for different partitions)
            if node_name in nodes:
                # Just add the partition
                if parts[6] and parts[6] not in nodes[node_name].partitions:
                    nodes[node_name].partitions.append(parts[6])
                continue
            
            state = parts[1]
            cpus_total = int(parts[2]) if parts[2].isdigit() else 0
            
            # CPU allocation format: A/I/O/T
            cpu_alloc_parts = parts[3].split('/')
            if len(cpu_alloc_parts) == 4:
                cpus_allocated = int(cpu_alloc_parts[0]) if cpu_alloc_parts[0].isdigit() else 0
            else:
                cpus_allocated = 0
            
            memory_total = int(parts[4]) if parts[4].isdigit() else 0
            memory_free = int(parts[5]) if parts[5].isdigit() else 0
            memory_allocated = memory_total - memory_free
            
            partitions_list = [p for p in parts[6].split(',') if p]
            
            features_str = parts[8] if len(parts) > 8 else ""
            gpus = _parse_gres(parts[7], features_str)
            features = [f for f in features_str.split(',') if f]
            
            nodes[node_name] = NodeInfo(
                node_name=node_name,
                state=state,
                cpus_total=cpus_total,
                cpus_allocated=cpus_allocated,
                cpus_available=cpus_total - cpus_allocated,
                memory_total_mb=memory_total,
                memory_allocated_mb=memory_allocated,
                memory_available_mb=memory_free,
                partitions=partitions_list,
                gpus=gpus if gpus else None,
                features=features,
            )
        
        return list(nodes.values())
    
    async def get_gpu_info(
        self,
        partition: Optional[str] = None,
    ) -> dict:
        """Get GPU availability information.
        
        Args:
            partition: Filter by partition.
            
        Returns:
            Dictionary with GPU availability info.
        """
        # Include features (%f) to detect GPU type
        cmd = "sinfo -h -o '%P|%G|%D|%T|%f' --Node"
        if partition:
            cmd += f" -p {partition}"
        
        result = await self.ssh.execute(cmd)
        
        gpu_info = {
            "by_partition": {},
            "by_type": {},
            "total_gpus": 0,
            "allocated_gpus": 0,
            "available_gpus": 0,
        }
        
        if not result.success:
            return gpu_info
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 4:
                continue
            
            part_name = parts[0].rstrip('*')
            gres = parts[1]
            node_count = int(parts[2]) if parts[2].isdigit() else 0
            state = parts[3].lower()
            features = parts[4] if len(parts) > 4 else ""
            
            gpus = _parse_gres(gres, features)
            if not gpus:
                continue
            
            for gpu in gpus:
                total = gpu.count * node_count
                
                # Determine allocated based on state
                if 'alloc' in state or 'mix' in state:
                    allocated = total if 'alloc' in state else total // 2
                else:
                    allocated = 0
                
                available = total - allocated
                
                # Update partition stats
                if part_name not in gpu_info["by_partition"]:
                    gpu_info["by_partition"][part_name] = {
                        "total": 0, "allocated": 0, "available": 0, "types": []
                    }
                gpu_info["by_partition"][part_name]["total"] += total
                gpu_info["by_partition"][part_name]["allocated"] += allocated
                gpu_info["by_partition"][part_name]["available"] += available
                if gpu.gpu_type not in gpu_info["by_partition"][part_name]["types"]:
                    gpu_info["by_partition"][part_name]["types"].append(gpu.gpu_type)
                
                # Update type stats
                if gpu.gpu_type not in gpu_info["by_type"]:
                    gpu_info["by_type"][gpu.gpu_type] = {"total": 0, "allocated": 0, "available": 0}
                gpu_info["by_type"][gpu.gpu_type]["total"] += total
                gpu_info["by_type"][gpu.gpu_type]["allocated"] += allocated
                gpu_info["by_type"][gpu.gpu_type]["available"] += available
                
                # Update totals
                gpu_info["total_gpus"] += total
                gpu_info["allocated_gpus"] += allocated
                gpu_info["available_gpus"] += available
        
        return gpu_info
    
    # =========================================================================
    # Job Management Commands
    # =========================================================================
    
    async def squeue(
        self,
        user: Optional[str] = None,
        partition: Optional[str] = None,
        job_id: Optional[int] = None,
        state: Optional[str] = None,
    ) -> str:
        """Run squeue command and return raw output.
        
        Args:
            user: Filter by username.
            partition: Filter by partition.
            job_id: Filter by job ID.
            state: Filter by job state.
            
        Returns:
            Raw squeue output.
        """
        cmd = "squeue"
        
        if user:
            cmd += f" -u {user}"
        if partition:
            cmd += f" -p {partition}"
        if job_id:
            cmd += f" -j {job_id}"
        if state:
            cmd += f" -t {state}"
        
        result = await self.ssh.execute(cmd)
        return result.stdout if result.success else result.stderr
    
    async def get_jobs(
        self,
        user: Optional[str] = None,
        partition: Optional[str] = None,
        state: Optional[str] = None,
    ) -> list[JobInfo]:
        """Get list of jobs in the queue.
        
        Args:
            user: Filter by username.
            partition: Filter by partition.
            state: Filter by job state (PENDING, RUNNING, etc.).
            
        Returns:
            List of JobInfo objects.
        """
        # Format string for squeue
        format_str = "%i|%j|%u|%T|%P|%N|%D|%C|%m|%l|%M|%L|%V|%S|%r"
        # JobID|Name|User|State|Partition|Nodes|NumNodes|NumCPUs|Memory|TimeLimit|TimeUsed|TimeRemaining|SubmitTime|StartTime|Reason
        
        cmd = f"squeue -h -o '{format_str}'"
        if user:
            cmd += f" -u {user}"
        if partition:
            cmd += f" -p {partition}"
        if state:
            cmd += f" -t {state}"
        
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            logger.error(f"squeue failed: {result.stderr}")
            return []
        
        jobs = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 15:
                continue
            
            try:
                job_id = int(parts[0].split('_')[0])  # Handle array jobs
            except ValueError:
                continue
            
            jobs.append(JobInfo(
                job_id=job_id,
                job_name=parts[1],
                user=parts[2],
                state=parts[3],
                partition=parts[4],
                nodes=parts[5] if parts[5] else None,
                num_nodes=int(parts[6]) if parts[6].isdigit() else 1,
                num_cpus=int(parts[7]) if parts[7].isdigit() else 1,
                memory=parts[8] if parts[8] else None,
                time_limit=parts[9] if parts[9] else None,
                time_used=parts[10] if parts[10] else None,
                time_remaining=parts[11] if parts[11] else None,
                reason=parts[14] if len(parts) > 14 and parts[14] else None,
            ))
        
        return jobs
    
    async def get_job_details(self, job_id: int) -> Optional[JobInfo]:
        """Get detailed information about a specific job.
        
        Args:
            job_id: Slurm job ID.
            
        Returns:
            JobInfo object or None if not found.
        """
        cmd = f"scontrol show job {job_id}"
        result = await self.ssh.execute(cmd)
        
        if not result.success or "Invalid job id" in result.stderr:
            return None
        
        # Parse scontrol output
        info = {}
        for line in result.stdout.split('\n'):
            for part in line.split():
                if '=' in part:
                    key, value = part.split('=', 1)
                    info[key] = value
        
        if not info.get('JobId'):
            return None
        
        # Parse timestamps
        submit_time = None
        start_time = None
        end_time = None
        
        if info.get('SubmitTime') and info['SubmitTime'] != 'Unknown':
            try:
                submit_time = datetime.strptime(info['SubmitTime'], '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                pass
        
        if info.get('StartTime') and info['StartTime'] != 'Unknown':
            try:
                start_time = datetime.strptime(info['StartTime'], '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                pass
        
        if info.get('EndTime') and info['EndTime'] != 'Unknown':
            try:
                end_time = datetime.strptime(info['EndTime'], '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                pass
        
        # Parse GPU count from GRES
        num_gpus = 0
        if info.get('Gres'):
            gpus = _parse_gres(info['Gres'])
            num_gpus = sum(g.count for g in gpus)
        
        return JobInfo(
            job_id=int(info['JobId']),
            job_name=info.get('JobName', 'unknown'),
            user=info.get('UserId', '').split('(')[0],
            state=info.get('JobState', 'UNKNOWN'),
            partition=info.get('Partition', ''),
            nodes=info.get('NodeList'),
            num_nodes=int(info.get('NumNodes', 1)),
            num_cpus=int(info.get('NumCPUs', 1)),
            num_gpus=num_gpus,
            memory=info.get('MinMemoryNode'),
            time_limit=info.get('TimeLimit'),
            time_used=info.get('RunTime'),
            submit_time=submit_time,
            start_time=start_time,
            end_time=end_time,
            work_dir=info.get('WorkDir'),
            stdout_path=info.get('StdOut'),
            stderr_path=info.get('StdErr'),
            exit_code=int(info['ExitCode'].split(':')[0]) if info.get('ExitCode') else None,
            reason=info.get('Reason'),
            comment=info.get('Comment') if info.get('Comment') and info.get('Comment') != '(null)' else None,
            tres_per_node=info.get('TresPerNode') if info.get('TresPerNode') and info.get('TresPerNode') != '(null)' else None,
        )
    
    async def sbatch(self, script_path: str) -> int:
        """Submit a job script.
        
        Args:
            script_path: Path to the batch script on the remote host.
            
        Returns:
            Submitted job ID.
            
        Raises:
            SSHCommandError: If submission fails.
        """
        cmd = f"sbatch {script_path}"
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            raise SSHCommandError(f"sbatch failed: {result.stderr}")
        
        # Parse job ID from output like "Submitted batch job 12345"
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            return int(match.group(1))
        
        raise SSHCommandError(f"Could not parse job ID from sbatch output: {result.stdout}")
    
    async def submit_job(self, job: JobSubmission) -> int:
        """Submit a job with the given parameters.
        
        Args:
            job: JobSubmission object with job parameters.
            
        Returns:
            Submitted job ID.
        """
        # Generate the script
        script_content = job.generate_sbatch_script(
            default_partition=self.settings.default_partition,
            default_account=self.settings.default_account,
            default_mounts=self.settings.get_container_mounts(),
        )
        
        # Write script to temporary file
        import uuid
        script_name = f".slurm_mcp_job_{uuid.uuid4().hex[:8]}.sh"
        script_path = f"/tmp/{script_name}"
        
        await self.ssh.write_remote_file(script_content, script_path, mode=0o755)
        
        try:
            job_id = await self.sbatch(script_path)
            return job_id
        finally:
            # Clean up the temporary script
            try:
                await self.ssh.delete_file(script_path)
            except Exception:
                pass
    
    async def scancel(
        self,
        job_id: int,
        signal: Optional[str] = None,
    ) -> bool:
        """Cancel a job.
        
        Args:
            job_id: Job ID to cancel.
            signal: Optional signal to send (e.g., 'SIGTERM', 'SIGKILL').
            
        Returns:
            True if cancellation succeeded.
        """
        cmd = f"scancel {job_id}"
        if signal:
            cmd += f" --signal={signal}"
        
        result = await self.ssh.execute(cmd)
        return result.success
    
    async def scontrol_hold(self, job_id: int) -> bool:
        """Put a job on hold.
        
        Args:
            job_id: Job ID to hold.
            
        Returns:
            True if successful.
        """
        result = await self.ssh.execute(f"scontrol hold {job_id}")
        return result.success
    
    async def scontrol_release(self, job_id: int) -> bool:
        """Release a held job.
        
        Args:
            job_id: Job ID to release.
            
        Returns:
            True if successful.
        """
        result = await self.ssh.execute(f"scontrol release {job_id}")
        return result.success
    
    # =========================================================================
    # Job History / Accounting
    # =========================================================================
    
    async def sacct(
        self,
        job_id: Optional[int] = None,
        user: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        format_str: Optional[str] = None,
    ) -> str:
        """Run sacct command for job accounting.
        
        Args:
            job_id: Filter by job ID.
            user: Filter by username.
            start_time: Start time filter (e.g., '2024-01-01', 'now-7days').
            end_time: End time filter.
            format_str: Custom format string.
            
        Returns:
            Raw sacct output.
        """
        cmd = "sacct"
        
        if job_id:
            cmd += f" -j {job_id}"
        if user:
            cmd += f" -u {user}"
        if start_time:
            cmd += f" -S {start_time}"
        if end_time:
            cmd += f" -E {end_time}"
        if format_str:
            cmd += f" -o {format_str}"
        else:
            cmd += " -o JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,NCPUS"
        
        cmd += " --parsable2"
        
        result = await self.ssh.execute(cmd)
        return result.stdout if result.success else result.stderr
    
    # =========================================================================
    # Container Image Operations
    # =========================================================================
    
    async def list_container_images(
        self,
        directory: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[ContainerImage]:
        """List available container images (.sqsh files).
        
        Args:
            directory: Directory to search (uses config default if not specified).
            pattern: Glob pattern to filter images.
            
        Returns:
            List of ContainerImage objects.
        """
        search_dir = directory or self.settings.image_dir
        if not search_dir:
            return []
        
        # Build find command (quote path for spaces/special chars)
        quoted_dir = _quote_path(search_dir)
        if pattern:
            cmd = f"find {quoted_dir} -maxdepth 2 -name '{pattern}' -name '*.sqsh' -type f"
        else:
            cmd = f"find {quoted_dir} -maxdepth 2 -name '*.sqsh' -type f"
        
        cmd += " -printf '%p|%s|%T@\\n' 2>/dev/null | sort -t'|' -k3 -rn"
        
        result = await self.ssh.execute(cmd)
        
        if not result.success:
            return []
        
        images = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 3:
                continue
            
            path = parts[0]
            size = int(parts[1]) if parts[1].isdigit() else 0
            mtime = float(parts[2]) if parts[2] else 0
            
            images.append(ContainerImage(
                name=path.split('/')[-1],
                path=path,
                size_bytes=size,
                size_human=_bytes_to_human(size),
                modified_time=datetime.fromtimestamp(mtime),
            ))
        
        return images
    
    async def validate_container_image(self, image_path: str) -> bool:
        """Validate that a container image exists and is readable.
        
        Args:
            image_path: Path to the .sqsh image.
            
        Returns:
            True if image is valid.
        """
        quoted_path = _quote_path(image_path)
        result = await self.ssh.execute(f"test -r {quoted_path} && file {quoted_path}")
        
        if not result.success:
            return False
        
        # Check if it's a squashfs file
        return 'squashfs' in result.stdout.lower() or '.sqsh' in image_path
    
    # =========================================================================
    # Interactive Session Support
    # =========================================================================
    
    async def srun_command(
        self,
        command: str,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        nodes: int = 1,
        gpus_per_node: Optional[int] = None,
        time_limit: Optional[str] = None,
        container_image: Optional[str] = None,
        container_mounts: Optional[str] = None,
        no_container_mount_home: bool = True,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Execute a command via srun (one-shot execution).
        
        Args:
            command: Command to execute.
            partition: Partition to use.
            account: Account for billing.
            nodes: Number of nodes.
            gpus_per_node: GPUs per node.
            time_limit: Time limit.
            container_image: Container image path.
            container_mounts: Container mounts.
            no_container_mount_home: Don't mount home in container.
            working_directory: Working directory.
            timeout: Command timeout.
            
        Returns:
            CommandResult with output.
        """
        # Build srun command
        cmd = "srun"
        
        partition = partition or self.settings.interactive_partition
        account = account or self.settings.interactive_account
        time_limit = time_limit or self.settings.interactive_default_time
        gpus_per_node = gpus_per_node if gpus_per_node is not None else self.settings.interactive_default_gpus
        
        if account:
            cmd += f" -A {account}"
        cmd += f" -p {partition}"
        cmd += f" -N {nodes}"
        cmd += f" -t {time_limit}"
        
        if gpus_per_node:
            cmd += f" --gpus-per-node={gpus_per_node}"
        
        if container_image:
            cmd += f" --container-image={container_image}"
            
            mounts = container_mounts or self.settings.get_container_mounts()
            if mounts:
                cmd += f" --container-mounts={mounts}"
            
            if no_container_mount_home:
                cmd += " --no-container-mount-home"
        
        # Wrap the command (escape single quotes to avoid shell parsing issues)
        if working_directory:
            full_command = f"cd {working_directory} && {command}"
        else:
            full_command = command
        
        escaped_command = _escape_for_single_quotes(full_command)
        cmd += f" bash -c '{escaped_command}'"
        
        # Use longer timeout for interactive commands
        exec_timeout = timeout or max(300, self.settings.command_timeout)
        
        return await self.ssh.execute(cmd, timeout=exec_timeout)
    
    async def salloc(
        self,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        nodes: int = 1,
        gpus_per_node: Optional[int] = None,
        time_limit: Optional[str] = None,
        job_name: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> int:
        """Allocate resources and return job ID.

        Args:
            partition: Partition to use.
            account: Account for billing.
            nodes: Number of nodes.
            gpus_per_node: GPUs per node.
            time_limit: Time limit.
            job_name: Job name.
            comment: Slurm job comment string (used to persist session metadata).

        Returns:
            Allocated job ID.

        Raises:
            SSHCommandError: If allocation fails.
        """
        cmd = "salloc --no-shell"

        partition = partition or self.settings.interactive_partition
        account = account or self.settings.interactive_account
        time_limit = time_limit or self.settings.interactive_default_time
        gpus_per_node = gpus_per_node if gpus_per_node is not None else self.settings.interactive_default_gpus

        if account:
            cmd += f" -A {account}"
        cmd += f" -p {partition}"
        cmd += f" -N {nodes}"
        cmd += f" -t {time_limit}"

        if gpus_per_node:
            cmd += f" --gpus-per-node={gpus_per_node}"

        if job_name:
            cmd += f" -J {job_name}"

        if comment:
            # Single-quote the value; our encoding (base64 + ':') contains no single quotes.
            cmd += f" --comment='{comment}'"
        
        # salloc with --no-shell returns immediately with job ID
        result = await self.ssh.execute(cmd, timeout=120)
        
        if not result.success:
            raise SSHCommandError(f"salloc failed: {result.stderr}")
        
        # Parse job ID from output
        # Output format: "salloc: Granted job allocation 12345"
        match = re.search(r'Granted job allocation (\d+)', result.stderr + result.stdout)
        if match:
            return int(match.group(1))
        
        raise SSHCommandError(f"Could not parse job ID from salloc output: {result.stdout} {result.stderr}")
    
    async def srun_in_allocation(
        self,
        job_id: int,
        command: str,
        container_image: Optional[str] = None,
        container_mounts: Optional[str] = None,
        no_container_mount_home: bool = True,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Run a command in an existing allocation.
        
        Args:
            job_id: Job ID of the allocation.
            command: Command to execute.
            container_image: Container image path.
            container_mounts: Container mounts.
            no_container_mount_home: Don't mount home in container.
            working_directory: Working directory.
            timeout: Command timeout.
            
        Returns:
            CommandResult with output.
        """
        cmd = f"srun --jobid={job_id}"
        
        if container_image:
            cmd += f" --container-image={container_image}"
            
            mounts = container_mounts or self.settings.get_container_mounts()
            if mounts:
                cmd += f" --container-mounts={mounts}"
            
            if no_container_mount_home:
                cmd += " --no-container-mount-home"
        
        # Wrap the command (escape single quotes to avoid shell parsing issues)
        if working_directory:
            full_command = f"cd {working_directory} && {command}"
        else:
            full_command = command
        
        escaped_command = _escape_for_single_quotes(full_command)
        cmd += f" bash -c '{escaped_command}'"
        
        exec_timeout = timeout or max(300, self.settings.command_timeout)
        
        return await self.ssh.execute(cmd, timeout=exec_timeout)
