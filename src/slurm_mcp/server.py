"""Main MCP server for Slurm cluster management."""

import asyncio
import logging
from typing import Annotated, Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from slurm_mcp.config import Settings, get_settings
from slurm_mcp.directories import DirectoryManager, get_directory_manager
from slurm_mcp.interactive import InteractiveSessionManager, get_session_manager
from slurm_mcp.models import InteractiveProfile, JobSubmission
from slurm_mcp.profiles import ProfileManager, get_profile_manager
from slurm_mcp.slurm_commands import SlurmCommands
from slurm_mcp.ssh_client import SSHClient, SSHCommandError, get_ssh_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP(
    "slurm-mcp",
    instructions="MCP server for remote Slurm cluster management via SSH",
)

# Global instances (initialized on startup)
_settings: Optional[Settings] = None
_ssh: Optional[SSHClient] = None
_slurm: Optional[SlurmCommands] = None
_sessions: Optional[InteractiveSessionManager] = None
_profiles: Optional[ProfileManager] = None
_directories: Optional[DirectoryManager] = None


async def get_instances():
    """Get or initialize global instances."""
    global _settings, _ssh, _slurm, _sessions, _profiles, _directories
    
    if _settings is None:
        _settings = get_settings()
    
    if _ssh is None:
        _ssh = get_ssh_client(_settings)
        await _ssh.connect()
    
    if _slurm is None:
        _slurm = SlurmCommands(_ssh, _settings)
    
    if _sessions is None:
        _sessions = get_session_manager(_ssh, _slurm, _settings)
    
    if _profiles is None:
        _profiles = get_profile_manager(_ssh, _settings)
    
    if _directories is None:
        _directories = get_directory_manager(_ssh, _settings)
    
    return _settings, _ssh, _slurm, _sessions, _profiles, _directories


# =============================================================================
# Cluster Status Tools
# =============================================================================

@mcp.tool()
async def get_cluster_status(
    partition: Annotated[Optional[str], Field(description="Filter by partition name")] = None,
) -> str:
    """Get the current status of the Slurm cluster including partitions and node availability."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        partitions = await slurm.get_partitions()
        
        if partition:
            partitions = [p for p in partitions if p.name == partition]
        
        if not partitions:
            return "No partitions found."
        
        lines = ["Cluster Status:", ""]
        
        for p in partitions:
            default_marker = " (default)" if p.default else ""
            gpu_info = f", GPUs: {p.available_gpus}/{p.total_gpus}" if p.has_gpus else ""
            lines.append(
                f"  {p.name}{default_marker}: {p.state}\n"
                f"    Nodes: {p.available_nodes}/{p.total_nodes} available\n"
                f"    CPUs: {p.available_cpus}/{p.total_cpus} available{gpu_info}\n"
                f"    Max Time: {p.max_time or 'unlimited'}"
            )
            if p.gpu_types:
                lines.append(f"    GPU Types: {', '.join(p.gpu_types)}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get cluster status: {e}")


@mcp.tool()
async def get_partition_info(
    partition_name: Annotated[Optional[str], Field(description="Specific partition name, or None for all")] = None,
) -> str:
    """Get detailed information about cluster partitions."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        partitions = await slurm.get_partitions()
        
        if partition_name:
            partitions = [p for p in partitions if p.name == partition_name]
            if not partitions:
                return f"Partition '{partition_name}' not found."
        
        lines = []
        for p in partitions:
            lines.append(f"Partition: {p.name}")
            lines.append(f"  State: {p.state}")
            lines.append(f"  Default: {p.default}")
            lines.append(f"  Total Nodes: {p.total_nodes}")
            lines.append(f"  Available Nodes: {p.available_nodes}")
            lines.append(f"  Total CPUs: {p.total_cpus}")
            lines.append(f"  Available CPUs: {p.available_cpus}")
            lines.append(f"  Max Time: {p.max_time or 'unlimited'}")
            lines.append(f"  Has GPUs: {p.has_gpus}")
            if p.has_gpus:
                lines.append(f"  GPU Types: {', '.join(p.gpu_types) if p.gpu_types else 'unknown'}")
                lines.append(f"  Total GPUs: {p.total_gpus}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get partition info: {e}")


@mcp.tool()
async def get_node_info(
    node_name: Annotated[Optional[str], Field(description="Specific node name")] = None,
    partition: Annotated[Optional[str], Field(description="Filter by partition")] = None,
    state: Annotated[Optional[str], Field(description="Filter by state (idle, allocated, down, etc.)")] = None,
) -> str:
    """Get information about cluster nodes."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        nodes = await slurm.get_nodes(partition=partition, state=state)
        
        if node_name:
            nodes = [n for n in nodes if n.node_name == node_name]
        
        if not nodes:
            return "No nodes found matching criteria."
        
        lines = [f"Found {len(nodes)} nodes:", ""]
        
        for n in nodes:
            gpu_info = ""
            if n.gpus:
                gpu_strs = [f"{g.gpu_type}:{g.count}" for g in n.gpus]
                gpu_info = f", GPUs: {', '.join(gpu_strs)}"
            
            lines.append(
                f"  {n.node_name}: {n.state}\n"
                f"    CPUs: {n.cpus_available}/{n.cpus_total} available\n"
                f"    Memory: {n.memory_available_mb}MB/{n.memory_total_mb}MB available{gpu_info}\n"
                f"    Partitions: {', '.join(n.partitions)}"
            )
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get node info: {e}")


@mcp.tool()
async def get_gpu_info(
    partition: Annotated[Optional[str], Field(description="Filter by partition")] = None,
    gpu_type: Annotated[Optional[str], Field(description="Filter by GPU type (e.g., 'a100', 'v100')")] = None,
) -> str:
    """Get information about available GPU resources in the cluster."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        gpu_info = await slurm.get_gpu_info(partition=partition)
        
        lines = ["GPU Information:", ""]
        lines.append(f"Total GPUs: {gpu_info['total_gpus']}")
        lines.append(f"Allocated: {gpu_info['allocated_gpus']}")
        lines.append(f"Available: {gpu_info['available_gpus']}")
        lines.append("")
        
        if gpu_info["by_type"]:
            lines.append("By GPU Type:")
            for gtype, stats in gpu_info["by_type"].items():
                if gpu_type and gtype != gpu_type:
                    continue
                lines.append(f"  {gtype}: {stats['available']}/{stats['total']} available")
        
        if gpu_info["by_partition"]:
            lines.append("")
            lines.append("By Partition:")
            for part, stats in gpu_info["by_partition"].items():
                lines.append(f"  {part}: {stats['available']}/{stats['total']} available ({', '.join(stats['types'])})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get GPU info: {e}")


@mcp.tool()
async def get_gpu_availability(
    partition: Annotated[Optional[str], Field(description="Filter by partition")] = None,
    gpu_type: Annotated[Optional[str], Field(description="Filter by GPU type")] = None,
    min_gpus: Annotated[Optional[int], Field(description="Minimum number of GPUs needed")] = None,
) -> str:
    """Check current GPU availability - how many GPUs are free vs allocated."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        gpu_info = await slurm.get_gpu_info(partition=partition)
        
        available = gpu_info["available_gpus"]
        total = gpu_info["total_gpus"]
        
        if gpu_type and gpu_type in gpu_info["by_type"]:
            available = gpu_info["by_type"][gpu_type]["available"]
            total = gpu_info["by_type"][gpu_type]["total"]
        
        lines = [f"GPU Availability: {available}/{total} GPUs free"]
        
        if min_gpus:
            if available >= min_gpus:
                lines.append(f"✓ {min_gpus} GPUs are available")
            else:
                lines.append(f"✗ Only {available} GPUs available, need {min_gpus}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to check GPU availability: {e}")


# =============================================================================
# Job Management Tools
# =============================================================================

@mcp.tool()
async def list_jobs(
    user: Annotated[Optional[str], Field(description="Filter by username")] = None,
    partition: Annotated[Optional[str], Field(description="Filter by partition")] = None,
    state: Annotated[Optional[str], Field(description="Filter by job state (PENDING, RUNNING, etc.)")] = None,
) -> str:
    """List jobs in the Slurm queue."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        jobs = await slurm.get_jobs(user=user, partition=partition, state=state)
        
        if not jobs:
            return "No jobs found matching criteria."
        
        lines = [f"Found {len(jobs)} jobs:", ""]
        
        for j in jobs:
            gpu_info = f", GPUs: {j.num_gpus}" if j.num_gpus else ""
            lines.append(
                f"  Job {j.job_id}: {j.job_name}\n"
                f"    User: {j.user}, State: {j.state}\n"
                f"    Partition: {j.partition}, Nodes: {j.num_nodes}, CPUs: {j.num_cpus}{gpu_info}\n"
                f"    Time: {j.time_used or 'N/A'} / {j.time_limit or 'N/A'}"
            )
            if j.reason and j.state == "PENDING":
                lines.append(f"    Reason: {j.reason}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list jobs: {e}")


@mcp.tool()
async def get_job_details(
    job_id: Annotated[int, Field(description="The Slurm job ID")],
) -> str:
    """Get detailed information about a specific job."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        job = await slurm.get_job_details(job_id)
        
        if not job:
            return f"Job {job_id} not found."
        
        lines = [
            f"Job {job.job_id}: {job.job_name}",
            f"  User: {job.user}",
            f"  State: {job.state}",
            f"  Partition: {job.partition}",
            f"  Nodes: {job.nodes or 'N/A'} ({job.num_nodes} requested)",
            f"  CPUs: {job.num_cpus}",
            f"  GPUs: {job.num_gpus}",
            f"  Memory: {job.memory or 'N/A'}",
            f"  Time Limit: {job.time_limit or 'N/A'}",
            f"  Time Used: {job.time_used or 'N/A'}",
            f"  Working Dir: {job.work_dir or 'N/A'}",
            f"  Stdout: {job.stdout_path or 'N/A'}",
            f"  Stderr: {job.stderr_path or 'N/A'}",
        ]
        
        if job.submit_time:
            lines.append(f"  Submitted: {job.submit_time}")
        if job.start_time:
            lines.append(f"  Started: {job.start_time}")
        if job.end_time:
            lines.append(f"  Ended: {job.end_time}")
        if job.exit_code is not None:
            lines.append(f"  Exit Code: {job.exit_code}")
        if job.reason:
            lines.append(f"  Reason: {job.reason}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get job details: {e}")


@mcp.tool()
async def submit_job(
    script_content: Annotated[str, Field(description="The SBATCH script content (commands to run)")],
    job_name: Annotated[Optional[str], Field(description="Job name")] = None,
    partition: Annotated[Optional[str], Field(description="Partition to submit to")] = None,
    account: Annotated[Optional[str], Field(description="Account/project for billing")] = None,
    nodes: Annotated[Optional[int], Field(description="Number of nodes")] = None,
    ntasks: Annotated[Optional[int], Field(description="Number of tasks")] = None,
    cpus_per_task: Annotated[Optional[int], Field(description="CPUs per task")] = None,
    memory: Annotated[Optional[str], Field(description="Memory per node (e.g., '4G', '4000M')")] = None,
    time_limit: Annotated[Optional[str], Field(description="Time limit (e.g., '1:00:00', '1-00:00:00')")] = None,
    output_file: Annotated[Optional[str], Field(description="Output file path")] = None,
    error_file: Annotated[Optional[str], Field(description="Error file path")] = None,
    working_directory: Annotated[Optional[str], Field(description="Working directory on cluster")] = None,
    gpus: Annotated[Optional[int], Field(description="Number of GPUs per node")] = None,
    gpus_per_task: Annotated[Optional[int], Field(description="Number of GPUs per task")] = None,
    gpu_type: Annotated[Optional[str], Field(description="Specific GPU type (e.g., 'a100', 'v100')")] = None,
    container_image: Annotated[Optional[str], Field(description="Path to container .sqsh image file")] = None,
    container_mounts: Annotated[Optional[str], Field(description="Container bind mounts")] = None,
    container_workdir: Annotated[Optional[str], Field(description="Working directory inside container")] = None,
) -> str:
    """Submit a batch job to the Slurm cluster. Returns the job ID on success."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        job = JobSubmission(
            script_content=script_content,
            job_name=job_name,
            partition=partition,
            account=account,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            memory=memory,
            time_limit=time_limit,
            output_file=output_file,
            error_file=error_file,
            working_directory=working_directory,
            gpus=gpus,
            gpus_per_task=gpus_per_task,
            gpu_type=gpu_type,
            container_image=container_image,
            container_mounts=container_mounts,
            container_workdir=container_workdir,
        )
        
        job_id = await slurm.submit_job(job)
        
        return f"Job submitted successfully. Job ID: {job_id}"
        
    except SSHCommandError as e:
        raise ToolError(f"Failed to submit job: {e}")
    except Exception as e:
        raise ToolError(f"Failed to submit job: {e}")


@mcp.tool()
async def cancel_job(
    job_id: Annotated[int, Field(description="The Slurm job ID to cancel")],
    signal: Annotated[Optional[str], Field(description="Signal to send (e.g., 'SIGTERM', 'SIGKILL')")] = None,
) -> str:
    """Cancel a running or pending job."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        success = await slurm.scancel(job_id, signal=signal)
        
        if success:
            return f"Job {job_id} cancelled successfully."
        else:
            return f"Failed to cancel job {job_id}."
        
    except Exception as e:
        raise ToolError(f"Failed to cancel job: {e}")


@mcp.tool()
async def hold_job(
    job_id: Annotated[int, Field(description="The Slurm job ID to hold")],
) -> str:
    """Put a pending job on hold."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        success = await slurm.scontrol_hold(job_id)
        
        if success:
            return f"Job {job_id} put on hold."
        else:
            return f"Failed to hold job {job_id}."
        
    except Exception as e:
        raise ToolError(f"Failed to hold job: {e}")


@mcp.tool()
async def release_job(
    job_id: Annotated[int, Field(description="The Slurm job ID to release")],
) -> str:
    """Release a held job."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        success = await slurm.scontrol_release(job_id)
        
        if success:
            return f"Job {job_id} released."
        else:
            return f"Failed to release job {job_id}."
        
    except Exception as e:
        raise ToolError(f"Failed to release job: {e}")


@mcp.tool()
async def get_job_history(
    job_id: Annotated[Optional[int], Field(description="Specific job ID")] = None,
    user: Annotated[Optional[str], Field(description="Filter by username")] = None,
    start_time: Annotated[Optional[str], Field(description="Start time (e.g., '2024-01-01', 'now-7days')")] = None,
    end_time: Annotated[Optional[str], Field(description="End time")] = None,
) -> str:
    """Get job accounting/history information."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        output = await slurm.sacct(
            job_id=job_id,
            user=user,
            start_time=start_time,
            end_time=end_time,
        )
        
        return output if output else "No job history found."
        
    except Exception as e:
        raise ToolError(f"Failed to get job history: {e}")


# =============================================================================
# Container Image Tools
# =============================================================================

@mcp.tool()
async def list_container_images(
    image_dir: Annotated[Optional[str], Field(description="Directory to search for .sqsh images")] = None,
    pattern: Annotated[Optional[str], Field(description="Filter images by name pattern (e.g., 'pytorch*')")] = None,
) -> str:
    """List available container images (.sqsh files) for Pyxis/enroot."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        images = await slurm.list_container_images(directory=image_dir, pattern=pattern)
        
        if not images:
            return "No container images found."
        
        lines = [f"Found {len(images)} container images:", ""]
        
        for img in images:
            lines.append(f"  {img.name}")
            lines.append(f"    Path: {img.path}")
            lines.append(f"    Size: {img.size_human}")
            lines.append(f"    Modified: {img.modified_time.strftime('%Y-%m-%d %H:%M')}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list container images: {e}")


@mcp.tool()
async def validate_container_image(
    image_path: Annotated[str, Field(description="Path to the .sqsh container image")],
) -> str:
    """Validate that a container image exists and is readable."""
    try:
        _, _, slurm, _, _, _ = await get_instances()
        
        is_valid = await slurm.validate_container_image(image_path)
        
        if is_valid:
            return f"Container image is valid: {image_path}"
        else:
            return f"Container image is invalid or not found: {image_path}"
        
    except Exception as e:
        raise ToolError(f"Failed to validate container image: {e}")


# =============================================================================
# Interactive Session Tools
# =============================================================================

@mcp.tool()
async def run_interactive_command(
    command: Annotated[str, Field(description="Command to execute")],
    partition: Annotated[Optional[str], Field(description="Partition (default: interactive)")] = None,
    account: Annotated[Optional[str], Field(description="Account/project for billing")] = None,
    nodes: Annotated[int, Field(description="Number of nodes")] = 1,
    gpus_per_node: Annotated[Optional[int], Field(description="GPUs per node")] = None,
    time_limit: Annotated[Optional[str], Field(description="Time limit (e.g., '4:00:00')")] = None,
    container_image: Annotated[Optional[str], Field(description="Container .sqsh image path")] = None,
    container_mounts: Annotated[Optional[str], Field(description="Container mounts")] = None,
    working_directory: Annotated[Optional[str], Field(description="Working directory for command")] = None,
    timeout: Annotated[Optional[int], Field(description="Command timeout in seconds")] = None,
) -> str:
    """Execute a single command with interactive-partition resources (one-shot allocation)."""
    try:
        _, _, _, sessions, _, _ = await get_instances()
        
        result = await sessions.run_command(
            command=command,
            partition=partition,
            account=account,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            container_mounts=container_mounts,
            working_directory=working_directory,
            timeout=timeout,
        )
        
        output = result.stdout if result.stdout else result.stderr
        
        if result.success:
            return output if output else "Command completed successfully (no output)."
        else:
            return f"Command failed (exit code {result.return_code}):\n{output}"
        
    except Exception as e:
        raise ToolError(f"Failed to run interactive command: {e}")


@mcp.tool()
async def start_interactive_session(
    session_name: Annotated[Optional[str], Field(description="Name for this session")] = None,
    partition: Annotated[Optional[str], Field(description="Partition (default: interactive)")] = None,
    account: Annotated[Optional[str], Field(description="Account/project for billing")] = None,
    nodes: Annotated[int, Field(description="Number of nodes")] = 1,
    gpus_per_node: Annotated[Optional[int], Field(description="GPUs per node")] = None,
    time_limit: Annotated[Optional[str], Field(description="Time limit (e.g., '4:00:00')")] = None,
    container_image: Annotated[Optional[str], Field(description="Container .sqsh image path")] = None,
    container_mounts: Annotated[Optional[str], Field(description="Container mounts")] = None,
) -> str:
    """Start a persistent interactive session using salloc."""
    try:
        _, _, _, sessions, _, _ = await get_instances()
        
        session = await sessions.start_session(
            session_name=session_name,
            partition=partition,
            account=account,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            container_mounts=container_mounts,
        )
        
        name_str = f" '{session.session_name}'" if session.session_name else ""
        return (
            f"Session{name_str} started successfully.\n"
            f"  Session ID: {session.session_id}\n"
            f"  Job ID: {session.job_id}\n"
            f"  Partition: {session.partition}\n"
            f"  Nodes: {session.nodes}\n"
            f"  GPUs/Node: {session.gpus_per_node or 0}\n"
            f"  Time Limit: {session.time_limit}\n"
            f"  Node List: {session.node_list or 'pending'}\n\n"
            f"Use exec_in_session(session_id='{session.session_id}', command='...') to run commands."
        )
        
    except Exception as e:
        raise ToolError(f"Failed to start interactive session: {e}")


@mcp.tool()
async def exec_in_session(
    session_id: Annotated[str, Field(description="Session ID from start_interactive_session")],
    command: Annotated[str, Field(description="Command to execute")],
    working_directory: Annotated[Optional[str], Field(description="Working directory")] = None,
    timeout: Annotated[Optional[int], Field(description="Command timeout in seconds")] = None,
) -> str:
    """Execute a command in an existing interactive session."""
    try:
        _, _, _, sessions, _, _ = await get_instances()
        
        result = await sessions.exec_command(
            session_id=session_id,
            command=command,
            working_directory=working_directory,
            timeout=timeout,
        )
        
        output = result.stdout if result.stdout else result.stderr
        
        if result.success:
            return output if output else "Command completed successfully (no output)."
        else:
            return f"Command failed (exit code {result.return_code}):\n{output}"
        
    except ValueError as e:
        raise ToolError(str(e))
    except Exception as e:
        raise ToolError(f"Failed to execute command in session: {e}")


@mcp.tool()
async def list_interactive_sessions() -> str:
    """List all active interactive sessions managed by this MCP server."""
    try:
        _, _, _, sessions, _, _ = await get_instances()
        
        active_sessions = await sessions.list_sessions()
        
        if not active_sessions:
            return "No active interactive sessions."
        
        lines = [f"Active Sessions ({len(active_sessions)}):", ""]
        
        for s in active_sessions:
            name_str = f" ({s.session_name})" if s.session_name else ""
            lines.append(f"  Session {s.session_id}{name_str}")
            lines.append(f"    Job ID: {s.job_id}")
            lines.append(f"    Partition: {s.partition}")
            lines.append(f"    Nodes: {s.nodes}, GPUs/Node: {s.gpus_per_node or 0}")
            lines.append(f"    Time Remaining: {s.time_remaining or 'unknown'}")
            lines.append(f"    Status: {s.status}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list sessions: {e}")


@mcp.tool()
async def end_interactive_session(
    session_id: Annotated[str, Field(description="Session ID to terminate")],
) -> str:
    """End an interactive session and release its resources."""
    try:
        _, _, _, sessions, _, _ = await get_instances()
        
        success = await sessions.end_session(session_id)
        
        if success:
            return f"Session {session_id} ended successfully."
        else:
            return f"Session {session_id} not found or already ended."
        
    except Exception as e:
        raise ToolError(f"Failed to end session: {e}")


@mcp.tool()
async def get_interactive_session_info(
    session_id: Annotated[str, Field(description="Session ID")],
) -> str:
    """Get detailed information about an interactive session."""
    try:
        _, _, _, sessions, _, _ = await get_instances()
        
        session = await sessions.get_session(session_id)
        
        if not session:
            return f"Session {session_id} not found."
        
        lines = [
            f"Session: {session.session_id}",
            f"  Name: {session.session_name or 'N/A'}",
            f"  Job ID: {session.job_id}",
            f"  Partition: {session.partition}",
            f"  Nodes: {session.nodes}",
            f"  GPUs/Node: {session.gpus_per_node or 0}",
            f"  Container: {session.container_image or 'None'}",
            f"  Start Time: {session.start_time}",
            f"  Time Limit: {session.time_limit}",
            f"  Time Remaining: {session.time_remaining or 'unknown'}",
            f"  Status: {session.status}",
            f"  Node List: {session.node_list or 'N/A'}",
        ]
        
        if session.last_command_time:
            lines.append(f"  Last Command: {session.last_command_time}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get session info: {e}")


# =============================================================================
# Profile Management Tools
# =============================================================================

@mcp.tool()
async def save_interactive_profile(
    profile_name: Annotated[str, Field(description="Name for this profile")],
    description: Annotated[Optional[str], Field(description="Profile description")] = None,
    partition: Annotated[Optional[str], Field(description="Partition")] = None,
    account: Annotated[Optional[str], Field(description="Account")] = None,
    nodes: Annotated[int, Field(description="Number of nodes")] = 1,
    gpus_per_node: Annotated[Optional[int], Field(description="GPUs per node")] = None,
    time_limit: Annotated[Optional[str], Field(description="Time limit")] = None,
    container_image: Annotated[Optional[str], Field(description="Container image")] = None,
    container_mounts: Annotated[Optional[str], Field(description="Container mounts")] = None,
) -> str:
    """Save an interactive session profile for quick reuse."""
    try:
        _, _, _, _, profiles, _ = await get_instances()
        
        profile = InteractiveProfile(
            name=profile_name,
            description=description,
            partition=partition,
            account=account,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            container_mounts=container_mounts,
        )
        
        await profiles.save_profile(profile)
        
        return f"Profile '{profile_name}' saved successfully."
        
    except Exception as e:
        raise ToolError(f"Failed to save profile: {e}")


@mcp.tool()
async def list_interactive_profiles() -> str:
    """List saved interactive session profiles."""
    try:
        _, _, _, _, profiles, _ = await get_instances()
        
        profile_list = await profiles.list_profiles()
        
        if not profile_list:
            return "No profiles saved."
        
        lines = [f"Saved Profiles ({len(profile_list)}):", ""]
        
        for p in profile_list:
            lines.append(f"  {p.name}")
            if p.description:
                lines.append(f"    Description: {p.description}")
            lines.append(f"    Partition: {p.partition or 'default'}")
            lines.append(f"    Nodes: {p.nodes}, GPUs/Node: {p.gpus_per_node or 0}")
            lines.append(f"    Time Limit: {p.time_limit or 'default'}")
            if p.container_image:
                lines.append(f"    Container: {p.container_image}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list profiles: {e}")


@mcp.tool()
async def start_session_from_profile(
    profile_name: Annotated[str, Field(description="Profile name to use")],
    session_name: Annotated[Optional[str], Field(description="Optional session name")] = None,
    time_limit: Annotated[Optional[str], Field(description="Override time limit")] = None,
) -> str:
    """Start an interactive session using a saved profile."""
    try:
        _, _, _, sessions, profiles, _ = await get_instances()
        
        profile = await profiles.get_profile(profile_name)
        
        if not profile:
            return f"Profile '{profile_name}' not found."
        
        session = await sessions.start_session(
            session_name=session_name or profile_name,
            partition=profile.partition,
            account=profile.account,
            nodes=profile.nodes,
            gpus_per_node=profile.gpus_per_node,
            time_limit=time_limit or profile.time_limit,
            container_image=profile.container_image,
            container_mounts=profile.container_mounts,
        )
        
        return (
            f"Session started from profile '{profile_name}'.\n"
            f"  Session ID: {session.session_id}\n"
            f"  Job ID: {session.job_id}\n\n"
            f"Use exec_in_session(session_id='{session.session_id}', command='...') to run commands."
        )
        
    except Exception as e:
        raise ToolError(f"Failed to start session from profile: {e}")


# =============================================================================
# Directory Management Tools
# =============================================================================

@mcp.tool()
async def get_cluster_directories() -> str:
    """Get the configured cluster directory structure."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        dirs = directories.get_cluster_directories()
        
        lines = [
            "Cluster Directory Structure:",
            f"  User Root: {dirs.user_root}",
            "",
            "Configured Directories (host path -> container mount):",
            f"  Datasets:       {dirs.datasets} -> /datasets",
            f"  Results:        {dirs.results} -> /results",
            f"  Models:         {dirs.models} -> /models",
            f"  Logs:           {dirs.logs} -> /logs",
        ]
        
        if dirs.projects:
            lines.append(f"  Projects:       {dirs.projects} -> /projects")
        if dirs.container_root:
            lines.append(f"  Container Root: {dirs.container_root} -> /root")
        if dirs.home:
            lines.append(f"  Home:           {dirs.home} -> /home")
        if dirs.gpfs_root:
            lines.append(f"  GPFS Root:      {dirs.gpfs_root} -> /lustre")
        if dirs.images:
            lines.append(f"  Images:         {dirs.images}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get cluster directories: {e}")


@mcp.tool()
async def list_directory(
    path: Annotated[str, Field(description="Directory path to list")] = "",
    directory_type: Annotated[Optional[str], Field(description="Directory type: 'datasets', 'results', 'models', 'logs', 'projects'")] = None,
    pattern: Annotated[Optional[str], Field(description="Filter by glob pattern")] = None,
) -> str:
    """List contents of a directory on the cluster."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        listing = await directories.list_directory(
            path=path,
            directory_type=directory_type,
            pattern=pattern,
        )
        
        lines = [f"Directory: {listing.path}", f"Total: {listing.total_items} items ({listing.total_size_human})", ""]
        
        if listing.subdirs:
            lines.append("Directories:")
            for d in listing.subdirs:
                lines.append(f"  {d.name}/")
            lines.append("")
        
        if listing.files:
            lines.append("Files:")
            for f in listing.files:
                lines.append(f"  {f.name} ({f.size_human}, {f.modified_time.strftime('%Y-%m-%d %H:%M')})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list directory: {e}")


@mcp.tool()
async def list_datasets(
    pattern: Annotated[Optional[str], Field(description="Filter by pattern")] = None,
) -> str:
    """List available datasets in the datasets directory."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        items = await directories.list_datasets(pattern=pattern)
        
        if not items:
            return "No datasets found."
        
        lines = [f"Datasets ({len(items)}):", ""]
        for item in items:
            type_str = "/" if item.is_dir else ""
            lines.append(f"  {item.name}{type_str} ({item.size_human})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list datasets: {e}")


@mcp.tool()
async def list_model_checkpoints(
    model_name: Annotated[Optional[str], Field(description="Filter by model name/directory")] = None,
    pattern: Annotated[Optional[str], Field(description="Filter by pattern")] = None,
) -> str:
    """List model checkpoints in the models directory."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        items = await directories.list_model_checkpoints(model_name=model_name, pattern=pattern)
        
        if not items:
            return "No checkpoints found."
        
        lines = [f"Model Checkpoints ({len(items)}):", ""]
        for item in items:
            type_str = "/" if item.is_dir else ""
            lines.append(f"  {item.name}{type_str} ({item.size_human}, {item.modified_time.strftime('%Y-%m-%d %H:%M')})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list checkpoints: {e}")


@mcp.tool()
async def list_job_logs(
    job_id: Annotated[Optional[int], Field(description="Filter by job ID")] = None,
    job_name: Annotated[Optional[str], Field(description="Filter by job name pattern")] = None,
    recent: Annotated[Optional[int], Field(description="Only show N most recent logs")] = None,
) -> str:
    """List job log files in the logs directory."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        items = await directories.list_job_logs(job_id=job_id, job_name=job_name, recent=recent)
        
        if not items:
            return "No log files found."
        
        lines = [f"Job Logs ({len(items)}):", ""]
        for item in items:
            lines.append(f"  {item.name} ({item.size_human}, {item.modified_time.strftime('%Y-%m-%d %H:%M')})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to list job logs: {e}")


@mcp.tool()
async def read_file(
    path: Annotated[str, Field(description="File path")],
    directory_type: Annotated[Optional[str], Field(description="Base directory type")] = None,
    tail_lines: Annotated[Optional[int], Field(description="Only read last N lines")] = None,
    head_lines: Annotated[Optional[int], Field(description="Only read first N lines")] = None,
) -> str:
    """Read contents of a file on the cluster."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        content = await directories.read_file(
            path=path,
            directory_type=directory_type,
            tail_lines=tail_lines,
            head_lines=head_lines,
        )
        
        return content if content else "(empty file)"
        
    except Exception as e:
        raise ToolError(f"Failed to read file: {e}")


@mcp.tool()
async def write_file(
    path: Annotated[str, Field(description="File path")],
    content: Annotated[str, Field(description="File content")],
    directory_type: Annotated[Optional[str], Field(description="Base directory type")] = None,
    append: Annotated[bool, Field(description="Append instead of overwrite")] = False,
) -> str:
    """Write content to a file on the cluster."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        await directories.write_file(
            path=path,
            content=content,
            directory_type=directory_type,
            append=append,
        )
        
        action = "Appended to" if append else "Wrote"
        return f"{action} file successfully."
        
    except Exception as e:
        raise ToolError(f"Failed to write file: {e}")


@mcp.tool()
async def find_files(
    pattern: Annotated[str, Field(description="Search pattern (glob)")],
    directory_type: Annotated[Optional[str], Field(description="Directory to search in")] = None,
    path: Annotated[Optional[str], Field(description="Specific path to search in")] = None,
    file_type: Annotated[Optional[str], Field(description="Filter by type: 'file', 'dir', 'link'")] = None,
    min_size: Annotated[Optional[str], Field(description="Minimum size (e.g., '1G', '100M')")] = None,
    max_age: Annotated[Optional[str], Field(description="Maximum age (e.g., '7d', '24h')")] = None,
) -> str:
    """Search for files across cluster directories."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        items = await directories.find_files(
            pattern=pattern,
            directory_type=directory_type,
            path=path,
            file_type=file_type,
            min_size=min_size,
            max_age=max_age,
        )
        
        if not items:
            return "No files found matching criteria."
        
        lines = [f"Found {len(items)} files:", ""]
        for item in items:
            type_str = "/" if item.is_dir else ""
            lines.append(f"  {item.path}{type_str} ({item.size_human})")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to find files: {e}")


@mcp.tool()
async def delete_file(
    path: Annotated[str, Field(description="File or directory path")],
    directory_type: Annotated[Optional[str], Field(description="Base directory type")] = None,
    recursive: Annotated[bool, Field(description="Delete directories recursively")] = False,
    confirm: Annotated[bool, Field(description="Confirm deletion (must be True)")] = False,
) -> str:
    """Delete a file or directory on the cluster. Requires confirm=True for safety."""
    if not confirm:
        return "Deletion not confirmed. Set confirm=True to delete."
    
    try:
        _, _, _, _, _, directories = await get_instances()
        
        await directories.delete_file(
            path=path,
            directory_type=directory_type,
            recursive=recursive,
        )
        
        return f"Deleted: {path}"
        
    except Exception as e:
        raise ToolError(f"Failed to delete: {e}")


@mcp.tool()
async def download_file(
    path: Annotated[str, Field(description="Remote file or directory path on the cluster")],
    directory_type: Annotated[Optional[str], Field(description="Base directory type for resolving 'path' (e.g., 'results', 'logs')")] = None,
    local_path: Annotated[Optional[str], Field(description="Local destination path. Defaults to ~/Downloads/<filename>. If a directory, the file is placed inside with its remote name.")] = None,
    recursive: Annotated[bool, Field(description="Required when downloading a directory")] = False,
    overwrite: Annotated[bool, Field(description="Overwrite if local path already exists")] = False,
) -> str:
    """Download a file (or directory, with recursive=True) from the cluster to the local machine.

    By default the file is saved to the user's ~/Downloads folder.
    """
    try:
        _, _, _, _, _, directories = await get_instances()

        result = await directories.download(
            path=path,
            directory_type=directory_type,
            local_path=local_path,
            recursive=recursive,
            overwrite=overwrite,
        )

        from slurm_mcp.directories import _bytes_to_human
        size_human = _bytes_to_human(result["bytes"])

        return (
            f"Downloaded {result['files']} file(s) ({size_human})\n"
            f"  From: {result['remote_path']}\n"
            f"  To:   {result['local_path']}"
        )

    except FileExistsError as e:
        raise ToolError(str(e))
    except Exception as e:
        raise ToolError(f"Failed to download: {e}")


@mcp.tool()
async def get_disk_usage(
    directory_type: Annotated[Optional[str], Field(description="Check specific directory type")] = None,
    path: Annotated[Optional[str], Field(description="Check specific path")] = None,
) -> str:
    """Get disk usage for cluster directories."""
    try:
        _, _, _, _, _, directories = await get_instances()
        
        usage = await directories.get_disk_usage(
            directory_type=directory_type,
            path=path,
        )
        
        lines = ["Disk Usage:", ""]
        
        for name, stats in usage.items():
            if name == "filesystem":
                continue
            lines.append(f"  {name}: {stats['size_human']}")
            lines.append(f"    Path: {stats['path']}")
        
        if "filesystem" in usage:
            fs = usage["filesystem"]
            lines.append("")
            lines.append("Filesystem:")
            lines.append(f"  Total: {fs['total_human']}")
            lines.append(f"  Used: {fs['used_human']}")
            lines.append(f"  Available: {fs['available_human']}")
        
        return "\n".join(lines)
        
    except Exception as e:
        raise ToolError(f"Failed to get disk usage: {e}")


@mcp.tool()
async def run_shell_command(
    command: Annotated[str, Field(description="Shell command to execute")],
    working_directory: Annotated[Optional[str], Field(description="Working directory")] = None,
    timeout: Annotated[Optional[int], Field(description="Timeout in seconds")] = None,
) -> str:
    """Execute a shell command on the Slurm login node. Use with caution."""
    try:
        _, ssh, _, _, _, _ = await get_instances()
        
        result = await ssh.execute(
            command=command,
            working_directory=working_directory,
            timeout=timeout,
        )
        
        output = result.stdout if result.stdout else result.stderr
        
        if result.success:
            return output if output else "Command completed successfully (no output)."
        else:
            return f"Command failed (exit code {result.return_code}):\n{output}"
        
    except Exception as e:
        raise ToolError(f"Failed to run command: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
