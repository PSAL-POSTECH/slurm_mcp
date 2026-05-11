"""Pydantic models for Slurm MCP server data structures."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Slurm job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    COMPLETING = "COMPLETING"
    CONFIGURING = "CONFIGURING"
    RESIZING = "RESIZING"
    REVOKED = "REVOKED"
    SPECIAL_EXIT = "SPECIAL_EXIT"


class NodeState(str, Enum):
    """Slurm node states."""
    IDLE = "idle"
    ALLOCATED = "allocated"
    MIXED = "mixed"
    DOWN = "down"
    DRAINED = "drained"
    DRAINING = "draining"
    RESERVED = "reserved"
    UNKNOWN = "unknown"


class CommandResult(BaseModel):
    """Result of executing a command via SSH."""
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    return_code: int = Field(description="Command return code")
    
    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.return_code == 0
    
    @property
    def output(self) -> str:
        """Get combined output, preferring stdout."""
        return self.stdout if self.stdout else self.stderr


class GPUInfo(BaseModel):
    """GPU resource information."""
    gpu_type: str = Field(description="GPU type (e.g., 'a100', 'v100', 'h100')")
    count: int = Field(description="Number of GPUs")
    memory_gb: Optional[int] = Field(default=None, description="GPU memory in GB")
    available: Optional[int] = Field(default=None, description="Number of available GPUs")
    allocated: Optional[int] = Field(default=None, description="Number of allocated GPUs")


class NodeInfo(BaseModel):
    """Information about a cluster node."""
    node_name: str = Field(description="Node name")
    state: str = Field(description="Node state")
    cpus_total: int = Field(description="Total CPUs on node")
    cpus_allocated: int = Field(default=0, description="Allocated CPUs")
    cpus_available: int = Field(default=0, description="Available CPUs")
    memory_total_mb: int = Field(description="Total memory in MB")
    memory_allocated_mb: int = Field(default=0, description="Allocated memory in MB")
    memory_available_mb: int = Field(default=0, description="Available memory in MB")
    partitions: list[str] = Field(default_factory=list, description="Partitions this node belongs to")
    gpus: Optional[list[GPUInfo]] = Field(default=None, description="GPU information")
    features: list[str] = Field(default_factory=list, description="Node features")


class PartitionInfo(BaseModel):
    """Information about a cluster partition."""
    name: str = Field(description="Partition name")
    state: str = Field(description="Partition state (up/down)")
    total_nodes: int = Field(description="Total nodes in partition")
    available_nodes: int = Field(default=0, description="Available nodes")
    total_cpus: int = Field(description="Total CPUs in partition")
    available_cpus: int = Field(default=0, description="Available CPUs")
    max_time: Optional[str] = Field(default=None, description="Maximum time limit")
    default: bool = Field(default=False, description="Whether this is the default partition")
    has_gpus: bool = Field(default=False, description="Whether partition has GPU nodes")
    gpu_types: list[str] = Field(default_factory=list, description="Available GPU types")
    total_gpus: int = Field(default=0, description="Total GPUs in partition")
    available_gpus: int = Field(default=0, description="Available GPUs")


class JobInfo(BaseModel):
    """Information about a Slurm job."""
    job_id: int = Field(description="Slurm job ID")
    job_name: str = Field(description="Job name")
    user: str = Field(description="Username who submitted the job")
    state: str = Field(description="Job state")
    partition: str = Field(description="Partition name")
    nodes: Optional[str] = Field(default=None, description="Allocated nodes")
    num_nodes: int = Field(default=1, description="Number of nodes")
    num_cpus: int = Field(default=1, description="Number of CPUs")
    num_gpus: int = Field(default=0, description="Number of GPUs")
    memory: Optional[str] = Field(default=None, description="Memory allocation")
    time_limit: Optional[str] = Field(default=None, description="Time limit")
    time_used: Optional[str] = Field(default=None, description="Time used")
    time_remaining: Optional[str] = Field(default=None, description="Time remaining")
    submit_time: Optional[datetime] = Field(default=None, description="Submission time")
    start_time: Optional[datetime] = Field(default=None, description="Start time")
    end_time: Optional[datetime] = Field(default=None, description="End time")
    work_dir: Optional[str] = Field(default=None, description="Working directory")
    stdout_path: Optional[str] = Field(default=None, description="Stdout file path")
    stderr_path: Optional[str] = Field(default=None, description="Stderr file path")
    container_image: Optional[str] = Field(default=None, description="Container image used")
    exit_code: Optional[int] = Field(default=None, description="Exit code (if completed)")
    reason: Optional[str] = Field(default=None, description="Reason for pending/failed state")
    comment: Optional[str] = Field(default=None, description="Slurm job comment (used to carry session metadata)")
    tres_per_node: Optional[str] = Field(default=None, description="TRES per node (e.g., 'gres/gpu:4')")


class JobSubmission(BaseModel):
    """Parameters for submitting a Slurm job."""
    script_content: str = Field(description="The batch script content (commands to run)")
    job_name: Optional[str] = Field(default=None, description="Job name")
    partition: Optional[str] = Field(default=None, description="Partition to submit to")
    account: Optional[str] = Field(default=None, description="Account/project for billing")
    nodes: Optional[int] = Field(default=None, description="Number of nodes")
    ntasks: Optional[int] = Field(default=None, description="Number of tasks")
    cpus_per_task: Optional[int] = Field(default=None, description="CPUs per task")
    memory: Optional[str] = Field(default=None, description="Memory per node (e.g., '4G')")
    time_limit: Optional[str] = Field(default=None, description="Time limit (e.g., '1:00:00')")
    output_file: Optional[str] = Field(default=None, description="Output file path")
    error_file: Optional[str] = Field(default=None, description="Error file path")
    working_directory: Optional[str] = Field(default=None, description="Working directory")
    
    # GPU options
    gpus: Optional[int] = Field(default=None, description="Number of GPUs per node")
    gpus_per_task: Optional[int] = Field(default=None, description="GPUs per task")
    gpu_type: Optional[str] = Field(default=None, description="Specific GPU type")
    
    # Container options (Pyxis)
    container_image: Optional[str] = Field(default=None, description="Container .sqsh image path")
    container_mounts: Optional[str] = Field(default=None, description="Container bind mounts")
    container_workdir: Optional[str] = Field(default=None, description="Working directory inside container")
    container_env: Optional[str] = Field(default=None, description="Environment variables for container")
    no_container_mount_home: bool = Field(default=True, description="Don't mount home in container")
    
    # Array job
    array: Optional[str] = Field(default=None, description="Array job specification (e.g., '0-9', '1,3,5')")
    
    # Dependencies
    dependency: Optional[str] = Field(default=None, description="Job dependencies (e.g., 'afterok:12345')")
    
    def generate_sbatch_script(self, default_partition: Optional[str] = None,
                                default_account: Optional[str] = None,
                                default_mounts: Optional[str] = None) -> str:
        """Generate a complete SBATCH script with directives."""
        lines = ["#!/bin/bash"]
        
        # Job name
        if self.job_name:
            lines.append(f"#SBATCH --job-name={self.job_name}")
        
        # Partition
        partition = self.partition or default_partition
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        
        # Account
        account = self.account or default_account
        if account:
            lines.append(f"#SBATCH --account={account}")
        
        # Resources
        if self.nodes:
            lines.append(f"#SBATCH --nodes={self.nodes}")
        if self.ntasks:
            lines.append(f"#SBATCH --ntasks={self.ntasks}")
        if self.cpus_per_task:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.memory:
            lines.append(f"#SBATCH --mem={self.memory}")
        if self.time_limit:
            lines.append(f"#SBATCH --time={self.time_limit}")
        
        # GPU resources
        if self.gpus:
            if self.gpu_type:
                lines.append(f"#SBATCH --gpus-per-node={self.gpu_type}:{self.gpus}")
            else:
                lines.append(f"#SBATCH --gpus-per-node={self.gpus}")
        if self.gpus_per_task:
            lines.append(f"#SBATCH --gpus-per-task={self.gpus_per_task}")
        
        # Output files
        if self.output_file:
            lines.append(f"#SBATCH --output={self.output_file}")
        if self.error_file:
            lines.append(f"#SBATCH --error={self.error_file}")
        
        # Working directory
        if self.working_directory:
            lines.append(f"#SBATCH --chdir={self.working_directory}")
        
        # Array job
        if self.array:
            lines.append(f"#SBATCH --array={self.array}")
        
        # Dependencies
        if self.dependency:
            lines.append(f"#SBATCH --dependency={self.dependency}")
        
        # Container options (Pyxis)
        if self.container_image:
            lines.append(f"#SBATCH --container-image={self.container_image}")
            
            # Container mounts
            mounts = self.container_mounts or default_mounts
            if mounts:
                lines.append(f"#SBATCH --container-mounts={mounts}")
            
            if self.no_container_mount_home:
                lines.append("#SBATCH --no-container-mount-home")
            
            if self.container_workdir:
                lines.append(f"#SBATCH --container-workdir={self.container_workdir}")
        
        # Add blank line before script content
        lines.append("")
        
        # Add environment variables if specified
        if self.container_env:
            for env_var in self.container_env.split(","):
                env_var = env_var.strip()
                if "=" in env_var:
                    lines.append(f"export {env_var}")
            lines.append("")
        
        # Add the actual script content
        lines.append(self.script_content)
        
        return "\n".join(lines)


class ContainerImage(BaseModel):
    """Container image information."""
    name: str = Field(description="Image name (filename without path)")
    path: str = Field(description="Full path to the .sqsh file")
    size_bytes: int = Field(description="File size in bytes")
    size_human: str = Field(description="Human-readable size")
    modified_time: datetime = Field(description="Last modification time")
    description: Optional[str] = Field(default=None, description="Image description if available")


class InteractiveSession(BaseModel):
    """Information about an active interactive session."""
    session_id: str = Field(description="Unique session identifier")
    job_id: int = Field(description="Slurm job ID for the allocation")
    session_name: Optional[str] = Field(default=None, description="User-provided session name")
    partition: str = Field(description="Partition name")
    nodes: int = Field(description="Number of allocated nodes")
    gpus_per_node: Optional[int] = Field(default=None, description="GPUs per node")
    container_image: Optional[str] = Field(default=None, description="Container image path")
    container_mounts: Optional[str] = Field(default=None, description="Container mounts")
    start_time: datetime = Field(description="Session start time")
    time_limit: str = Field(description="Time limit")
    time_remaining: Optional[str] = Field(default=None, description="Time remaining")
    status: str = Field(description="Session status (active, ending, ended)")
    node_list: Optional[str] = Field(default=None, description="Allocated node names")
    last_command_time: Optional[datetime] = Field(default=None, description="Last command execution time")


class InteractiveProfile(BaseModel):
    """Saved configuration for interactive sessions."""
    name: str = Field(description="Profile name")
    description: Optional[str] = Field(default=None, description="Profile description")
    partition: Optional[str] = Field(default=None, description="Partition")
    account: Optional[str] = Field(default=None, description="Account")
    nodes: int = Field(default=1, description="Number of nodes")
    gpus_per_node: Optional[int] = Field(default=None, description="GPUs per node")
    cpus_per_task: Optional[int] = Field(default=None, description="CPUs per task")
    memory: Optional[str] = Field(default=None, description="Memory allocation")
    time_limit: Optional[str] = Field(default=None, description="Time limit")
    container_image: Optional[str] = Field(default=None, description="Container image")
    container_mounts: Optional[str] = Field(default=None, description="Container mounts")
    no_container_mount_home: bool = Field(default=True, description="Don't mount home")
    env_vars: Optional[dict[str, str]] = Field(default=None, description="Environment variables")
    created_at: Optional[datetime] = Field(default=None, description="Creation time")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")


class ClusterDirectories(BaseModel):
    """Configured cluster directory structure."""
    user_root: str = Field(description="User's root directory")
    datasets: str = Field(description="Datasets directory")
    results: str = Field(description="Results directory")
    models: str = Field(description="Models directory")
    logs: str = Field(description="Logs directory")
    projects: Optional[str] = Field(default=None, description="Projects directory")
    scratch: Optional[str] = Field(default=None, description="Scratch directory")
    home: Optional[str] = Field(default=None, description="Home directory")
    container_root: Optional[str] = Field(default=None, description="Container root overlay")
    gpfs_root: Optional[str] = Field(default=None, description="GPFS/Lustre root")
    images: Optional[str] = Field(default=None, description="Container images directory")
    
    def get_mount_mapping(self) -> dict[str, str]:
        """Get mapping of host paths to container mount points."""
        mapping = {
            self.datasets: "/datasets",
            self.results: "/results",
            self.models: "/models",
            self.logs: "/logs",
        }
        if self.projects:
            mapping[self.projects] = "/projects"
        if self.scratch:
            mapping[self.scratch] = "/scratch"
        if self.home:
            mapping[self.home] = "/home"
        if self.container_root:
            mapping[self.container_root] = "/root"
        if self.gpfs_root:
            mapping[self.gpfs_root] = "/lustre"
        return mapping


class FileInfo(BaseModel):
    """Information about a file or directory."""
    name: str = Field(description="File/directory name")
    path: str = Field(description="Full path")
    size_bytes: int = Field(description="Size in bytes")
    size_human: str = Field(description="Human-readable size")
    modified_time: datetime = Field(description="Last modification time")
    is_dir: bool = Field(description="Whether this is a directory")
    is_link: bool = Field(default=False, description="Whether this is a symlink")
    permissions: str = Field(description="Permission string (e.g., 'rwxr-xr-x')")
    owner: Optional[str] = Field(default=None, description="Owner username")
    group: Optional[str] = Field(default=None, description="Group name")


class DirectoryListing(BaseModel):
    """Contents of a directory."""
    path: str = Field(description="Directory path")
    files: list[FileInfo] = Field(default_factory=list, description="Files in directory")
    subdirs: list[FileInfo] = Field(default_factory=list, description="Subdirectories")
    total_items: int = Field(description="Total number of items")
    total_size_bytes: int = Field(description="Total size of all items")
    total_size_human: str = Field(description="Human-readable total size")
