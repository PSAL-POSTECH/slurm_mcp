"""Interactive session manager for persistent Slurm allocations.

Sessions are NOT tracked in memory. Slurm is the source of truth:
- ``salloc --no-shell`` creates the allocation; it survives the MCP server
  going away (and the user's machine being powered off).
- Job name encodes ``mcp-session-{session_id}``.
- Job comment encodes per-session metadata (container image/mounts,
  user-facing session name, gpus-per-node) as ``mcp:<base64-json>``.

On every call we query ``squeue``/``scontrol`` and reconstruct
``InteractiveSession`` objects, so restarting the MCP server (or running
multiple servers) does not lose sessions.
"""

import base64
import binascii
import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from slurm_mcp.config import Settings
from slurm_mcp.models import CommandResult, InteractiveSession, JobInfo
from slurm_mcp.slurm_commands import SlurmCommands
from slurm_mcp.ssh_client import SSHClient, SSHCommandError

logger = logging.getLogger(__name__)


SESSION_JOB_PREFIX = "mcp-session-"
COMMENT_PREFIX = "mcp:"


def _encode_metadata(metadata: dict) -> str:
    """Encode session metadata for the Slurm job ``--comment`` field.

    Compact JSON + base64 keeps the value free of whitespace and quotes,
    so it round-trips cleanly through scontrol's ``Key=Value`` output.
    """
    raw = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)
    encoded = base64.b64encode(raw.encode("ascii")).decode("ascii")
    return f"{COMMENT_PREFIX}{encoded}"


def _decode_metadata(comment: Optional[str]) -> dict:
    if not comment or not comment.startswith(COMMENT_PREFIX):
        return {}
    payload = comment[len(COMMENT_PREFIX):]
    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
        result = json.loads(raw.decode("utf-8"))
        return result if isinstance(result, dict) else {}
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Failed to decode session metadata from comment: %r", comment)
        return {}


def _gpus_per_node_from_job(job: JobInfo) -> Optional[int]:
    """Best-effort extraction of GPUs-per-node from a Slurm job."""
    if job.tres_per_node:
        # Format: 'gres/gpu:4' or 'gres/gpu:a100:4'
        for piece in job.tres_per_node.split(","):
            piece = piece.strip()
            if piece.startswith("gres/gpu"):
                tail = piece.split(":")[-1]
                if tail.isdigit():
                    return int(tail)
    if job.num_gpus and job.num_nodes:
        per_node, remainder = divmod(job.num_gpus, job.num_nodes)
        if remainder == 0 and per_node > 0:
            return per_node
    return None


def _job_to_session(job: JobInfo, session_id: str) -> InteractiveSession:
    """Reconstruct an InteractiveSession from a Slurm JobInfo."""
    metadata = _decode_metadata(job.comment)
    gpus_per_node = metadata.get("gpus_per_node")
    if gpus_per_node is None:
        gpus_per_node = _gpus_per_node_from_job(job)

    status = "active" if job.state in ("RUNNING", "PENDING") else "ended"

    return InteractiveSession(
        session_id=session_id,
        job_id=job.job_id,
        session_name=metadata.get("session_name"),
        partition=job.partition,
        nodes=job.num_nodes,
        gpus_per_node=gpus_per_node,
        container_image=metadata.get("container_image"),
        container_mounts=metadata.get("container_mounts"),
        start_time=job.start_time or job.submit_time or datetime.now(),
        time_limit=job.time_limit or "",
        time_remaining=job.time_remaining,
        status=status,
        node_list=job.nodes,
    )


def _session_id_from_job_name(job_name: str) -> Optional[str]:
    if not job_name or not job_name.startswith(SESSION_JOB_PREFIX):
        return None
    session_id = job_name[len(SESSION_JOB_PREFIX):]
    return session_id or None


class InteractiveSessionManager:
    """Manages persistent interactive Slurm sessions.

    All session state lives in Slurm (job name + ``--comment``). Methods
    here are thin wrappers that query Slurm on demand.
    """

    def __init__(
        self,
        ssh_client: SSHClient,
        slurm: SlurmCommands,
        settings: Settings,
    ):
        self.ssh = ssh_client
        self.slurm = slurm
        self.settings = settings

    async def _find_job_for_session(self, session_id: str) -> Optional[JobInfo]:
        """Locate the Slurm job backing a given session_id, if still alive."""
        job_name = f"{SESSION_JOB_PREFIX}{session_id}"
        # squeue -n filters by exact name. Restrict to our user to keep it cheap.
        cmd = f"squeue -h -u {self.settings.ssh_user} -n {job_name} -o '%i'"
        result = await self.ssh.execute(cmd)
        if not result.success:
            logger.debug("squeue for session %s failed: %s", session_id, result.stderr)
            return None
        line = result.stdout.strip().splitlines()[:1]
        if not line:
            return None
        try:
            job_id = int(line[0].strip().split("_")[0])
        except ValueError:
            return None
        return await self.slurm.get_job_details(job_id)

    async def start_session(
        self,
        session_name: Optional[str] = None,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        nodes: int = 1,
        gpus_per_node: Optional[int] = None,
        time_limit: Optional[str] = None,
        container_image: Optional[str] = None,
        container_mounts: Optional[str] = None,
        no_container_mount_home: bool = True,
    ) -> InteractiveSession:
        """Start a new interactive session.

        Resources are allocated via ``salloc --no-shell``; the allocation
        survives MCP server restarts. Session metadata is stored in the
        Slurm job comment so it can be recovered later.
        """
        session_id = str(uuid.uuid4())[:8]
        job_name = f"{SESSION_JOB_PREFIX}{session_id}"

        partition = partition or self.settings.interactive_partition
        account = account or self.settings.interactive_account
        time_limit = time_limit or self.settings.interactive_default_time
        if gpus_per_node is None:
            gpus_per_node = self.settings.interactive_default_gpus
        container_mounts = container_mounts or self.settings.get_container_mounts()

        metadata = {
            "session_name": session_name,
            "container_image": container_image,
            "container_mounts": container_mounts,
            "gpus_per_node": gpus_per_node,
        }
        comment = _encode_metadata({k: v for k, v in metadata.items() if v is not None})

        logger.info("Starting interactive session %s on partition %s", session_id, partition)

        job_id = await self.slurm.salloc(
            partition=partition,
            account=account,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            job_name=job_name,
            comment=comment,
        )

        logger.info("Session %s allocated job %s", session_id, job_id)

        job_info = await self.slurm.get_job_details(job_id)
        if job_info is None:
            # Allocation succeeded but we cannot inspect it; build a minimal record.
            return InteractiveSession(
                session_id=session_id,
                job_id=job_id,
                session_name=session_name,
                partition=partition,
                nodes=nodes,
                gpus_per_node=gpus_per_node,
                container_image=container_image,
                container_mounts=container_mounts,
                start_time=datetime.now(),
                time_limit=time_limit,
                status="active",
            )

        return _job_to_session(job_info, session_id)

    async def exec_command(
        self,
        session_id: str,
        command: str,
        working_directory: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Execute a command in an existing session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status != "active":
            raise ValueError(f"Session {session_id} is not active (status: {session.status})")

        logger.debug("Executing command in session %s: %s...", session_id, command[:50])

        return await self.slurm.srun_in_allocation(
            job_id=session.job_id,
            command=command,
            container_image=session.container_image,
            container_mounts=session.container_mounts,
            working_directory=working_directory,
            timeout=timeout,
        )

    async def end_session(self, session_id: str) -> bool:
        """End an interactive session by cancelling its Slurm allocation."""
        job = await self._find_job_for_session(session_id)
        if job is None:
            return False

        logger.info("Ending session %s (job %s)", session_id, job.job_id)
        return await self.slurm.scancel(job.job_id)

    async def get_session(self, session_id: str) -> Optional[InteractiveSession]:
        """Get session info from Slurm, or None if the allocation is gone."""
        job = await self._find_job_for_session(session_id)
        if job is None or job.state not in ("RUNNING", "PENDING"):
            return None
        return _job_to_session(job, session_id)

    async def list_sessions(self) -> list[InteractiveSession]:
        """List all active interactive sessions for the current user."""
        # Fetch only this user's jobs; filter by our naming convention.
        cmd = f"squeue -h -u {self.settings.ssh_user} -o '%i|%j'"
        result = await self.ssh.execute(cmd)
        if not result.success:
            logger.error("squeue failed while listing sessions: %s", result.stderr)
            return []

        session_jobs: list[tuple[int, str]] = []
        for line in result.stdout.strip().splitlines():
            if "|" not in line:
                continue
            jid_raw, name = line.split("|", 1)
            session_id = _session_id_from_job_name(name.strip())
            if not session_id:
                continue
            try:
                job_id = int(jid_raw.strip().split("_")[0])
            except ValueError:
                continue
            session_jobs.append((job_id, session_id))

        sessions: list[InteractiveSession] = []
        for job_id, session_id in session_jobs:
            details = await self.slurm.get_job_details(job_id)
            if details and details.state in ("RUNNING", "PENDING"):
                sessions.append(_job_to_session(details, session_id))
        return sessions

    async def run_command(
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
        """Execute a single command with interactive resources (one-shot).

        This allocates resources, runs the command, and releases resources.
        No persistent session is created.
        """
        return await self.slurm.srun_command(
            command=command,
            partition=partition,
            account=account,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            container_mounts=container_mounts,
            no_container_mount_home=no_container_mount_home,
            working_directory=working_directory,
            timeout=timeout,
        )


_session_manager: Optional[InteractiveSessionManager] = None


def get_session_manager(
    ssh_client: Optional[SSHClient] = None,
    slurm: Optional[SlurmCommands] = None,
    settings: Optional[Settings] = None,
) -> InteractiveSessionManager:
    """Get or create the global session manager instance."""
    global _session_manager

    if _session_manager is None:
        if ssh_client is None or slurm is None or settings is None:
            raise ValueError("ssh_client, slurm, and settings required on first call")
        _session_manager = InteractiveSessionManager(ssh_client, slurm, settings)

    return _session_manager


async def reset_session_manager() -> None:
    """Drop the cached session manager instance.

    Slurm allocations are not touched: they persist independently and can be
    recovered by any subsequent session manager.
    """
    global _session_manager
    _session_manager = None
