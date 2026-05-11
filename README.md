# Slurm MCP Server

An MCP (Model Context Protocol) server that enables AI agents to interact with remote Slurm clusters via SSH. This server provides tools for job management, cluster monitoring, interactive sessions, and file operations.

## Features

- **Remote Slurm Management**: Submit, monitor, and cancel jobs via SSH
- **GPU Support**: Query GPU availability and allocate GPU resources
- **Pyxis/Enroot Containers**: Support for containerized workloads with `.sqsh` images
- **Interactive Sessions**: Launch and manage persistent interactive sessions
- **Session Profiles**: Save and reuse interactive session configurations
- **File Operations**: Browse, read, and write files on the cluster
- **Directory Structure**: Organized access to datasets, models, results, and logs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/slurm-mcp.git
cd slurm-mcp

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
cd /path/to/slurm_mcp
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your cluster details

# Generate mcp.json for Cursor/Claude Desktop
slurm-mcp-config .env -o mcp.json

# Or run directly
slurm-mcp
```

## Configuration

The server is configured via environment variables with the `SLURM_` prefix.

### Required Settings

| Variable | Description |
|----------|-------------|
| `SLURM_SSH_HOST` | Remote Slurm login node hostname |
| `SLURM_SSH_USER` | SSH username |
| `SLURM_USER_ROOT` | User's root directory on cluster |

### SSH Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_SSH_PORT` | 22 | SSH port |
| `SLURM_SSH_KEY_PATH` | None | Path to SSH private key |
| `SLURM_SSH_PASSWORD` | None | SSH password or key passphrase |
| `SLURM_SSH_KNOWN_HOSTS` | None | Path to known_hosts file |

### Slurm Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_DEFAULT_PARTITION` | None | Default partition for jobs |
| `SLURM_DEFAULT_ACCOUNT` | None | Default account/project |
| `SLURM_COMMAND_TIMEOUT` | 60 | Command timeout in seconds |
| `SLURM_GPU_PARTITIONS` | None | Comma-separated GPU partition names |
| `SLURM_CPU_PARTITIONS` | None | Comma-separated CPU partition names |

### Interactive Session Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_INTERACTIVE_PARTITION` | interactive | Partition for interactive jobs |
| `SLURM_INTERACTIVE_ACCOUNT` | None | Account for interactive jobs |
| `SLURM_INTERACTIVE_DEFAULT_TIME` | 4:00:00 | Default time limit |
| `SLURM_INTERACTIVE_DEFAULT_GPUS` | 8 | Default GPUs per node |
| `SLURM_INTERACTIVE_SESSION_TIMEOUT` | 3600 | Idle timeout in seconds |

### Directory Structure

| Variable | Default | Container Mount |
|----------|---------|-----------------|
| `SLURM_USER_ROOT` | Required | - |
| `SLURM_DIR_DATASETS` | `$USER_ROOT/data` | `/datasets` |
| `SLURM_DIR_RESULTS` | `$USER_ROOT/results` | `/results` |
| `SLURM_DIR_MODELS` | `$USER_ROOT/models` | `/models` |
| `SLURM_DIR_LOGS` | `$USER_ROOT/logs` | `/logs` |
| `SLURM_DIR_PROJECTS` | `$USER_ROOT/Projects` | `/projects` |
| `SLURM_DIR_CONTAINER_ROOT` | `$USER_ROOT/root` | `/root` |
| `SLURM_IMAGE_DIR` | `$USER_ROOT/images` | - |
| `SLURM_GPFS_ROOT` | None | `/lustre` |

### Example Configuration

```bash
# Minimal configuration
export SLURM_SSH_HOST="login.cluster.example.com"
export SLURM_SSH_USER="username"
export SLURM_SSH_KEY_PATH="~/.ssh/id_rsa"
export SLURM_USER_ROOT="/lustre/users/username"
export SLURM_GPFS_ROOT="/lustre"

# Optional: Account and partition defaults
export SLURM_DEFAULT_ACCOUNT="my_project"
export SLURM_INTERACTIVE_PARTITION="interactive"
export SLURM_DEFAULT_IMAGE="/lustre/users/username/images/pytorch.sqsh"
```

## Generating MCP Configuration

The `slurm-mcp-config` tool automatically converts your `.env` file to `mcp.json` format:

```bash
# Output to stdout
slurm-mcp-config .env

# Write to a file
slurm-mcp-config .env -o mcp.json

# Merge into existing Cursor config
slurm-mcp-config .env --merge ~/.cursor/mcp.json

# Use a custom server name
slurm-mcp-config .env --server-name my-cluster
```

Options:
- `-o, --output FILE` - Write to file instead of stdout
- `--merge FILE` - Merge into an existing mcp.json (preserves other servers)
- `--server-name NAME` - Custom server name (default: "slurm")
- `--command CMD` - Custom command (default: "slurm-mcp")

## Usage with MCP Clients

### Cursor IDE Setup

#### Step 1: Open MCP Settings

- Press `Ctrl+Shift+P` (Linux/Windows) or `Cmd+Shift+P` (Mac)
- Search for "Cursor Settings" 
- Navigate to **MCP** section
- Or directly edit: `~/.cursor/mcp.json`

#### Step 2: Add Server Configuration

**Option A: Using environment variables directly**

```json
{
  "mcpServers": {
    "slurm": {
      "command": "slurm-mcp",
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "SLURM_SSH_HOST": "login.cluster.example.com",
        "SLURM_SSH_USER": "username",
        "SLURM_SSH_KEY_PATH": "~/.ssh/id_rsa",
        "SLURM_USER_ROOT": "/lustre/users/username",
        "SLURM_GPFS_ROOT": "/lustre",
        "SLURM_DEFAULT_ACCOUNT": "my_project",
        "SLURM_DEFAULT_PARTITION": "batch",
        "SLURM_INTERACTIVE_PARTITION": "interactive"
      }
    }
  }
}
```

**Option B: Using an existing .env file**

If you have already configured a `.env` file in the project directory:

```json
{
  "mcpServers": {
    "slurm": {
      "command": "bash",
      "args": [
        "-c",
        "cd /path/to/slurm_mcp && set -a && source .env && set +a && python src/slurm_mcp/server.py"
      ]
    }
  }
}
```

**Option C: Using Python directly (development)**

```json
{
  "mcpServers": {
    "slurm": {
      "command": "python",
      "args": ["/path/to/slurm_mcp/src/slurm_mcp/server.py"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "SLURM_SSH_HOST": "login.cluster.example.com",
        "SLURM_SSH_USER": "username",
        "SLURM_SSH_KEY_PATH": "~/.ssh/id_rsa",
        "SLURM_USER_ROOT": "/lustre/users/username"
      }
    }
  }
}
```

#### Step 3: Restart Cursor

After saving the configuration, restart Cursor for changes to take effect.

#### Step 4: Verify Connection

1. Open Cursor Settings → MCP
2. You should see "slurm" listed as a connected server
3. The server provides 34 tools for cluster management

### Claude Desktop Setup

Add to your Claude Desktop config file:

- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "slurm": {
      "command": "slurm-mcp",
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "SLURM_SSH_HOST": "login.cluster.example.com",
        "SLURM_SSH_USER": "username",
        "SLURM_SSH_KEY_PATH": "~/.ssh/id_rsa",
        "SLURM_USER_ROOT": "/lustre/users/username"
      }
    }
  }
}
```

### Running Standalone

```bash
# Run directly (requires environment variables or .env file)
slurm-mcp

# Or via Python
python -m slurm_mcp.server

# Or with explicit config
SLURM_SSH_HOST=login.example.com SLURM_SSH_USER=user slurm-mcp
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Server not appearing | Check JSON syntax in config file, restart Cursor |
| Connection timeout | Verify SSH host/credentials work: `ssh user@host` |
| Permission denied | Check SSH key path and permissions (`chmod 600`) |
| Tools not loading | Check server logs for errors, verify `.env` variables |

## Available Tools

### Cluster Status (5 tools)

| Tool | Description |
|------|-------------|
| `get_cluster_status` | Get overall cluster status with partition info |
| `get_partition_info` | Detailed partition information |
| `get_node_info` | Information about cluster nodes |
| `get_gpu_info` | GPU resources by partition and type |
| `get_gpu_availability` | Check available GPUs for allocation |

### Job Management (7 tools)

| Tool | Description |
|------|-------------|
| `list_jobs` | List jobs in the queue |
| `get_job_details` | Detailed info about a specific job |
| `submit_job` | Submit a batch job |
| `cancel_job` | Cancel a job |
| `hold_job` | Put a job on hold |
| `release_job` | Release a held job |
| `get_job_history` | Job accounting/history (sacct) |

### Container Images (2 tools)

| Tool | Description |
|------|-------------|
| `list_container_images` | List available .sqsh images |
| `validate_container_image` | Check if an image exists and is valid |

### Interactive Sessions (9 tools)

| Tool | Description |
|------|-------------|
| `run_interactive_command` | Run a single command (one-shot allocation) |
| `start_interactive_session` | Start a persistent session |
| `exec_in_session` | Run command in existing session |
| `list_interactive_sessions` | List active sessions |
| `end_interactive_session` | End a session |
| `get_interactive_session_info` | Session details |
| `save_interactive_profile` | Save session configuration |
| `list_interactive_profiles` | List saved profiles |
| `start_session_from_profile` | Start session from profile |

### Directory Management (12 tools)

| Tool | Description |
|------|-------------|
| `get_cluster_directories` | Show configured directory structure |
| `list_directory` | List directory contents |
| `list_datasets` | List datasets directory |
| `list_model_checkpoints` | List model checkpoints |
| `list_job_logs` | List job log files |
| `read_file` | Read file contents |
| `write_file` | Write to a file |
| `find_files` | Search for files |
| `delete_file` | Delete file or directory |
| `get_disk_usage` | Check disk usage |
| `run_shell_command` | Run arbitrary shell command |

## Example Workflows

### Submit a GPU Training Job

```
User: Submit a PyTorch training job with 4 GPUs for 24 hours

Agent:
1. list_container_images(pattern="pytorch*")  # Find PyTorch image
2. get_gpu_availability(min_gpus=4)           # Check availability
3. submit_job(
     script_content="python train.py",
     partition="gpu",
     gpus=4,
     time_limit="24:00:00",
     container_image="/path/to/pytorch.sqsh"
   )
```

### Interactive Development Session

```
User: Start an interactive session for debugging

Agent:
1. start_interactive_session(
     session_name="debug",
     gpus_per_node=8,
     time_limit="2:00:00",
     container_image="/path/to/dev.sqsh"
   )
   # Returns session_id

2. exec_in_session(session_id="abc123", command="nvidia-smi")
3. exec_in_session(session_id="abc123", command="python train.py --debug")
4. end_interactive_session(session_id="abc123")
```

### Check Job Logs

```
User: My job 12345 failed, show me the error logs

Agent:
1. get_job_details(job_id=12345)  # Get job info and log paths
2. read_file(path="/logs/job-12345.err", tail_lines=100)
```

## Security Considerations

1. **SSH Key Management**: Use key-based authentication, not passwords
2. **Path Validation**: All paths are validated to prevent traversal attacks
3. **Deletion Confirmation**: `delete_file` requires explicit `confirm=True`
4. **Session Timeouts**: Idle sessions are automatically cleaned up
5. **Command Sanitization**: User inputs are sanitized before shell execution

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/slurm_mcp

# Linting
ruff check src/slurm_mcp
```

## License

MIT License - see LICENSE file for details.
