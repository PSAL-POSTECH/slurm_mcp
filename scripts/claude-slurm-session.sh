#!/usr/bin/env bash
#
# Launch a Claude Code 'remote-control' session inside a running Slurm job.
#
# A tmux session on the login node owns `srun --jobid=<JID> ... claude
# remote-control`, so the Claude process survives SSH drops and you can
# tmux-attach later to drive or debug it directly.
#
# Usage:
#   claude-slurm-session.sh <job_id> [options]
#
# Options:
#   -n, --name NAME              Session name (default: slurm-<job_id>).
#                                Used for both tmux session 'claude-<name>'
#                                and the title shown at claude.ai/code.
#   -i, --container-image PATH   Pyxis .sqsh image to run Claude inside.
#                                If omitted, tries to read from the job's
#                                metadata (set by slurm_mcp); falls back to
#                                running bare on the compute node.
#   -m, --container-mounts SPEC  Pyxis container mounts (same inheritance
#                                rule as --container-image).
#       --no-container           Force bare execution, ignoring auto-detected
#                                container settings.
#   -b, --claude-binary PATH     Path to claude CLI in the exec environment
#                                (default: claude).
#   -t, --timeout SECS           Seconds to wait for the session URL to
#                                appear in tmux (default: 30).
#   -h, --help                   Show this help.
#
# Prerequisites:
#   - tmux installed on the login node.
#   - 'claude' CLI available in the execution environment (compute node, or
#     inside the container if --container-image is set).
#   - ~/.claude credentials visible to the execution environment. HPC home
#     mounts usually carry these into containers automatically.
#   - Outbound HTTPS to the Anthropic API allowed from the compute node.

set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed -n 's/^# \{0,1\}//p'
    exit "${1:-0}"
}

JOB_ID=""
NAME=""
CONTAINER_IMAGE=""
CONTAINER_MOUNTS=""
NO_CONTAINER=0
EXPLICIT_CONTAINER=0
CLAUDE_BIN="claude"
TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)              NAME="$2"; shift 2 ;;
        -i|--container-image)   CONTAINER_IMAGE="$2"; EXPLICIT_CONTAINER=1; shift 2 ;;
        -m|--container-mounts)  CONTAINER_MOUNTS="$2"; EXPLICIT_CONTAINER=1; shift 2 ;;
            --no-container)     NO_CONTAINER=1; shift ;;
        -b|--claude-binary)     CLAUDE_BIN="$2"; shift 2 ;;
        -t|--timeout)           TIMEOUT="$2"; shift 2 ;;
        -h|--help)              usage 0 ;;
        --)                     shift; break ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage 1
            ;;
        *)
            if [[ -z "$JOB_ID" ]]; then
                JOB_ID="$1"
                shift
            else
                echo "error: unexpected positional argument: $1" >&2
                usage 1
            fi
            ;;
    esac
done

if [[ -z "$JOB_ID" ]]; then
    echo "error: job_id is required" >&2
    usage 1
fi

if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "error: job_id must be a positive integer (got: $JOB_ID)" >&2
    exit 1
fi

for dep in tmux squeue; do
    if ! command -v "$dep" >/dev/null 2>&1; then
        echo "error: '$dep' not found on $(hostname). Are you on a Slurm login node?" >&2
        exit 1
    fi
done

job_state=$(squeue -h -j "$JOB_ID" -o '%T' 2>/dev/null | head -n1 || true)
if [[ -z "$job_state" ]]; then
    echo "error: Slurm job $JOB_ID not found (or not visible to you)" >&2
    exit 1
fi
if [[ "$job_state" != "RUNNING" ]]; then
    echo "error: Slurm job $JOB_ID is in state '$job_state', need RUNNING" >&2
    exit 1
fi

# Auto-detect container settings from slurm_mcp metadata (encoded in
# job comment as 'mcp:<base64 json>'), unless overridden.
if [[ $NO_CONTAINER -eq 0 && $EXPLICIT_CONTAINER -eq 0 ]]; then
    if command -v python3 >/dev/null 2>&1; then
        meta_json=$(
            scontrol show job "$JOB_ID" -o 2>/dev/null \
              | tr ' ' '\n' \
              | grep '^Comment=' \
              | head -n1 \
              | sed 's/^Comment=//' \
              | python3 -c '
import base64, json, sys
v = sys.stdin.read().strip()
if not v.startswith("mcp:"):
    sys.exit(0)
try:
    raw = base64.b64decode(v[4:], validate=True)
    d = json.loads(raw.decode("utf-8"))
    if isinstance(d, dict):
        ci = d.get("container_image") or ""
        cm = d.get("container_mounts") or ""
        print(ci); print(cm)
except Exception:
    pass
' 2>/dev/null || true)
        if [[ -n "$meta_json" ]]; then
            CONTAINER_IMAGE=$(echo "$meta_json" | sed -n '1p')
            CONTAINER_MOUNTS=$(echo "$meta_json" | sed -n '2p')
            if [[ -n "$CONTAINER_IMAGE" ]]; then
                echo "info: auto-detected container from job metadata: $CONTAINER_IMAGE" >&2
            fi
        fi
    fi
fi

if [[ $NO_CONTAINER -eq 1 ]]; then
    CONTAINER_IMAGE=""
    CONTAINER_MOUNTS=""
fi

NAME="${NAME:-slurm-$JOB_ID}"
TMUX_SESSION="claude-$NAME"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "error: tmux session '$TMUX_SESSION' already exists on $(hostname)" >&2
    echo "       stop it first: tmux kill-session -t $TMUX_SESSION" >&2
    exit 1
fi

LOGIN_HOST=$(hostname -f 2>/dev/null || hostname)

SRUN_ARGS=("srun" "--jobid=$JOB_ID" "--overlap" "--pty")
if [[ -n "$CONTAINER_IMAGE" ]]; then
    SRUN_ARGS+=("--container-image=$CONTAINER_IMAGE")
    if [[ -n "$CONTAINER_MOUNTS" ]]; then
        SRUN_ARGS+=("--container-mounts=$CONTAINER_MOUNTS")
    fi
fi
SRUN_ARGS+=("--" "$CLAUDE_BIN" "remote-control" "--spawn" "session" "--name" "$NAME")

SRUN_CMD=$(printf '%q ' "${SRUN_ARGS[@]}")

tmux new-session -d -s "$TMUX_SESSION" "$SRUN_CMD"
echo "info: tmux session '$TMUX_SESSION' started on $LOGIN_HOST"
echo "info: waiting up to ${TIMEOUT}s for Claude session URL..."

URL=""
for ((i=0; i<TIMEOUT; i++)); do
    sleep 1
    OUTPUT=$(tmux capture-pane -t "$TMUX_SESSION" -p -S -2000 2>/dev/null || true)
    URL=$(echo "$OUTPUT" \
            | grep -oE 'https://claude\.ai/code/[^[:space:]]+' \
            | head -n1 \
            | sed 's/[.,;)\"'"'"']*$//')
    if [[ -n "$URL" ]]; then
        break
    fi
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "" >&2
        echo "error: tmux session '$TMUX_SESSION' died before URL appeared" >&2
        echo "Last output:" >&2
        echo "$OUTPUT" | tail -n 30 >&2
        exit 1
    fi
done

if [[ -z "$URL" ]]; then
    echo "" >&2
    echo "error: Claude session URL did not appear within ${TIMEOUT}s" >&2
    echo "       tmux session is still running. Attach to diagnose:" >&2
    echo "         ssh $LOGIN_HOST -t tmux attach -t $TMUX_SESSION" >&2
    echo "       Or stop it: ssh $LOGIN_HOST tmux kill-session -t $TMUX_SESSION" >&2
    echo "" >&2
    echo "Recent output:" >&2
    tmux capture-pane -t "$TMUX_SESSION" -p 2>&1 | tail -n 30 >&2
    exit 1
fi

CONTAINER_DESC="${CONTAINER_IMAGE:-(none, bare compute node)}"

cat <<EOF

Claude remote-control session started.
  Name:         $NAME
  URL:          $URL
  Slurm Job:    $JOB_ID
  Login Host:   $LOGIN_HOST
  tmux session: $TMUX_SESSION
  Container:    $CONTAINER_DESC

Open the URL in claude.ai/code or the Claude mobile app to drive the session.

Attach for debugging: ssh $LOGIN_HOST -t tmux attach -t $TMUX_SESSION
Stop:                 ssh $LOGIN_HOST tmux kill-session -t $TMUX_SESSION
EOF
