#!/usr/bin/env python3
"""
Convert a .env file to mcp.json format for MCP clients (Cursor, Claude Desktop, etc.)

Usage:
    python scripts/env_to_mcp_json.py                    # Uses .env in current directory
    python scripts/env_to_mcp_json.py .env               # Specify .env file path
    python scripts/env_to_mcp_json.py .env -o mcp.json   # Specify output file
    python scripts/env_to_mcp_json.py .env --merge existing_mcp.json  # Merge into existing mcp.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file and return a dictionary of environment variables."""
    env_vars = {}
    
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    
    with open(env_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse KEY=VALUE format
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
            if match:
                key = match.group(1)
                value = match.group(2)
                
                # Remove surrounding quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # Only include SLURM_ prefixed variables
                if key.startswith("SLURM_"):
                    env_vars[key] = value
            else:
                # Skip malformed lines silently (could be multi-line values or other formats)
                pass
    
    return env_vars


def create_mcp_json(
    env_vars: dict[str, str],
    server_name: str = "slurm",
    command: str = "slurm-mcp"
) -> dict:
    """Create an mcp.json structure from environment variables."""
    env_with_utf8 = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        **env_vars,
    }
    return {
        "mcpServers": {
            server_name: {
                "command": command,
                "env": env_with_utf8
            }
        }
    }


def merge_mcp_json(existing: dict, new_server: dict) -> dict:
    """Merge a new server configuration into an existing mcp.json structure."""
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}
    
    existing["mcpServers"].update(new_server["mcpServers"])
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Convert .env file to mcp.json format for MCP clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                              # Convert .env to stdout
    %(prog)s .env -o mcp.json             # Convert .env to mcp.json file
    %(prog)s .env --merge ~/.cursor/mcp.json  # Merge into existing mcp.json
    %(prog)s .env --server-name my-cluster    # Use custom server name
        """
    )
    
    parser.add_argument(
        "env_file",
        nargs="?",
        default=".env",
        help="Path to .env file (default: .env)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--merge",
        metavar="FILE",
        help="Merge into an existing mcp.json file instead of creating new"
    )
    
    parser.add_argument(
        "--server-name",
        default="slurm",
        help="Name for the MCP server entry (default: slurm)"
    )
    
    parser.add_argument(
        "--command",
        default="slurm-mcp",
        help="Command to run the MCP server (default: slurm-mcp)"
    )
    
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Parse the .env file
    env_path = Path(args.env_file)
    try:
        env_vars = parse_env_file(env_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not env_vars:
        print("Warning: No SLURM_ environment variables found in the file", file=sys.stderr)
    
    # Create the MCP JSON structure
    mcp_json = create_mcp_json(
        env_vars,
        server_name=args.server_name,
        command=args.command
    )
    
    # Merge if requested
    if args.merge:
        merge_path = Path(args.merge)
        if merge_path.exists():
            try:
                with open(merge_path, "r") as f:
                    existing = json.load(f)
                mcp_json = merge_mcp_json(existing, mcp_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing existing mcp.json: {e}", file=sys.stderr)
                sys.exit(1)
        # If merge file doesn't exist, we'll create it with just the new config
    
    # Output the result
    json_output = json.dumps(mcp_json, indent=args.indent)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(json_output)
            f.write("\n")
        print(f"Written to {output_path}", file=sys.stderr)
    elif args.merge:
        merge_path = Path(args.merge)
        merge_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merge_path, "w") as f:
            f.write(json_output)
            f.write("\n")
        print(f"Merged into {merge_path}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
