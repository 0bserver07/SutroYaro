#!/usr/bin/env python3
"""
Export Claude Code session traces for the SutroYaro project.

Usage:
    python3 .traces/export_sessions.py              # export all sessions
    python3 .traces/export_sessions.py --list       # list sessions with metadata
    python3 .traces/export_sessions.py SESSION_ID   # export one session
    python3 .traces/export_sessions.py --team NAME  # export all sessions from a team

Reads from: ~/.claude/projects/-Users-yadkonrad-dev-dev-year26-feb26-SutroYaro/
Writes to:  .traces/sessions/
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TRACES_DIR = REPO_ROOT / ".traces" / "sessions"
SOURCE_DIR = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-yadkonrad-dev-dev-year26-feb26-SutroYaro"
)


def read_session_meta(jsonl_path):
    """Read first line to get session metadata."""
    with open(jsonl_path) as f:
        first_line = f.readline().strip()
        if not first_line:
            return None
        try:
            d = json.loads(first_line)
            return {
                "session_id": d.get("sessionId", jsonl_path.stem),
                "team": d.get("teamName", ""),
                "agent": d.get("agentName", ""),
                "type": d.get("type", ""),
                "branch": d.get("gitBranch", ""),
                "version": d.get("version", ""),
                "lines": sum(1 for _ in open(jsonl_path)),
            }
        except json.JSONDecodeError:
            return None


def extract_text(content):
    """Extract displayable text from a message content block."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    name = block.get("name", "unknown")
                    inp = block.get("input", {})
                    if name == "Read":
                        parts.append(f"  [Read: {inp.get('file_path', '?')}]")
                    elif name == "Write":
                        parts.append(f"  [Write: {inp.get('file_path', '?')}]")
                    elif name == "Edit":
                        parts.append(f"  [Edit: {inp.get('file_path', '?')}]")
                    elif name == "Bash":
                        cmd = inp.get("command", "?")
                        if len(cmd) > 120:
                            cmd = cmd[:120] + "..."
                        parts.append(f"  [Bash: {cmd}]")
                    elif name == "Grep":
                        parts.append(f"  [Grep: {inp.get('pattern', '?')} in {inp.get('path', '.')}]")
                    elif name == "Glob":
                        parts.append(f"  [Glob: {inp.get('pattern', '?')}]")
                    elif name == "Agent":
                        parts.append(f"  [Agent: {inp.get('description', '?')}]")
                    elif name == "TaskCreate":
                        parts.append(f"  [TaskCreate: {inp.get('subject', '?')}]")
                    elif name == "TaskUpdate":
                        parts.append(f"  [TaskUpdate: {inp.get('taskId', '?')} -> {inp.get('status', '?')}]")
                    elif name == "SendMessage":
                        parts.append(f"  [SendMessage: {inp.get('type', '?')} -> {inp.get('recipient', '?')}]")
                    else:
                        parts.append(f"  [Tool: {name}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def export_session(jsonl_path, output_dir):
    """Convert a session JSONL to readable text. Returns metadata dict or None."""
    meta = read_session_meta(jsonl_path)
    if not meta or meta["type"] == "file-history-snapshot":
        return None

    messages = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")
            timestamp = entry.get("timestamp", "")

            # Extract from nested message format
            msg = entry.get("message", {})
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else ""

            # Skip non-message types
            if entry_type in ("progress", "result", "file-history-snapshot"):
                continue
            if not role or role == "system":
                continue

            text = extract_text(content)
            if not text or not text.strip():
                continue

            # Clean system reminders
            text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
            text = text.strip()
            if not text:
                continue

            messages.append((role, text, timestamp))

    if not messages:
        return None

    # Build filename
    parts = []
    if meta["team"]:
        parts.append(meta["team"])
    if meta["agent"]:
        parts.append(meta["agent"])
    parts.append(meta["session_id"][:8])
    filename = "-".join(parts) + ".txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    lines = []
    lines.append(f"# Session: {meta['session_id']}")
    if meta["team"]:
        lines.append(f"# Team: {meta['team']}")
    if meta["agent"]:
        lines.append(f"# Agent: {meta['agent']}")
    lines.append(f"# Branch: {meta['branch']}")
    lines.append(f"# Messages: {len(messages)}")

    # Compute session duration from first/last timestamps
    timestamps = [t for _, _, t in messages if t]
    first_ts = last_ts = None
    if timestamps:
        try:
            first_ts = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
            last_ts = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
        except (ValueError, IndexError):
            pass
    if first_ts:
        lines.append(f"# Started: {first_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if first_ts and last_ts:
        duration = last_ts - first_ts
        mins = int(duration.total_seconds() // 60)
        secs = int(duration.total_seconds() % 60)
        lines.append(f"# Duration: {mins}m {secs}s")

    lines.append(f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    for role, text, ts in messages:
        label = "YOU" if role in ("user", "human") else "CLAUDE" if role == "assistant" else role.upper()
        ts_str = ""
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_str = f" [{t.strftime('%H:%M:%S')}]"
            except ValueError:
                pass
        lines.append(f"--- {label}{ts_str} ---")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    output_path.write_text("\n".join(lines))
    meta["messages"] = len(messages)
    meta["output"] = str(output_path.relative_to(REPO_ROOT))
    if first_ts and last_ts:
        duration = last_ts - first_ts
        meta["duration_mins"] = int(duration.total_seconds() // 60)
    if first_ts:
        meta["started"] = first_ts.strftime("%Y-%m-%d %H:%M")
    return meta


def list_sessions():
    """Print a table of all sessions."""
    sessions = []
    for f in sorted(SOURCE_DIR.glob("*.jsonl")):
        meta = read_session_meta(f)
        if meta and meta["type"] != "file-history-snapshot":
            sessions.append(meta)

    if not sessions:
        print("No sessions found.")
        return

    print(f"{'ID':10s} {'Team':22s} {'Agent':22s} {'Lines':>6s}")
    print("-" * 65)
    for s in sessions:
        print(f"{s['session_id'][:8]:10s} {s['team'] or '-':22s} {s['agent'] or '-':22s} {s['lines']:6d}")
    print(f"\n{len(sessions)} sessions total")


def main():
    if not SOURCE_DIR.exists():
        print(f"Source not found: {SOURCE_DIR}")
        sys.exit(1)

    if "--list" in sys.argv:
        list_sessions()
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return

    team_filter = None
    if "--team" in sys.argv:
        idx = sys.argv.index("--team")
        if idx + 1 < len(sys.argv):
            team_filter = sys.argv[idx + 1]
        else:
            print("Usage: --team NAME")
            sys.exit(1)

    target_id = None
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            target_id = arg
            break

    exported = []
    for f in sorted(SOURCE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        if target_id and target_id not in f.stem:
            continue
        if team_filter:
            meta = read_session_meta(f)
            if not meta or meta.get("team") != team_filter:
                continue

        result = export_session(f, TRACES_DIR)
        if result:
            exported.append(result)
            print(f"  {result['output']} ({result['messages']} messages)")

    if not exported:
        print("No sessions exported.")
        return

    # Write index
    index_path = TRACES_DIR / "INDEX.md"
    lines = [
        "# Session Traces",
        "",
        f"Exported {len(exported)} sessions from Claude Code on {datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        "",
        "| Team | Agent | Messages | Duration | Started | File |",
        "|------|-------|----------|----------|---------|------|",
    ]
    for s in exported:
        fname = Path(s["output"]).name
        dur = f"{s['duration_mins']}m" if s.get("duration_mins") is not None else "-"
        started = s.get("started", "-")
        lines.append(f"| {s['team'] or '-'} | {s['agent'] or '-'} | {s['messages']} | {dur} | {started} | [{fname}]({fname}) |")
    lines.append("")

    index_path.write_text("\n".join(lines))
    print(f"\n{len(exported)} sessions exported to .traces/sessions/")
    print(f"Index: {index_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
