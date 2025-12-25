#!/usr/bin/env python3
"""
Generate a single markdown file containing the entire codebase
for high-level architecture conversations with AI.

Usage:
    python scripts/generate_codebase_snapshot.py

Output:
    docs/CODEBASE_SNAPSHOT.md
"""

import os
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / "docs" / "CODEBASE_SNAPSHOT.md"

# Files/dirs to skip
SKIP_DIRS = {
    '.venv', 'venv', '__pycache__', '.git', 'node_modules',
    '.pytest_cache', 'data', 'outputs', 'logs', 'reports',
    '_archive', 'backups', 'Rlib', '.cursor', '.vscode',
    'nfl_quant.egg-info', 'deprecated'
}

SKIP_FILES = {'.DS_Store', '*.pyc', '*.pkl', '*.parquet', '*.csv', '*.joblib'}

INCLUDE_EXTENSIONS = {'.py', '.md', '.yaml', '.yml', '.toml', '.json', '.r', '.R', '.sh'}

# Max lines per file (truncate large files)
MAX_LINES = 200

# Skip files larger than this (bytes)
MAX_FILE_SIZE = 50000


def should_include_dir(path: Path) -> bool:
    """Check if directory should be traversed."""
    return path.name not in SKIP_DIRS


def should_include_file(path: Path) -> bool:
    """Check if file should be included."""
    # Check extension
    if path.suffix not in INCLUDE_EXTENSIONS:
        return False

    # Check if in skipped directory
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False

    # Check file size
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
    except OSError:
        return False

    return True


def truncate_content(content: str, max_lines: int = MAX_LINES) -> tuple:
    """
    Truncate file content if too long.

    Returns:
        (truncated_content, was_truncated)
    """
    lines = content.split('\n')
    if len(lines) <= max_lines:
        return content, False

    half = max_lines // 2
    truncated = '\n'.join(
        lines[:half] +
        [f'\n# ... [{len(lines) - max_lines} lines truncated] ...\n'] +
        lines[-half:]
    )
    return truncated, True


def get_language(filepath: Path) -> str:
    """Get markdown language identifier for syntax highlighting."""
    ext_map = {
        '.py': 'python',
        '.r': 'r',
        '.R': 'r',
        '.sh': 'bash',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown',
    }
    return ext_map.get(filepath.suffix, '')


def generate_tree(root_path: Path, prefix: str = "") -> list:
    """Generate tree structure as list of strings."""
    output = []

    try:
        entries = sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return output

    # Filter entries
    dirs = [e for e in entries if e.is_dir() and should_include_dir(e)]
    files = [e for e in entries if e.is_file() and should_include_file(e)]

    items = dirs + files

    for i, entry in enumerate(items):
        is_last = (i == len(items) - 1)
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "

        if entry.is_dir():
            output.append(f"{prefix}{current_prefix}{entry.name}/")
            output.extend(generate_tree(entry, prefix + next_prefix))
        else:
            output.append(f"{prefix}{current_prefix}{entry.name}")

    return output


def generate_snapshot():
    """Generate the codebase snapshot."""
    print(f"Generating codebase snapshot...")
    print(f"Project root: {PROJECT_ROOT}")

    output_lines = [
        "# NFL QUANT Codebase Snapshot",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Project**: {PROJECT_ROOT.name}",
        "",
        "This file contains a snapshot of the codebase for AI-assisted architecture discussions.",
        "Large files are truncated. Data files are excluded.",
        "",
        "---",
        "",
        "## Directory Structure",
        "",
        "```"
    ]

    # Generate tree
    output_lines.append(f"{PROJECT_ROOT.name}/")
    output_lines.extend(generate_tree(PROJECT_ROOT))
    output_lines.append("```")

    output_lines.extend(["", "---", "", "## Key Configuration Files", ""])

    # Priority files to include first
    priority_files = [
        'configs/model_config.py',
        'nfl_quant/config_paths.py',
        'nfl_quant/__init__.py',
        'scripts/run_pipeline.py',
        'pyproject.toml',
        'requirements.txt',
    ]

    included_files = set()

    # Add priority files first
    for rel_path in priority_files:
        filepath = PROJECT_ROOT / rel_path
        if filepath.exists() and should_include_file(filepath):
            included_files.add(str(filepath))
            output_lines.extend([
                f"### `{rel_path}`",
                "",
                f"```{get_language(filepath)}"
            ])

            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                truncated, was_truncated = truncate_content(content)
                output_lines.append(truncated)
                if was_truncated:
                    output_lines.append(f"# File truncated (>{MAX_LINES} lines)")
            except Exception as e:
                output_lines.append(f"# Error reading file: {e}")

            output_lines.extend(["```", ""])

    output_lines.extend(["---", "", "## Source Files", ""])

    # Add remaining source files
    file_count = 0
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)

        # Filter directories in-place
        dirs[:] = [d for d in dirs if should_include_dir(root_path / d)]

        for filename in sorted(files):
            filepath = root_path / filename

            if str(filepath) in included_files:
                continue

            if not should_include_file(filepath):
                continue

            relative_path = filepath.relative_to(PROJECT_ROOT)

            output_lines.extend([
                f"### `{relative_path}`",
                "",
                f"```{get_language(filepath)}"
            ])

            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                truncated, was_truncated = truncate_content(content)
                output_lines.append(truncated)
                if was_truncated:
                    output_lines.append(f"# File truncated (>{MAX_LINES} lines)")
            except Exception as e:
                output_lines.append(f"# Error reading file: {e}")

            output_lines.extend(["```", ""])
            file_count += 1

    # Summary
    output_lines.extend([
        "---",
        "",
        "## Summary",
        "",
        f"- **Files included**: {file_count + len(priority_files)}",
        f"- **Max lines per file**: {MAX_LINES}",
        f"- **Excluded directories**: {', '.join(sorted(SKIP_DIRS))}",
        "",
        "---",
        "",
        "*Generated by `scripts/generate_codebase_snapshot.py`*"
    ])

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    # Write output
    OUTPUT_FILE.write_text('\n'.join(output_lines))

    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    line_count = len(output_lines)

    print(f"")
    print(f"Generated: {OUTPUT_FILE}")
    print(f"  Size: {file_size_kb:.1f} KB")
    print(f"  Lines: {line_count}")
    print(f"  Files: {file_count + len(priority_files)}")


if __name__ == "__main__":
    generate_snapshot()
