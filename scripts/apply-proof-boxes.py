#!/usr/bin/env python3
"""
Apply standoff annotations to clean .tex files, generating boxed output.

Reads a JSON manifest of annotations (box type, title, position anchors)
and applies them to clean LaTeX source files, producing output files with
tcolorbox environments inserted at the specified locations.

Usage:
    python3 scripts/apply-proof-boxes.py [--manifest PATH] [--output-dir PATH] [--dry-run]

Defaults:
    --manifest   data/first-proof/latex/standoff-boxes.json
    --output-dir data/first-proof/latex/full-boxed/
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

PRETTY_TYPES = {
    "processnote": "Process Note",
    "revision": "Revision",
    "openobligation": "Open Obligation",
    "deadpath": "Dead Path",
}

# Heading pattern: \section{...}, \subsection{...}, \subsubsection{...}
HEADING_RE = re.compile(
    r"\\(section|subsection|subsubsection)"
    r"(?:\{|(?:\[.*?\]\{))"   # optional [...] before {
)

HEADING_LEVEL = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
}


def parse_heading(line):
    """Return (level, heading_text) if line contains a section heading, else None."""
    m = HEADING_RE.search(line)
    if not m:
        # Also handle \texorpdfstring variant
        m2 = re.search(r"\\(section|subsection|subsubsection)\{", line)
        if not m2:
            return None
        kind = m2.group(1)
    else:
        kind = m.group(1)
    level = HEADING_LEVEL[kind]
    # Extract heading text (everything between the outermost braces)
    brace_start = line.find("{", line.find("\\" + kind))
    if brace_start < 0:
        return None
    depth = 0
    for i in range(brace_start, len(line)):
        if line[i] == "{":
            depth += 1
        elif line[i] == "}":
            depth -= 1
            if depth == 0:
                text = line[brace_start + 1 : i]
                return (level, text)
    return None


def find_heading_line(lines, heading_substr):
    """Find the line index containing a section/subsection/subsubsection
    whose heading text contains heading_substr."""
    for i, line in enumerate(lines):
        parsed = parse_heading(line)
        if parsed is not None:
            _, text = parsed
            if heading_substr in text:
                return i
    return None


def find_section_end(lines, start_idx):
    """Find the end of a section starting at start_idx.
    Returns the index of the next heading at same or higher level, or len(lines)."""
    parsed = parse_heading(lines[start_idx])
    if parsed is None:
        return len(lines)
    start_level = parsed[0]
    for i in range(start_idx + 1, len(lines)):
        parsed_i = parse_heading(lines[i])
        if parsed_i is not None and parsed_i[0] <= start_level:
            return i
    return len(lines)


def find_marker_line(lines, marker, start_from=0):
    """Find the first line at or after start_from containing marker as a substring."""
    for i in range(start_from, len(lines)):
        if marker in lines[i]:
            return i
    return None


def format_box_title(box_type, title):
    """Format the full box title string."""
    pretty = PRETTY_TYPES.get(box_type, box_type)
    return f"{pretty} --- {title}"


def apply_annotations(lines, annotations, filename, dry_run=False):
    """Apply a list of annotations to the lines of a file.
    Returns the modified lines list and a count of applied annotations."""
    # First pass: resolve positions for all annotations
    resolved = []
    for ann in annotations:
        mode = ann["mode"]
        box = ann["box"]
        title = ann["title"]
        full_title = format_box_title(box, title)

        if mode == "section":
            heading = ann["heading"]
            start = find_heading_line(lines, heading)
            if start is None:
                print(f"  WARNING: heading '{heading}' not found in {filename}", file=sys.stderr)
                continue
            end = find_section_end(lines, start)
            resolved.append({
                "start": start,
                "end": end,
                "mode": "section",
                "box": box,
                "full_title": full_title,
                "desc": f"section '{heading}'",
            })
        elif mode == "block":
            start_marker = ann["start_marker"]
            end_before = ann.get("end_before")
            start = find_marker_line(lines, start_marker)
            if start is None:
                print(f"  WARNING: start_marker '{start_marker}' not found in {filename}", file=sys.stderr)
                continue
            if end_before:
                end = find_marker_line(lines, end_before, start_from=start + 1)
                if end is None:
                    print(f"  WARNING: end_before '{end_before}' not found after line {start} in {filename}", file=sys.stderr)
                    continue
            else:
                end = len(lines)
            resolved.append({
                "start": start,
                "end": end,
                "mode": "block",
                "box": box,
                "full_title": full_title,
                "desc": f"block '{start_marker[:40]}...'",
            })
        else:
            print(f"  WARNING: unknown mode '{mode}' in {filename}", file=sys.stderr)
            continue

    if not resolved:
        return lines, 0

    # Sort by start position descending (bottom-to-top)
    resolved.sort(key=lambda r: r["start"], reverse=True)

    if dry_run:
        for r in reversed(resolved):
            print(f"  [{r['box']}] line {r['start']+1}-{r['end']}: {r['desc']}")
        return lines, len(resolved)

    # Apply bottom-to-top
    modified = list(lines)
    for r in resolved:
        start = r["start"]
        end = r["end"]
        box = r["box"]
        full_title = r["full_title"]
        begin_line = f"\\begin{{{box}}}[title={{{full_title}}}]\n"
        end_line = f"\\end{{{box}}}\n"

        # Insert \end{box} before end line (or at EOF)
        if end < len(modified):
            modified.insert(end, end_line)
        else:
            # Ensure file ends with newline before appending
            if modified and not modified[-1].endswith("\n"):
                modified[-1] += "\n"
            modified.append(end_line)

        if r["mode"] == "section":
            # Replace the heading line with \begin{box}
            modified[start] = begin_line
        else:
            # Block mode: insert \begin{box} before the start line
            modified.insert(start, begin_line)

    return modified, len(resolved)


def main():
    parser = argparse.ArgumentParser(
        description="Apply standoff box annotations to clean .tex files"
    )
    parser.add_argument(
        "--manifest",
        default="data/first-proof/latex/standoff-boxes.json",
        help="Path to the annotation manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="data/first-proof/latex/full-boxed/",
        help="Output directory for boxed .tex files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files",
    )
    args = parser.parse_args()

    # Load manifest
    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(os.getcwd(), manifest_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    annotations = manifest["annotations"]
    print(f"Loaded {len(annotations)} annotations from {manifest_path}")

    # Resolve base directory (directory containing the manifest)
    base_dir = os.path.dirname(manifest_path)

    # Group annotations by file
    by_file = defaultdict(list)
    for ann in annotations:
        by_file[ann["file"]].append(ann)

    # Resolve output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # Also copy files that have no annotations (so full-boxed/ has all files)
    all_full_files = set()
    full_dir = os.path.join(base_dir, "full")
    if os.path.isdir(full_dir):
        for fname in os.listdir(full_dir):
            if fname.endswith(".tex"):
                all_full_files.add(f"full/{fname}")

    total_applied = 0
    files_processed = 0

    # Process annotated files
    for filepath, file_annotations in sorted(by_file.items()):
        abs_path = os.path.join(base_dir, filepath)
        if not os.path.exists(abs_path):
            print(f"ERROR: file not found: {abs_path}", file=sys.stderr)
            continue

        with open(abs_path) as f:
            lines = f.readlines()

        basename = os.path.basename(filepath)
        print(f"\n{basename}: {len(file_annotations)} annotation(s)")

        modified, count = apply_annotations(lines, file_annotations, basename, dry_run=args.dry_run)
        total_applied += count

        if not args.dry_run:
            out_path = os.path.join(output_dir, basename)
            with open(out_path, "w") as f:
                f.writelines(modified)
            print(f"  -> wrote {out_path}")

        files_processed += 1
        all_full_files.discard(filepath)

    # Copy unannotated files to output (so full-boxed/ is a complete mirror)
    for filepath in sorted(all_full_files):
        abs_path = os.path.join(base_dir, filepath)
        basename = os.path.basename(filepath)
        if not args.dry_run:
            with open(abs_path) as f:
                content = f.read()
            out_path = os.path.join(output_dir, basename)
            with open(out_path, "w") as f:
                f.write(content)
            print(f"\n{basename}: 0 annotations (copied unchanged)")
        files_processed += 1

    print(f"\nDone: {total_applied} annotations applied across {files_processed} files.")
    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
