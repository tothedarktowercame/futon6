#!/usr/bin/env python3
"""A7: Analyze Stage 8 LaTeX parse failures by construct family.

Streams expression-surfaces.json and categorizes fallback (failed) expressions
by the LaTeX constructs they contain to identify systematic parsing gaps.

Usage:
    python scripts/analyze-latex-failures.py /path/to/processed-gpu/
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# LaTeX construct patterns to check
CONSTRUCT_PATTERNS = {
    "multi-letter command": re.compile(r"\\[a-zA-Z]{4,}"),
    "environment": re.compile(r"\\begin\{"),
    "matrix/array": re.compile(r"\\begin\{(matrix|pmatrix|bmatrix|array|cases|align)"),
    "frac/binom": re.compile(r"\\(frac|binom|dfrac|tfrac)\{"),
    "sqrt": re.compile(r"\\sqrt"),
    "sum/prod/int": re.compile(r"\\(sum|prod|int|oint|iint|iiint)"),
    "limits/subscript chain": re.compile(r"_\{[^}]*_"),
    "text command": re.compile(r"\\(text|mathrm|mathbf|mathcal|mathbb|mathfrak|operatorname)\{"),
    "spacing command": re.compile(r"\\(quad|qquad|,|;|!|hspace|vspace)"),
    "decorators": re.compile(r"\\(hat|bar|tilde|dot|vec|overline|underline|widehat|widetilde)\{"),
    "over/underset": re.compile(r"\\(overset|underset|stackrel)\{"),
    "tensor index": re.compile(r"[_^]\{[^}]*\}[_^]\{[^}]*\}"),
    "colon types": re.compile(r"\\(colon|colonequals)"),
    "arrows": re.compile(r"\\(xrightarrow|xleftarrow|xmapsto|longrightarrow|hookrightarrow)"),
    "delimiters": re.compile(r"\\(left|right|big|Big|bigg|Bigg)[(\[|.\\]"),
    "color/cancel": re.compile(r"\\(color|cancel|boxed)\{"),
    "multi-line": re.compile(r"\\\\"),
    "ampersand": re.compile(r"&"),
    "plain colon-arrow": re.compile(r":\\s*\\(to|rightarrow|mapsto)"),
}


def stream_threads(expr_path: Path):
    """Stream threads from expression-surfaces.json without loading all into memory.

    The file is a JSON array of objects. We use ijson if available,
    otherwise fall back to line-by-line heuristic for JSONL, or
    chunked JSON array parsing.
    """
    # Try ijson first for true streaming
    try:
        import ijson
        with open(expr_path, "rb") as f:
            yield from ijson.items(f, "item")
        return
    except ImportError:
        pass

    # Fallback: load in chunks if file is small enough, else use a
    # simple streaming approach for JSON arrays
    import mmap
    file_size = expr_path.stat().st_size
    if file_size < 500_000_000:  # < 500MB, load normally
        with open(expr_path) as f:
            data = json.load(f)
        yield from data
        return

    # For large files, use a decoder that yields objects one at a time
    print(f"  (streaming {file_size / 1e9:.1f}GB file...)")
    decoder = json.JSONDecoder()
    with open(expr_path) as f:
        # Skip opening bracket
        content = f.read(1)
        while content and content.strip() in ('[', ',', ' ', '\n'):
            content = f.read(1)

        buf = content
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            buf += chunk
            buf = buf.lstrip(' ,\n\r\t')
            while buf and buf[0] == '{':
                try:
                    obj, end = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[end:].lstrip(' ,\n\r\t')
                except json.JSONDecodeError:
                    break  # need more data


def analyze_failures(data_dir: Path):
    expr_path = data_dir / "expression-surfaces.json"
    print(f"Loading {expr_path}...")

    threads = stream_threads(expr_path)

    total_expressions = 0
    total_fallbacks = 0
    construct_counts = Counter()
    length_buckets = Counter()  # length distribution of failures
    sample_failures = []  # keep first 20 for inspection

    for thread in threads:
        for expr in thread.get("expressions", []):
            total_expressions += 1
            if not expr.get("fallback", False):
                continue

            total_fallbacks += 1
            latex = expr.get("latex", "")

            # Length bucket
            if len(latex) < 20:
                length_buckets["<20 chars"] += 1
            elif len(latex) < 50:
                length_buckets["20-50 chars"] += 1
            elif len(latex) < 100:
                length_buckets["50-100 chars"] += 1
            elif len(latex) < 200:
                length_buckets["100-200 chars"] += 1
            else:
                length_buckets["200+ chars"] += 1

            # Check which constructs are present
            matched_any = False
            for name, pattern in CONSTRUCT_PATTERNS.items():
                if pattern.search(latex):
                    construct_counts[name] += 1
                    matched_any = True

            if not matched_any:
                construct_counts["(no known pattern)"] += 1

            if len(sample_failures) < 20:
                sample_failures.append(latex[:120])

    # Report
    fail_rate = total_fallbacks / total_expressions * 100 if total_expressions else 0
    print(f"\nTotal expressions: {total_expressions:,}")
    print(f"Fallbacks (failures): {total_fallbacks:,} ({fail_rate:.2f}%)")

    print(f"\n=== Failure Length Distribution ===")
    for bucket, count in sorted(length_buckets.items()):
        pct = count / total_fallbacks * 100 if total_fallbacks else 0
        print(f"  {bucket:>15s}: {count:>6,} ({pct:5.1f}%)")

    print(f"\n=== Construct Frequency in Failures ===")
    print(f"(percentages are of total failures; constructs can overlap)")
    for name, count in construct_counts.most_common(25):
        pct = count / total_fallbacks * 100 if total_fallbacks else 0
        print(f"  {name:>30s}: {count:>6,} ({pct:5.1f}%)")

    print(f"\n=== Sample Failures (first 20) ===")
    for i, latex in enumerate(sample_failures, 1):
        print(f"  {i:2d}. {latex}")

    return {
        "total_expressions": total_expressions,
        "total_fallbacks": total_fallbacks,
        "fail_rate_pct": round(fail_rate, 2),
        "construct_counts": dict(construct_counts.most_common()),
        "length_buckets": dict(length_buckets),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze LaTeX parse failures")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    stats = analyze_failures(args.data_dir)

    # Save report
    out_path = args.data_dir / "latex-failure-analysis.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
