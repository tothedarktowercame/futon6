#!/usr/bin/env python3
"""Superpod shard orchestrator: partition, merge, and run sharded pipelines.

Subcommands:
    merge   Combine N shard output directories into one merged directory
    run     Orchestrate: launch N parallel shard jobs → merge → post-merge stages
    post-merge  Merge existing shard outputs (optional) → run post-merge stages

Usage:
    # Merge 4 shard directories:
    python scripts/superpod-shard.py merge \
        --shard-dirs ./out-shard-0 ./out-shard-1 ./out-shard-2 ./out-shard-3 \
        --output-dir ./out-merged

    # Run 8-way sharded pipeline:
    python scripts/superpod-shard.py run \
        --posts-xml ./se-data/math.stackexchange.com/Posts.xml \
        --comments-xml ./se-data/math.stackexchange.com/Comments.xml \
        --site math.stackexchange \
        --num-shards 8 \
        --output-dir ./math-processed \
        -- --embed-device cuda --skip-llm

    # Re-run merge + post-merge stages only (no shard jobs):
    python scripts/superpod-shard.py post-merge \
        --shard-dirs ./math-processed-shard-0 ./math-processed-shard-1 \
        --output-dir ./math-processed \
        --graph-embed-epochs 20 --graph-embed-batch-size 1024
"""

import argparse
import builtins as _builtins
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _timestamp_now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def print(*args, **kwargs):  # type: ignore[override]
    kwargs.setdefault("flush", True)
    return _builtins.print(f"[{_timestamp_now()}]", *args, **kwargs)


def _available_cpu_count() -> int:
    """CPU count available to this job, honoring Slurm/cpuset affinity."""
    counts = []
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        counts.append(int(slurm_cpus))
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if not counts:
        counts.append(os.cpu_count() or 1)
    return max(1, min(c for c in counts if c > 0))


def _default_loader_workers_per_shard(num_shards: int) -> int:
    """Split the Slurm/cpuset CPU allotment across concurrent shard procs."""
    return max(1, min(16, _available_cpu_count()) // max(1, num_shards))


def _has_option(args, option: str) -> bool:
    return any(a == option or a.startswith(option + "=") for a in args)


def _stream_shard_output(
    shard_idx: int,
    pipe,
    outq: "queue.Queue",
) -> None:
    """Read shard output lines and push to a central queue."""
    try:
        for line in iter(pipe.readline, ""):
            outq.put((shard_idx, line.rstrip("\n")))
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        outq.put((shard_idx, None))


# ---------------------------------------------------------------------------
# File merge strategies
# ---------------------------------------------------------------------------

# JSON array files: concatenate by stripping outer brackets
JSON_ARRAY_FILES = [
    "entities.json",
    "relations.json",
    "tags.json",
    "ner-terms.json",
    "scopes.json",
    "pattern-tags.json",
    "reverse-morphogenesis.json",
    "thread-wiring-ct.json",
    "expression-surfaces.json",
    "hypergraphs.json",
]

# JSON list files (simple — small enough to parse)
JSON_LIST_FILES = [
    "hypergraph-thread-ids.json",
]

# JSONL files: concatenate lines (trivial merge, fast multiprocessing)
JSONL_FILES = [
    "thread-wiring-ct.jsonl",
]

# numpy array files: concatenate along axis 0
NPY_FILES = [
    "embeddings.npy",
]


def merge_json_array_files(shard_dirs, filename, output_path):
    """Merge JSON array files by text concatenation (no parsing).

    The pipeline writes arrays as:
        [\\n
        {item1},\\n
        {item2}\\n
        ]

    We strip the outer brackets and join with commas.
    """
    chunks = []
    for d in shard_dirs:
        p = d / filename
        if not p.exists():
            continue
        content = p.read_text().strip()
        if not content or content == "[]":
            continue
        # Strip outer [ and ]
        if content.startswith("["):
            content = content[1:]
        if content.endswith("]"):
            content = content[:-1]
        content = content.strip()
        if content:
            chunks.append(content)

    if not chunks:
        return 0

    with open(output_path, "w") as f:
        f.write("[\n")
        f.write(",\n".join(chunks))
        f.write("\n]")

    return len(chunks)


def merge_json_lists(shard_dirs, filename, output_path):
    """Merge JSON list files by parsing and concatenating."""
    merged = []
    for d in shard_dirs:
        p = d / filename
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            merged.extend(data)
    if merged:
        with open(output_path, "w") as f:
            json.dump(merged, f)
    return len(merged)


def merge_jsonl_files(shard_dirs, filename, output_path):
    """Merge JSONL files by concatenating lines."""
    n_lines = 0
    with open(output_path, "w") as out:
        for d in shard_dirs:
            p = d / filename
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line)
                        out.write("\n")
                        n_lines += 1
    return n_lines


def merge_npy_files(shard_dirs, filename, output_path):
    """Merge numpy arrays by concatenation along axis 0."""
    arrays = []
    for d in shard_dirs:
        p = d / filename
        if not p.exists():
            continue
        arrays.append(np.load(str(p)))
    if not arrays:
        return None
    merged = np.concatenate(arrays, axis=0)
    np.save(str(output_path), merged)
    return merged.shape


def merge_stats(shard_dirs, output_path):
    """Merge stats.json by summing numeric fields."""
    merged = {}
    for d in shard_dirs:
        p = d / "stats.json"
        if not p.exists():
            continue
        with open(p) as f:
            stats = json.load(f)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                merged[k] = merged.get(k, 0) + v
            elif k not in merged:
                merged[k] = v
    if merged:
        with open(output_path, "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


# ---------------------------------------------------------------------------
# merge subcommand
# ---------------------------------------------------------------------------

def cmd_merge(args):
    """Merge N shard output directories into one."""
    shard_dirs = [Path(d) for d in args.shard_dirs]
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Validate shard dirs exist
    for d in shard_dirs:
        if not d.exists():
            print(f"[merge] ERROR: shard dir not found: {d}", file=sys.stderr)
            sys.exit(1)

    print(f"[merge] merging {len(shard_dirs)} shards into {outdir}")

    # 1. JSON array files (text concatenation — handles multi-GB files)
    for filename in JSON_ARRAY_FILES:
        present = [d for d in shard_dirs if (d / filename).exists()]
        if present:
            t0 = time.time()
            n = merge_json_array_files(shard_dirs, filename, outdir / filename)
            sz = os.path.getsize(outdir / filename) / 1e6
            print(f"  {filename}: {len(present)} shards, {sz:.1f} MB "
                  f"({time.time()-t0:.1f}s)")
        else:
            print(f"  {filename}: not present in any shard (skipped)")

    # 2. JSONL files (line concatenation — fast and multiprocess-friendly)
    for filename in JSONL_FILES:
        present = [d for d in shard_dirs if (d / filename).exists()]
        if present:
            t0 = time.time()
            n = merge_jsonl_files(shard_dirs, filename, outdir / filename)
            sz = os.path.getsize(outdir / filename) / 1e6
            print(f"  {filename}: {n} lines, {sz:.1f} MB ({time.time()-t0:.1f}s)")

    # 3. JSON list files (parse and concatenate)
    for filename in JSON_LIST_FILES:
        present = [d for d in shard_dirs if (d / filename).exists()]
        if present:
            n = merge_json_lists(shard_dirs, filename, outdir / filename)
            print(f"  {filename}: {n} items merged")

    # 3. numpy array files
    for filename in NPY_FILES:
        present = [d for d in shard_dirs if (d / filename).exists()]
        if present:
            shape = merge_npy_files(shard_dirs, filename, outdir / filename)
            print(f"  {filename}: shape {shape}")
        else:
            print(f"  {filename}: not present (skipped)")

    # 4. stats.json (sum numeric fields)
    merged_stats = merge_stats(shard_dirs, outdir / "stats.json")
    if merged_stats:
        print(f"  stats.json: {merged_stats.get('qa_pairs', '?')} QA pairs total")

    # 5. Build merged manifest
    shard_manifests = []
    for d in shard_dirs:
        mp = d / "manifest.json"
        if mp.exists():
            with open(mp) as f:
                shard_manifests.append(json.load(f))

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "merged": True,
        "num_shards": len(shard_dirs),
        "shard_dirs": [str(d) for d in shard_dirs],
        "entity_count": sum(m.get("entity_count", 0) for m in shard_manifests),
        "stats": merged_stats,
        "shard_manifests": shard_manifests,
        "output_files": [f.name for f in outdir.iterdir() if f.is_file()],
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  manifest.json: {manifest['entity_count']} entities total")

    print(f"[merge] done. Output: {outdir}")


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

def run_post_merge_stages(outdir: Path, graph_embed_dim: int,
                          graph_embed_epochs: int, graph_embed_batch_size: int,
                          graph_embed_workers: int = 0,
                          skip_post_merge: bool = False,
                          log_prefix: str = "[run]"):
    """Run Phase C (stages 9b + 10) against an existing merged output dir."""
    hg_path = outdir / "hypergraphs.json"
    if hg_path.exists() and not skip_post_merge:
        print(f"\n{log_prefix} Phase C: post-merge stages (9b + 10)...")
        t2 = time.time()

        script_dir = Path(__file__).parent
        # Import and run stage 9b directly
        sys.path.insert(0, str(script_dir))
        sys.path.insert(0, str(script_dir.parent / "src"))

        from importlib import import_module
        spj = import_module("superpod-job")
        run_9b = spj.run_stage9b_graph_embedding
        run_10 = spj.run_stage10_faiss_index

        print(f"  [9b] Graph embedding (R-GCN, {graph_embed_dim}d, "
              f"{graph_embed_epochs} epochs, batch={graph_embed_batch_size}, "
              f"workers={graph_embed_workers})...")
        stats_9b, emb_path, model_path, thread_ids = run_9b(
            hg_path, outdir,
            embed_dim=graph_embed_dim,
            epochs=graph_embed_epochs,
            batch_size=graph_embed_batch_size,
            num_workers=graph_embed_workers,
        )
        print(f"  [9b] {stats_9b['n_embedded']} embeddings ({stats_9b['embed_dim']}d)")

        print(f"  [10] Building FAISS index...")
        stats_10, index_path = run_10(emb_path, thread_ids, outdir)
        print(f"  [10] {stats_10['n_vectors']} vectors indexed")

        # Update manifest with post-merge stages
        manifest_path = outdir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["stage9b_stats"] = stats_9b
            manifest["stage10_stats"] = stats_10
            stages_completed = list(manifest.get("stages_completed", []))
            for stage_name in ("graph_embedding", "faiss_index"):
                if stage_name not in stages_completed:
                    stages_completed.append(stage_name)
            manifest["stages_completed"] = stages_completed
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"{log_prefix} Phase C complete in {time.time()-t2:.0f}s")
    elif skip_post_merge:
        print(f"\n{log_prefix} Phase C skipped (--skip-post-merge)")
    else:
        print(f"\n{log_prefix} Phase C skipped (no hypergraphs.json in merged output)")

def visible_gpu_devices():
    """Return GPU tokens this process is allowed to hand to shard workers."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [token.strip() for token in visible.split(",") if token.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


def cmd_run(args):
    """Orchestrate: parallel shard jobs → merge → post-merge stages."""
    num_shards = args.num_shards
    if args.graph_embed_batch_size <= 0:
        print("[run] ERROR: --graph-embed-batch-size must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.graph_embed_workers < 0:
        print("[run] ERROR: --graph-embed-workers must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.llm_loader_workers is None:
        env_loader_workers = os.environ.get("LLM_LOADER_WORKERS")
        if env_loader_workers:
            if not env_loader_workers.isdigit():
                print("[run] ERROR: LLM_LOADER_WORKERS must be >= 0", file=sys.stderr)
                sys.exit(2)
            args.llm_loader_workers = int(env_loader_workers)
        else:
            args.llm_loader_workers = _default_loader_workers_per_shard(num_shards)
    if args.llm_loader_workers < 0:
        print("[run] ERROR: --llm-loader-workers must be >= 0", file=sys.stderr)
        sys.exit(2)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Detect GPUs for assignment. Respect CUDA_VISIBLE_DEVICES if a launcher has
    # already narrowed the process to the Slurm-authorized allocation.
    gpu_devices = visible_gpu_devices()
    num_gpus = len(gpu_devices)
    print(f"[run] {num_shards} shards, {num_gpus} GPUs visible")
    print(f"[run] CPU affinity: {_available_cpu_count()} cores; "
          f"LLM loader workers per shard: {args.llm_loader_workers}")
    print(f"[run] output: {outdir}")

    # Build shard output dirs
    shard_dirs = [Path(f"{outdir}-shard-{i}") for i in range(num_shards)]

    # Build base command (everything after --)
    script_dir = Path(__file__).parent
    base_cmd = [
        sys.executable, "-u", str(script_dir / "superpod-job.py"),
        args.posts_xml,
        "--site", args.site,
        "--embed-batch-size", str(args.embed_batch_size),
    ]
    if args.comments_xml:
        base_cmd += ["--comments-xml", args.comments_xml]
    if getattr(args, "input_dir", None):
        base_cmd += ["--input-dir", args.input_dir]
    if not _has_option(args.extra_args, "--llm-loader-workers"):
        base_cmd += ["--llm-loader-workers", str(args.llm_loader_workers)]
    # Pass through extra flags
    base_cmd += args.extra_args

    # Phase A: launch parallel shard jobs
    print(f"\n[run] Phase A: launching {num_shards} shard jobs...")
    t0 = time.time()
    processes = []
    outq: queue.Queue = queue.Queue()
    stream_threads: dict[int, threading.Thread] = {}
    for i in range(num_shards):
        shard_cmd = base_cmd + [
            "--output-dir", str(shard_dirs[i]),
            "--shard-index", str(i),
            "--num-shards", str(num_shards),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        shard_gpu = gpu_devices[i % num_gpus] if num_gpus else None
        if shard_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = shard_gpu
        print(f"  shard {i}: CUDA_VISIBLE_DEVICES={shard_gpu if shard_gpu is not None else 'none'} "
              f"-> {shard_dirs[i]}")
        proc = subprocess.Popen(
            shard_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        processes.append((i, proc))
        if proc.stdout is None:
            raise RuntimeError(f"failed to capture stdout for shard {i}")
        t = threading.Thread(
            target=_stream_shard_output,
            args=(i, proc.stdout, outq),
            daemon=True,
        )
        t.start()
        stream_threads[i] = t

    # Wait for all shards, streaming output
    failed = []
    completed: set[int] = set()
    while len(completed) < len(processes):
        try:
            shard_idx, line = outq.get(timeout=0.2)
            if line is not None:
                print(f"  [shard-{shard_idx}] {line}")
        except queue.Empty:
            pass

        for i, proc in processes:
            if i in completed:
                continue
            rc = proc.poll()
            if rc is None:
                continue
            completed.add(i)
            if rc != 0:
                failed.append(i)
                print(f"  [shard-{i}] FAILED (exit code {rc})")
            else:
                print(f"  [shard-{i}] completed OK")

    # Drain any trailing lines that raced with process shutdown.
    for i, t in stream_threads.items():
        t.join(timeout=1.0)
    while True:
        try:
            shard_idx, line = outq.get_nowait()
            if line is not None:
                print(f"  [shard-{shard_idx}] {line}")
        except queue.Empty:
            break

    if failed:
        print(f"\n[run] FATAL: shards {failed} failed. Aborting.")
        sys.exit(1)

    print(f"\n[run] Phase A complete in {time.time()-t0:.0f}s")

    # Phase B: merge
    print(f"\n[run] Phase B: merging shards...")
    t1 = time.time()
    merge_args = argparse.Namespace(
        shard_dirs=[str(d) for d in shard_dirs],
        output_dir=str(outdir),
    )
    cmd_merge(merge_args)
    print(f"[run] Phase B complete in {time.time()-t1:.0f}s")

    run_post_merge_stages(
        outdir=outdir,
        graph_embed_dim=args.graph_embed_dim,
        graph_embed_epochs=args.graph_embed_epochs,
        graph_embed_batch_size=args.graph_embed_batch_size,
        graph_embed_workers=args.graph_embed_workers,
        skip_post_merge=args.skip_post_merge,
        log_prefix="[run]",
    )

    total = time.time() - t0
    print(f"\n[run] all phases complete in {total:.0f}s ({total/60:.1f} min)")


def cmd_post_merge(args):
    """Optionally merge existing shard outputs, then run stages 9b + 10."""
    if args.graph_embed_batch_size <= 0:
        print("[post] ERROR: --graph-embed-batch-size must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.graph_embed_workers < 0:
        print("[post] ERROR: --graph-embed-workers must be >= 0", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.output_dir)
    t0 = time.time()

    if args.shard_dirs:
        print(f"[post] Phase B: merging {len(args.shard_dirs)} shard dirs...")
        t1 = time.time()
        merge_args = argparse.Namespace(
            shard_dirs=args.shard_dirs,
            output_dir=args.output_dir,
        )
        cmd_merge(merge_args)
        print(f"[post] Phase B complete in {time.time()-t1:.0f}s")
    else:
        if not outdir.exists():
            print(f"[post] ERROR: output dir not found: {outdir}", file=sys.stderr)
            sys.exit(1)
        print(f"[post] Phase B skipped (no --shard-dirs provided). Using {outdir}")

    run_post_merge_stages(
        outdir=outdir,
        graph_embed_dim=args.graph_embed_dim,
        graph_embed_epochs=args.graph_embed_epochs,
        graph_embed_batch_size=args.graph_embed_batch_size,
        graph_embed_workers=args.graph_embed_workers,
        skip_post_merge=args.skip_post_merge,
        log_prefix="[post]",
    )

    total = time.time() - t0
    print(f"\n[post] complete in {total:.0f}s ({total/60:.1f} min)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Superpod shard orchestrator: merge, run, and post-merge pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- merge ---
    p_merge = sub.add_parser("merge", help="Merge N shard output directories")
    p_merge.add_argument("--shard-dirs", nargs="+", required=True,
                         help="Shard output directories to merge")
    p_merge.add_argument("--output-dir", "-o", required=True,
                         help="Merged output directory")

    # --- run ---
    p_run = sub.add_parser("run", help="Run sharded pipeline end-to-end")
    p_run.add_argument("--posts-xml", required=True,
                       help="Path to Posts.xml")
    p_run.add_argument("--comments-xml", default=None,
                       help="Path to Comments.xml")
    p_run.add_argument("--site", default="math.stackexchange",
                       help="SE site name")
    p_run.add_argument("--num-shards", type=int, required=True,
                       help="Number of shards")
    p_run.add_argument("--output-dir", "-o", required=True,
                       help="Final merged output directory")
    p_run.add_argument("--embed-batch-size", type=int, default=2048,
                       help="Embedding batch size per shard (default: 2048)")
    p_run.add_argument("--graph-embed-dim", type=int, default=128,
                       help="Hypergraph embedding dimension (default: 128)")
    p_run.add_argument("--graph-embed-epochs", type=int, default=50,
                       help="GNN training epochs (default: 50)")
    p_run.add_argument("--graph-embed-batch-size", type=int, default=1024,
                       help="Batch size for post-merge Stage 9b graph embedding (default: 1024)")
    p_run.add_argument("--graph-embed-workers", type=int, default=16,
                       help="CPU workers for Stage 9b batch prep (default: 16)")
    p_run.add_argument("--llm-loader-workers", type=int, default=None,
                       help="Python workers feeding each shard's Dataset-backed "
                            "transformers pipelines. Default: split "
                            "min(16, Slurm/cpuset CPU affinity) across shards.")
    p_run.add_argument("--input-dir", default=None,
                       help="Base directory for input data (Posts.xml, 7z files). "
                            "Use when data lives on /scratch/ or another filesystem.")
    p_run.add_argument("--skip-post-merge", action="store_true",
                       help="Skip post-merge stages 9b + 10")
    p_run.add_argument("extra_args", nargs="*",
                       help="Extra flags passed through to superpod-job.py "
                            "(put after --)")

    # --- post-merge ---
    p_post = sub.add_parser(
        "post-merge",
        help="Merge existing shards (optional) and run post-merge stages 9b + 10",
    )
    p_post.add_argument("--output-dir", "-o", required=True,
                        help="Merged output directory (existing or merge target)")
    p_post.add_argument("--shard-dirs", nargs="*", default=None,
                        help="Optional shard output directories to merge before post-merge stages")
    p_post.add_argument("--graph-embed-dim", type=int, default=128,
                        help="Hypergraph embedding dimension (default: 128)")
    p_post.add_argument("--graph-embed-epochs", type=int, default=50,
                        help="GNN training epochs (default: 50)")
    p_post.add_argument("--graph-embed-batch-size", type=int, default=1024,
                        help="Batch size for Stage 9b graph embedding (default: 1024)")
    p_post.add_argument("--graph-embed-workers", type=int, default=16,
                        help="CPU workers for Stage 9b batch prep (default: 16)")
    p_post.add_argument("--skip-post-merge", action="store_true",
                        help="Skip stages 9b + 10 (useful for merge-only refresh)")

    args = parser.parse_args()

    if args.command == "merge":
        cmd_merge(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "post-merge":
        cmd_post_merge(args)


if __name__ == "__main__":
    main()
