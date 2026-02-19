#!/usr/bin/env python3
"""Superpod shard orchestrator: partition, merge, and run sharded pipelines.

Subcommands:
    merge   Combine N shard output directories into one merged directory
    run     Orchestrate: launch N parallel shard jobs → merge → post-merge stages

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
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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

def detect_gpu_count():
    """Detect number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return len(result.stdout.strip().splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def cmd_run(args):
    """Orchestrate: parallel shard jobs → merge → post-merge stages."""
    num_shards = args.num_shards
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Detect GPUs for assignment
    num_gpus = detect_gpu_count()
    print(f"[run] {num_shards} shards, {num_gpus} GPUs detected")
    print(f"[run] output: {outdir}")

    # Build shard output dirs
    shard_dirs = [Path(f"{outdir}-shard-{i}") for i in range(num_shards)]

    # Build base command (everything after --)
    script_dir = Path(__file__).parent
    base_cmd = [
        sys.executable, str(script_dir / "superpod-job.py"),
        args.posts_xml,
        "--site", args.site,
        "--embed-batch-size", str(args.embed_batch_size),
    ]
    if args.comments_xml:
        base_cmd += ["--comments-xml", args.comments_xml]
    if getattr(args, "input_dir", None):
        base_cmd += ["--input-dir", args.input_dir]
    # Pass through extra flags
    base_cmd += args.extra_args

    # Phase A: launch parallel shard jobs
    print(f"\n[run] Phase A: launching {num_shards} shard jobs...")
    t0 = time.time()
    processes = []
    for i in range(num_shards):
        shard_cmd = base_cmd + [
            "--output-dir", str(shard_dirs[i]),
            "--shard-index", str(i),
            "--num-shards", str(num_shards),
        ]
        env = os.environ.copy()
        if num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(i % num_gpus)
        print(f"  shard {i}: CUDA_VISIBLE_DEVICES={i % num_gpus if num_gpus else 'none'} "
              f"→ {shard_dirs[i]}")
        proc = subprocess.Popen(
            shard_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        processes.append((i, proc))

    # Wait for all shards, streaming output
    failed = []
    for i, proc in processes:
        stdout, _ = proc.communicate()
        if stdout:
            # Print shard output with prefix
            for line in stdout.decode(errors="replace").splitlines():
                print(f"  [shard-{i}] {line}")
        if proc.returncode != 0:
            failed.append(i)
            print(f"  [shard-{i}] FAILED (exit code {proc.returncode})")
        else:
            print(f"  [shard-{i}] completed OK")

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

    # Phase C: post-merge stages (9b + 10)
    hg_path = outdir / "hypergraphs.json"
    if hg_path.exists() and not args.skip_post_merge:
        print(f"\n[run] Phase C: post-merge stages (9b + 10)...")
        t2 = time.time()

        # Import and run stage 9b directly
        sys.path.insert(0, str(script_dir))
        sys.path.insert(0, str(script_dir.parent / "src"))

        from importlib import import_module
        spj = import_module("superpod-job")
        run_9b = spj.run_stage9b_graph_embedding
        run_10 = spj.run_stage10_faiss_index

        print(f"  [9b] Graph embedding (R-GCN, bs={args.graph_embed_batch_size}, "
              f"workers={args.graph_embed_workers})...")
        stats_9b, emb_path, model_path, thread_ids = run_9b(
            hg_path, outdir,
            embed_dim=args.graph_embed_dim,
            epochs=args.graph_embed_epochs,
            batch_size=args.graph_embed_batch_size,
            num_workers=args.graph_embed_workers,
        )
        print(f"  [9b] {stats_9b['n_embedded']} embeddings ({stats_9b['embed_dim']}d)")

        print(f"  [10] Building FAISS index...")
        stats_10, index_path = run_10(emb_path, thread_ids, outdir)
        print(f"  [10] {stats_10['n_vectors']} vectors indexed")

        # Update manifest with post-merge stages
        manifest_path = outdir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["stage9b_stats"] = stats_9b
        manifest["stage10_stats"] = stats_10
        manifest["stages_completed"] = manifest.get("stages_completed", []) + [
            "graph_embedding", "faiss_index"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"[run] Phase C complete in {time.time()-t2:.0f}s")
    elif args.skip_post_merge:
        print(f"\n[run] Phase C skipped (--skip-post-merge)")
    else:
        print(f"\n[run] Phase C skipped (no hypergraphs.json in merged output)")

    total = time.time() - t0
    print(f"\n[run] all phases complete in {total:.0f}s ({total/60:.1f} min)")


def cmd_post_merge(args):
    """Run only Phase C (stages 9b + 10) on an existing merged output directory."""
    outdir = Path(args.output_dir)
    hg_path = outdir / "hypergraphs.json"

    if not hg_path.exists():
        print(f"[post-merge] FATAL: {hg_path} not found. "
              f"Run 'merge' or full 'run' first.")
        sys.exit(1)

    print(f"[post-merge] Phase C on {outdir}")
    t0 = time.time()

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir.parent / "src"))

    from importlib import import_module
    spj = import_module("superpod-job")
    run_9b = spj.run_stage9b_graph_embedding
    run_10 = spj.run_stage10_faiss_index

    print(f"  [9b] Graph embedding (R-GCN, bs={args.graph_embed_batch_size}, "
          f"workers={args.graph_embed_workers})...")
    stats_9b, emb_path, model_path, thread_ids = run_9b(
        hg_path, outdir,
        embed_dim=args.graph_embed_dim,
        epochs=args.graph_embed_epochs,
        batch_size=args.graph_embed_batch_size,
        num_workers=args.graph_embed_workers,
    )
    print(f"  [9b] {stats_9b['n_embedded']} embeddings ({stats_9b['embed_dim']}d)")

    print(f"  [10] Building FAISS index...")
    stats_10, index_path = run_10(emb_path, thread_ids, outdir)
    print(f"  [10] {stats_10['n_vectors']} vectors indexed")

    # Update manifest
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["stage9b_stats"] = stats_9b
        manifest["stage10_stats"] = stats_10
        manifest["stages_completed"] = manifest.get("stages_completed", []) + [
            "graph_embedding", "faiss_index"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[post-merge] complete in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Superpod shard orchestrator: merge and run sharded pipelines")
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
    p_run.add_argument("--embed-batch-size", type=int, default=4096,
                       help="Embedding batch size per shard (default: 4096)")
    p_run.add_argument("--graph-embed-dim", type=int, default=128,
                       help="Hypergraph embedding dimension (default: 128)")
    p_run.add_argument("--graph-embed-epochs", type=int, default=50,
                       help="GNN training epochs (default: 50)")
    p_run.add_argument("--graph-embed-batch-size", type=int, default=512,
                       help="GNN training batch size (default: 512)")
    p_run.add_argument("--graph-embed-workers", type=int, default=4,
                       help="DataLoader workers for GNN training (default: 4, 0=inline)")
    p_run.add_argument("--input-dir", default=None,
                       help="Base directory for input data (Posts.xml, 7z files). "
                            "Use when data lives on /scratch/ or another filesystem.")
    p_run.add_argument("--skip-post-merge", action="store_true",
                       help="Skip post-merge stages 9b + 10")
    p_run.add_argument("extra_args", nargs="*",
                       help="Extra flags passed through to superpod-job.py "
                            "(put after --)")

    # --- post-merge (Phase C only) ---
    p_pm = sub.add_parser("post-merge",
                          help="Run only Phase C (9b + 10) on existing merged output")
    p_pm.add_argument("--output-dir", "-o", required=True,
                      help="Merged output directory (must contain hypergraphs.json)")
    p_pm.add_argument("--graph-embed-dim", type=int, default=128,
                      help="Hypergraph embedding dimension (default: 128)")
    p_pm.add_argument("--graph-embed-epochs", type=int, default=50,
                      help="GNN training epochs (default: 50)")
    p_pm.add_argument("--graph-embed-batch-size", type=int, default=512,
                      help="GNN training batch size (default: 512)")
    p_pm.add_argument("--graph-embed-workers", type=int, default=4,
                      help="DataLoader workers for GNN training (default: 4, 0=inline)")

    args = parser.parse_args()

    if args.command == "merge":
        cmd_merge(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "post-merge":
        cmd_post_merge(args)


if __name__ == "__main__":
    main()
