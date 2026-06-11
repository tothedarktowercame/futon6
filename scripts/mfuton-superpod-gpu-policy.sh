#!/usr/bin/env bash
#
# Superpod GPU policy gate for futon6 launchers.
#
# Authority lives in mfuton:
#   agent_skills/development/superpod/current-job-gpus.sh
#
# This helper fails fast if the current Slurm job allocation cannot be resolved.
# It exports CUDA_VISIBLE_DEVICES to exactly the current-host GPU IDs Slurm
# allocated to the job. On dev/short partitions that must be a single GPU.

mfuton_superpod_find_home() {
  if [[ -n "${MFUTON_HOME:-}" ]]; then
    printf '%s\n' "$MFUTON_HOME"
    return 0
  fi

  local candidate
  for candidate in \
    "$HOME/gh/mfuton" \
    "/users/rjmeyers/gh/mfuton" \
    "/home/rjmeyers/gh/mfuton"
  do
    if [[ -x "$candidate/agent_skills/development/superpod/current-job-gpus.sh" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "[gpu-policy] FATAL: MFUTON_HOME is unset and mfuton was not found in known locations." >&2
  return 1
}

mfuton_superpod_apply_gpu_policy() {
  local requested="${1:-slurm-current-job}"
  local mfuton_home
  mfuton_home="$(mfuton_superpod_find_home)"

  local skill="$mfuton_home/agent_skills/development/superpod/current-job-gpus.sh"
  if [[ ! -x "$skill" ]]; then
    echo "[gpu-policy] FATAL: GPU allocation skill is not executable: $skill" >&2
    return 1
  fi

  local info_json
  if ! info_json="$(MFUTON_HOME="$mfuton_home" "$skill" --format json --request "$requested")"; then
    echo "[gpu-policy] FATAL: failed to resolve current Slurm GPU allocation." >&2
    echo "$info_json" >&2
    return 1
  fi

  local parsed
  if ! parsed="$(GPU_INFO_JSON="$info_json" python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["GPU_INFO_JSON"])
if not payload.get("ok"):
    raise SystemExit(payload.get("error") or "GPU allocation skill returned ok=false")

ids = payload.get("selected_gpu_ids")
if ids is None:
    ids = payload.get("current_host_gpu_ids")
if not isinstance(ids, list) or not ids:
    raise SystemExit(f"no current-host GPU IDs in allocation payload: {payload!r}")
if not all(isinstance(value, int) and value >= 0 for value in ids):
    raise SystemExit(f"invalid GPU ID list in allocation payload: {ids!r}")

partition = str(payload.get("partition") or "")
if partition.rstrip("*").lower() in {"short", "dev"} and len(ids) != 1:
    raise SystemExit(
        f"policy violation: partition {partition!r} must expose exactly one GPU, got {ids}"
    )

print(",".join(str(value) for value in ids))
print(str(len(ids)))
print(partition)
print(str(payload.get("job_id") or ""))
print(str(payload.get("current_host") or ""))
PY
  )"; then
    echo "[gpu-policy] FATAL: invalid current Slurm GPU allocation payload." >&2
    echo "$info_json" >&2
    return 1
  fi

  local gpu_ids gpu_count partition job_id current_host
  gpu_ids="$(sed -n '1p' <<<"$parsed")"
  gpu_count="$(sed -n '2p' <<<"$parsed")"
  partition="$(sed -n '3p' <<<"$parsed")"
  job_id="$(sed -n '4p' <<<"$parsed")"
  current_host="$(sed -n '5p' <<<"$parsed")"

  export CUDA_VISIBLE_DEVICES="$gpu_ids"
  export MFUTON_SUPERPOD_GPU_IDS="$gpu_ids"
  export MFUTON_SUPERPOD_GPU_COUNT="$gpu_count"
  export MFUTON_SUPERPOD_PARTITION="$partition"
  export MFUTON_SUPERPOD_JOB_ID="$job_id"
  export MFUTON_SUPERPOD_HOST="$current_host"

  echo "[gpu-policy] job=$job_id partition=$partition host=$current_host CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES count=$gpu_count"
}
