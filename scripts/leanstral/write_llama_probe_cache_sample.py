import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from deepspec.data.target_cache_dataset import (
    CacheCollator,
    CacheDataset,
    LocalCacheWriteSummary,
    LocalTargetCacheWriter,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE_DIR = (
    "/srv/models-hdd/models/leanstral-nvfp4/probes/"
    "20260704T013721Z-llama-dspark-cache-probe-full"
)
DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/target-cache-probes"


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_git(args):
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def parse_layers(raw: str):
    layers = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid decreasing layer range {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    if not layers:
        raise ValueError("At least one layer id is required.")
    return layers


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def summarize_array(array: np.ndarray):
    finite = np.isfinite(array)
    finite_values = array[finite]
    if finite_values.size:
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        mean_value = float(finite_values.mean(dtype=np.float64))
        l2_value = float(np.sqrt(np.square(finite_values, dtype=np.float64).sum()))
    else:
        min_value = 0.0
        max_value = 0.0
        mean_value = 0.0
        l2_value = 0.0
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "finite": int(finite.sum()),
        "nonfinite": int(array.size - finite.sum()),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "l2": l2_value,
    }


def read_f32_matrix(path: Path, *, seq_len: int, hidden_size: int) -> np.ndarray:
    array = np.fromfile(path, dtype=np.float32)
    expected = int(seq_len) * int(hidden_size)
    if array.size != expected:
        raise ValueError(
            f"{path} has {array.size} float32 values, expected {expected}."
        )
    array = array.reshape(int(seq_len), int(hidden_size))
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains nonfinite values.")
    return array


def layer_file_from_probe(probe_dir: Path, probe_json, layer_id: int) -> Path:
    layers = probe_json["outputs"]["layers"]
    for item in layers:
        if int(item["layer"]) == int(layer_id):
            path = Path(item["path"])
            if not path.is_absolute():
                path = probe_dir / path
            return path
    raise KeyError(f"Layer {layer_id} not present in probe output.")


def last_hidden_file_from_probe(probe_dir: Path, probe_json) -> Path:
    last_hidden = probe_json["outputs"].get("target_last_hidden_state")
    if not last_hidden:
        raise KeyError("Probe output does not contain target_last_hidden_state.")
    path = Path(last_hidden["path"])
    if not path.is_absolute():
        path = probe_dir / path
    return path


def choose_output_dir(output_root: Path, output_dir: str | None, label: str) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-{label}").resolve()


def verify_cache(cache_dir: Path, source_tensors):
    dataset = CacheDataset(str(cache_dir))
    try:
        sample = dataset[0]
        batch = CacheCollator()([sample])
        source_hidden, source_last_hidden = source_tensors
        read_hidden = sample["target_hidden_states"].to(torch.float32)
        read_last_hidden = sample["target_last_hidden_states"].to(torch.float32)
        hidden_delta = (
            read_hidden - torch.from_numpy(source_hidden).to(torch.float32)
        ).abs()
        last_hidden_delta = (
            read_last_hidden - torch.from_numpy(source_last_hidden).to(torch.float32)
        ).abs()
        return {
            "status": "ok",
            "dataset_len": len(dataset),
            "sample": {
                "input_ids_shape": [int(dim) for dim in sample["input_ids"].shape],
                "loss_mask_shape": [int(dim) for dim in sample["loss_mask"].shape],
                "target_hidden_states_shape": [
                    int(dim) for dim in sample["target_hidden_states"].shape
                ],
                "target_last_hidden_states_shape": [
                    int(dim) for dim in sample["target_last_hidden_states"].shape
                ],
                "target_hidden_states_dtype": str(
                    sample["target_hidden_states"].dtype
                ).replace("torch.", ""),
                "target_last_hidden_states_dtype": str(
                    sample["target_last_hidden_states"].dtype
                ).replace("torch.", ""),
                "max_abs_delta_hidden_vs_f32_source": float(hidden_delta.max().item()),
                "max_abs_delta_last_hidden_vs_f32_source": float(
                    last_hidden_delta.max().item()
                ),
            },
            "batch": {
                key: [int(dim) for dim in value.shape]
                for key, value in batch.items()
            },
        }
    finally:
        dataset.close()


def main():
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    parser = argparse.ArgumentParser(
        description=(
            "Write one DeepSpec target-cache sample from a recorded llama.cpp "
            "Leanstral DSpark activation probe."
        )
    )
    parser.add_argument("--probe-dir", default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--layers", default="1,9,17,25,33")
    parser.add_argument(
        "--loss-mask",
        choices=("all", "not-bos"),
        default="all",
        help="Probe-only loss mask policy. This is not a training-data recipe.",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=1024**3)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    probe_dir = Path(args.probe_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_dir = choose_output_dir(
        output_root=output_root,
        output_dir=args.output_dir,
        label="llama-probe-cache-sample",
    )
    layer_ids = parse_layers(args.layers)
    probe_json_path = probe_dir / "probe_result.json"
    probe_json = load_json(probe_json_path)

    seq_len = int(probe_json["tokenization"]["n_tokens"])
    tokens = np.asarray(probe_json["tokenization"]["tokens"], dtype=np.int32)
    hidden_size = int(probe_json["model"]["n_embd"])
    if tokens.shape != (seq_len,):
        raise ValueError(f"Token count mismatch: {tokens.shape} != ({seq_len},)")
    if probe_json["outputs"]["dtype"] != "float32":
        raise ValueError(f"Unsupported probe dtype: {probe_json['outputs']['dtype']}")
    if not bool(probe_json["outputs"]["row_major"]):
        raise ValueError("Probe tensors must be row-major.")

    layer_arrays = []
    input_artifacts = {
        "probe_result_json": {
            "path": str(probe_json_path),
            "sha256": sha256_file(probe_json_path),
            "bytes": probe_json_path.stat().st_size,
        }
    }
    for layer_id in layer_ids:
        layer_path = layer_file_from_probe(probe_dir, probe_json, layer_id)
        layer_array = read_f32_matrix(
            layer_path,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )
        layer_arrays.append(layer_array)
        input_artifacts[f"layer_{layer_id:03d}_input_f32"] = {
            "path": str(layer_path),
            "sha256": sha256_file(layer_path),
            "bytes": layer_path.stat().st_size,
            "stats": summarize_array(layer_array),
        }

    last_hidden_path = last_hidden_file_from_probe(probe_dir, probe_json)
    last_hidden = read_f32_matrix(
        last_hidden_path,
        seq_len=seq_len,
        hidden_size=hidden_size,
    )
    input_artifacts["target_last_hidden_f32"] = {
        "path": str(last_hidden_path),
        "sha256": sha256_file(last_hidden_path),
        "bytes": last_hidden_path.stat().st_size,
        "stats": summarize_array(last_hidden),
    }

    target_hidden = np.concatenate(layer_arrays, axis=1)
    attention_mask = np.ones((seq_len,), dtype=np.uint8)
    loss_mask = np.ones((seq_len,), dtype=np.uint8)
    if args.loss_mask == "not-bos":
        loss_mask[0] = 0

    prepare_target_cache_output_dir(str(output_dir))
    rank_dir = output_dir / "_tmp" / "rank_0"
    rank_dir.mkdir(parents=True, exist_ok=True)
    writer = LocalTargetCacheWriter(
        rank_dir=str(rank_dir),
        max_shard_bytes=int(args.max_shard_bytes),
    )
    try:
        writer.write_sample(
            sample_id=0,
            input_ids=torch.from_numpy(tokens),
            attention_mask=torch.from_numpy(attention_mask),
            loss_mask=torch.from_numpy(loss_mask),
            target_hidden_states=torch.from_numpy(target_hidden),
            target_last_hidden_states=torch.from_numpy(last_hidden),
        )
    finally:
        writer.close()

    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=1,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    ).to_json()
    atomic_json_dump(summary, str(rank_dir / "summary.json"))
    shard_map, shards = build_global_target_cache_shard_map([summary])
    rename_local_target_cache_shards(
        output_dir=str(output_dir),
        rank_dir=str(rank_dir),
        summary=summary,
        shard_map=shard_map,
    )
    num_samples = finalize_target_cache_index(
        output_dir=str(output_dir),
        summaries=[summary],
        shard_map=shard_map,
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "condition_name": "llama_probe_to_deepspec_cache_sample",
            "condition_label": "controlled_probe_cache_sample",
            "source_probe_dir": str(probe_dir),
            "source_probe_command": str(probe_json["condition"].get("command", "")),
            "loss_mask_policy": str(args.loss_mask),
            "target_model_name_or_path": str(
                probe_json["condition"].get("model_path", "")
            ),
            "chat_template": None,
            "max_length": int(seq_len),
            "min_loss_tokens": int(loss_mask.sum()),
            "git_sha": run_git(["rev-parse", "HEAD"])["stdout"],
        },
    )
    write_target_cache_manifest(output_dir=str(output_dir), manifest=manifest)
    cleanup_target_cache_tmp_dir(str(output_dir))

    verification = verify_cache(output_dir, (target_hidden, last_hidden))
    output_artifacts = {}
    for name in ("manifest.json", "samples.idx", "shard-00000.bin"):
        path = output_dir / name
        output_artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    result = {
        "status": "ok",
        "started_at_utc": started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "condition": {
            "name": "llama_probe_to_deepspec_cache_sample",
            "label": "controlled_probe_cache_sample",
            "note": (
                "This validates DeepSpec cache mechanics from one recorded "
                "llama.cpp activation probe. It is not a training run or benchmark."
            ),
            "argv": sys.argv,
            "working_directory": os.getcwd(),
            "source_script": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "git": {
                "head": run_git(["rev-parse", "HEAD"]),
                "status_short": run_git(["status", "--short"]),
            },
        },
        "probe": {
            "dir": str(probe_dir),
            "model": probe_json["model"],
            "context": probe_json["context"],
            "tokenization": probe_json["tokenization"],
            "decode": probe_json["decode"],
        },
        "cache": {
            "dir": str(output_dir),
            "target_layer_ids": layer_ids,
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "num_samples": int(num_samples),
            "loss_mask_policy": str(args.loss_mask),
            "loss_tokens": int(loss_mask.sum()),
            "target_hidden_states": summarize_array(target_hidden),
            "target_last_hidden_states": summarize_array(last_hidden),
        },
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "verification": verification,
    }
    result_path = output_dir / "bridge_result.json"
    atomic_json_dump(result, str(result_path))
    result_sha_path = output_dir / "bridge_result.sha256"
    with open(result_sha_path, "w", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(result_path)}  {result_path.name}\n")
    print(str(result_path))


if __name__ == "__main__":
    main()
