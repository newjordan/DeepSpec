import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel

from scripts.leanstral.plan_leanstral_weight_remap import (
    build_expected_metadata,
    build_mapping,
    read_source_metadata,
    summarize,
)
from scripts.leanstral.probe_leanstral_target import (
    build_leanstral_deepseek_v2_config,
    dump_json,
    load_json,
    sha256_file,
)
from scripts.leanstral.smoke_leanstral_weight_remap import (
    SourceTensorReader,
    dtype_from_name,
    materialize_direct_targets,
    materialize_full_expert_packs,
    tensor_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_PARTIAL_LOAD_BYTES = 16 * 1024**3


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
    layers = set()
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
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def layer_direct_targets(layer_id: int):
    prefix = f"layers.{layer_id}"
    return [
        f"{prefix}.self_attn.q_a_layernorm.weight",
        f"{prefix}.self_attn.kv_a_layernorm.weight",
        f"{prefix}.self_attn.q_a_proj.weight",
        f"{prefix}.self_attn.q_b_proj.weight",
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        f"{prefix}.self_attn.kv_b_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.mlp.gate.weight",
        f"{prefix}.mlp.shared_experts.gate_proj.weight",
        f"{prefix}.mlp.shared_experts.up_proj.weight",
        f"{prefix}.mlp.shared_experts.down_proj.weight",
    ]


def verify_file(path: Path, records, verify_digests: bool):
    loaded = load_file(str(path), device="cpu")
    tensors = {}
    for key, record in records.items():
        item = {"present": key in loaded}
        if key in loaded:
            tensor = loaded[key]
            item["shape"] = [int(dim) for dim in tensor.shape]
            item["dtype"] = str(tensor.dtype).replace("torch.", "")
            if verify_digests:
                digest = tensor_digest(tensor)
                item["sha256_raw_tensor_bytes"] = digest
                item["matches_written_digest"] = (
                    digest == record["stats"]["sha256_raw_tensor_bytes"]
                )
        tensors[key] = item
    digest_values = [
        item.get("matches_written_digest")
        for item in tensors.values()
        if "matches_written_digest" in item
    ]
    return {
        "all_present": all(item["present"] for item in tensors.values()),
        "all_digest_match": all(digest_values) if verify_digests else None,
        "extra_keys": sorted(set(loaded) - set(records)),
        "tensors": tensors,
    }


def partial_hf_load(config_dir: Path, shard_paths):
    try:
        state = {}
        for path in shard_paths:
            state.update(load_file(str(path), device="cpu"))
        config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
        with torch.device("meta"):
            model = AutoModel.from_config(config)
        incompatible = model.load_state_dict(state, strict=False, assign=True)
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "loadable_key_count": len(state),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "missing_key_count": len(incompatible.missing_keys),
        "missing_key_sample_first_40": list(incompatible.missing_keys)[:40],
    }


def copy_tokenizer_files(source_dir: Path, hf_dir: Path):
    copied = []
    for name in ("tekken.json", "params.json", ".gitattributes"):
        source = source_dir / name
        if source.exists():
            target = hf_dir / name
            shutil.copy2(source, target)
            copied.append(str(target))
    return copied


def write_shard(path: Path, tensors, records, verify_digests: bool):
    save_file(tensors, str(path))
    verification = verify_file(path, records, verify_digests)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(tensors),
        "verification": {
            "all_present": verification["all_present"],
            "all_digest_match": verification["all_digest_match"],
            "extra_keys": verification["extra_keys"],
            "tensor_count": len(verification["tensors"]),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Write HF-named Leanstral AutoModel safetensors shards from Mistral FP8 shards."
    )
    parser.add_argument(
        "--source-dir",
        default="/srv/models-hdd/models/leanstral-nvfp4/upstream",
    )
    parser.add_argument(
        "--output-root",
        default="/srv/models-hdd/models/leanstral-nvfp4/hf-remap",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layers/ranges, e.g. 1 or 0-35.",
    )
    parser.add_argument("--include-global", action="store_true")
    parser.add_argument("--copy-tokenizer", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--no-verify-digests",
        action="store_true",
        help="Still verifies key presence and shapes, but skips per-tensor raw-byte digest checks.",
    )
    parser.add_argument(
        "--max-partial-load-bytes",
        type=int,
        default=DEFAULT_MAX_PARTIAL_LOAD_BYTES,
        help=(
            "Maximum total shard bytes to load into one in-memory partial HF load "
            "probe. Set to 0 to skip. Default: 16 GiB."
        ),
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_root).resolve()
    selected_layers = parse_layers(args.layers)
    target_dtype = dtype_from_name(args.dtype)
    verify_digests = not args.no_verify_digests
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    layer_label = "layers-" + "-".join(str(layer) for layer in selected_layers)
    if args.include_global:
        layer_label += "-global"
    output_dir = output_root / f"{timestamp}-{layer_label}"
    hf_dir = output_dir / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=False)

    params_path = source_dir / "params.json"
    index_path = source_dir / "consolidated.safetensors.index.json"
    params = load_json(params_path)
    index = load_json(index_path)
    config = build_leanstral_deepseek_v2_config(params)
    dump_json(hf_dir / "config.json", config)
    copied_tokenizer_files = copy_tokenizer_files(source_dir, hf_dir) if args.copy_tokenizer else []

    source_metadata = read_source_metadata(source_dir, index)
    expected_metadata = build_expected_metadata(params, output_dir / "expected")
    mapping = build_mapping(expected_metadata, source_metadata, params)
    mapping_summary = summarize(mapping, expected_metadata, source_metadata)

    shard_infos = []
    weight_map = {}
    total_size = 0
    reader = SourceTensorReader(source_dir, index)
    try:
        if args.include_global:
            global_tensors, global_records = materialize_direct_targets(
                reader,
                index,
                mapping,
                ["embed_tokens.weight", "norm.weight"],
                target_dtype,
            )
            shard_name = "model-global.safetensors"
            shard_path = hf_dir / shard_name
            shard_info = write_shard(shard_path, global_tensors, global_records, verify_digests)
            shard_infos.append(shard_info)
            total_size += shard_info["bytes"]
            for key in global_tensors:
                weight_map[key] = shard_name
            del global_tensors

        for layer_id in selected_layers:
            direct_tensors, direct_records = materialize_direct_targets(
                reader, index, mapping, layer_direct_targets(layer_id), target_dtype
            )
            expert_tensors, expert_records = materialize_full_expert_packs(
                reader, index, mapping, [layer_id], target_dtype
            )
            tensors = {**direct_tensors, **expert_tensors}
            records = {**direct_records, **expert_records}
            shard_name = f"model-layer-{layer_id:03d}.safetensors"
            shard_path = hf_dir / shard_name
            shard_info = write_shard(shard_path, tensors, records, verify_digests)
            shard_infos.append(shard_info)
            total_size += shard_info["bytes"]
            for key in tensors:
                weight_map[key] = shard_name
            del direct_tensors, direct_records, expert_tensors, expert_records, tensors
    finally:
        reader.close()

    index_payload = {
        "metadata": {
            "total_size": total_size,
            "format": "pt",
            "partial": len(weight_map) != len(expected_metadata),
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    dump_json(hf_dir / "model.safetensors.index.json", index_payload)
    if args.max_partial_load_bytes > 0 and total_size <= args.max_partial_load_bytes:
        partial_load = partial_hf_load(
            hf_dir,
            [Path(info["path"]) for info in shard_infos],
        )
    else:
        partial_load = {
            "status": "skipped",
            "reason": "total shard bytes exceed in-memory partial-load probe cap"
            if args.max_partial_load_bytes > 0
            else "partial-load probe disabled",
            "total_size": total_size,
            "max_partial_load_bytes": args.max_partial_load_bytes,
        }

    all_present = all(info["verification"]["all_present"] for info in shard_infos)
    digest_values = [
        info["verification"]["all_digest_match"]
        for info in shard_infos
        if info["verification"]["all_digest_match"] is not None
    ]
    result = {
        "condition": {
            "name": "leanstral_hf_shard_writer",
            "label": "controlled_partial_hf_model_writer",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "hf_dir": str(hf_dir),
            "command": " ".join(sys.argv),
            "python_command": " ".join([sys.executable, *sys.argv]),
            "cwd": os.getcwd(),
            "started_at_utc": timestamp,
            "layers": selected_layers,
            "include_global": bool(args.include_global),
            "copy_tokenizer": bool(args.copy_tokenizer),
            "target_dtype": str(target_dtype).replace("torch.", ""),
            "verify_digests": verify_digests,
            "max_partial_load_bytes": args.max_partial_load_bytes,
            "offline_env": {
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "git": {
            "sha": run_git(["rev-parse", "HEAD"]),
            "status_short": run_git(["status", "--short", "--branch"]),
            "diff_stat": run_git(["diff", "--stat"]),
        },
        "source_hashes": {
            "params_json_sha256": sha256_file(params_path),
            "index_json_sha256": sha256_file(index_path),
        },
        "mapping_summary": {
            "expected_target_count": mapping_summary["expected_target_count"],
            "planned_target_count": mapping_summary["planned_target_count"],
            "shape_failure_count": len(mapping_summary["shape_failures"]),
            "operation_counts": mapping_summary["operation_counts"],
        },
        "output": {
            "hf_dir": str(hf_dir),
            "index_path": str(hf_dir / "model.safetensors.index.json"),
            "index_sha256": sha256_file(hf_dir / "model.safetensors.index.json"),
            "config_sha256": sha256_file(hf_dir / "config.json"),
            "copied_tokenizer_files": copied_tokenizer_files,
            "shard_count": len(shard_infos),
            "tensor_count": len(weight_map),
            "total_size": total_size,
            "all_present": all_present,
            "all_digest_match": all(digest_values) if verify_digests else None,
            "is_full_automodel": len(weight_map) == len(expected_metadata),
            "missing_expected_keys": sorted(set(expected_metadata) - set(weight_map)),
            "shards": [
                {
                    "path": info["path"],
                    "sha256": info["sha256"],
                    "bytes": info["bytes"],
                    "tensor_count": info["tensor_count"],
                    "verification": info["verification"],
                }
                for info in shard_infos
            ],
        },
        "hf_partial_load_probe": partial_load,
    }
    dump_json(output_dir / "write_result.json", result)
    partial_load_ok = partial_load["status"] in {"ok", "skipped"}
    print(
        json.dumps(
            {
                "status": "ok"
                if all_present
                and (not verify_digests or all(digest_values))
                and partial_load_ok
                else "needs_review",
                "output_dir": str(output_dir),
                "hf_dir": str(hf_dir),
                "result_path": str(output_dir / "write_result.json"),
                "shard_count": len(shard_infos),
                "tensor_count": len(weight_map),
                "total_size": total_size,
                "partial_load_status": partial_load["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
