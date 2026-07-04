import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

from scripts.leanstral.probe_leanstral_target import (
    build_leanstral_deepseek_v2_config,
    dump_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shape_of(value):
    return [int(dim) for dim in tuple(value.shape)]


def dtype_name(value):
    return str(value.dtype).replace("torch.", "")


def read_source_metadata(source_dir: Path, index):
    by_shard = {}
    for key, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(key)

    metadata = {}
    for shard, keys in sorted(by_shard.items()):
        shard_path = source_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in sorted(keys):
                tensor_slice = handle.get_slice(key)
                metadata[key] = {
                    "shape": [int(dim) for dim in tensor_slice.get_shape()],
                    "dtype": str(tensor_slice.get_dtype()),
                    "shard": shard,
                }
    return metadata


def build_expected_metadata(params, output_dir: Path):
    config_dir = output_dir / "config"
    dump_json(config_dir / "config.json", build_leanstral_deepseek_v2_config(params))
    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    torch.set_default_device("meta")
    try:
        model = AutoModel.from_config(config)
    finally:
        torch.set_default_device("cpu")
    return {
        key: {"shape": shape_of(value), "dtype": dtype_name(value)}
        for key, value in model.state_dict().items()
    }


def source_with_sidecars(source_key: str, source_metadata):
    keys = [source_key]
    prefix = source_key.removesuffix(".weight")
    for suffix in ("qscale_act", "qscale_weight"):
        sidecar = f"{prefix}.{suffix}"
        if sidecar in source_metadata:
            keys.append(sidecar)
    return keys


def direct_entry(target_key: str, source_key: str, expected_metadata, source_metadata):
    target = expected_metadata[target_key]
    source = source_metadata.get(source_key)
    ok = source is not None and source["shape"] == target["shape"]
    return {
        "target": target_key,
        "operation": "direct_dequant_or_copy",
        "sources": source_with_sidecars(source_key, source_metadata),
        "target_shape": target["shape"],
        "source_shape": source["shape"] if source else None,
        "source_dtype": source["dtype"] if source else None,
        "shape_ok": ok,
    }


def expert_gate_up_entry(layer_id: int, expected_metadata, source_metadata, num_experts: int):
    target_key = f"layers.{layer_id}.mlp.experts.gate_up_proj"
    target = expected_metadata[target_key]
    sources = []
    missing = []
    for expert_id in range(num_experts):
        for which in ("w1", "w3"):
            source_key = f"layers.{layer_id}.experts.{expert_id}.{which}.weight"
            if source_key not in source_metadata:
                missing.append(source_key)
            sources.extend(source_with_sidecars(source_key, source_metadata))
    w1_shape = source_metadata.get(f"layers.{layer_id}.experts.0.w1.weight", {}).get("shape")
    w3_shape = source_metadata.get(f"layers.{layer_id}.experts.0.w3.weight", {}).get("shape")
    computed = None
    if w1_shape and w3_shape:
        computed = [num_experts, int(w1_shape[0]) + int(w3_shape[0]), int(w1_shape[1])]
    return {
        "target": target_key,
        "operation": "stack_experts_concat_w1_w3_dim0",
        "sources": sources,
        "target_shape": target["shape"],
        "computed_source_shape": computed,
        "shape_ok": computed == target["shape"] and not missing,
        "missing_sources": missing,
    }


def expert_down_entry(layer_id: int, expected_metadata, source_metadata, num_experts: int):
    target_key = f"layers.{layer_id}.mlp.experts.down_proj"
    target = expected_metadata[target_key]
    sources = []
    missing = []
    for expert_id in range(num_experts):
        source_key = f"layers.{layer_id}.experts.{expert_id}.w2.weight"
        if source_key not in source_metadata:
            missing.append(source_key)
        sources.extend(source_with_sidecars(source_key, source_metadata))
    w2_shape = source_metadata.get(f"layers.{layer_id}.experts.0.w2.weight", {}).get("shape")
    computed = [num_experts, *w2_shape] if w2_shape else None
    return {
        "target": target_key,
        "operation": "stack_experts_w2",
        "sources": sources,
        "target_shape": target["shape"],
        "computed_source_shape": computed,
        "shape_ok": computed == target["shape"] and not missing,
        "missing_sources": missing,
    }


def build_mapping(expected_metadata, source_metadata, params):
    num_layers = int(params["n_layers"])
    num_experts = int(params["moe"]["num_experts"])
    mapping = []

    mapping.append(direct_entry("embed_tokens.weight", "tok_embeddings.weight", expected_metadata, source_metadata))
    mapping.append(direct_entry("norm.weight", "norm.weight", expected_metadata, source_metadata))

    for layer_id in range(num_layers):
        prefix = f"layers.{layer_id}"
        direct_pairs = [
            ("self_attn.q_a_proj.weight", "attention.wq_a.weight"),
            ("self_attn.q_a_layernorm.weight", "attention.q_a_norm.weight"),
            ("self_attn.q_b_proj.weight", "attention.wq_b.weight"),
            ("self_attn.kv_a_proj_with_mqa.weight", "attention.wkv_a_with_mqa.weight"),
            ("self_attn.kv_a_layernorm.weight", "attention.kv_a_norm.weight"),
            ("self_attn.kv_b_proj.weight", "attention.wkv_b.weight"),
            ("self_attn.o_proj.weight", "attention.wo.weight"),
            ("mlp.gate.weight", "gate.weight"),
            ("mlp.shared_experts.gate_proj.weight", "shared_experts.w1.weight"),
            ("mlp.shared_experts.up_proj.weight", "shared_experts.w3.weight"),
            ("mlp.shared_experts.down_proj.weight", "shared_experts.w2.weight"),
            ("input_layernorm.weight", "attention_norm.weight"),
            ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ]
        for target_suffix, source_suffix in direct_pairs:
            mapping.append(
                direct_entry(
                    f"{prefix}.{target_suffix}",
                    f"{prefix}.{source_suffix}",
                    expected_metadata,
                    source_metadata,
                )
            )
        mapping.append(
            expert_gate_up_entry(layer_id, expected_metadata, source_metadata, num_experts)
        )
        mapping.append(
            expert_down_entry(layer_id, expected_metadata, source_metadata, num_experts)
        )
    return mapping


def summarize(mapping, expected_metadata, source_metadata):
    planned_targets = {entry["target"] for entry in mapping}
    expected_targets = set(expected_metadata)
    used_sources = set()
    for entry in mapping:
        used_sources.update(entry["sources"])
    language_sources = {
        key
        for key in source_metadata
        if not key.startswith("vision_")
        and not key.startswith("patch_merger.")
        and not key.startswith("pre_mm_projector")
        and not key.startswith("vision_language_adapter.")
    }
    qscale_sources = {
        key
        for key in source_metadata
        if key.endswith(".qscale_act") or key.endswith(".qscale_weight")
    }
    payload_language_sources = language_sources - qscale_sources
    return {
        "expected_target_count": len(expected_targets),
        "planned_target_count": len(planned_targets),
        "all_expected_targets_planned": planned_targets == expected_targets,
        "missing_expected_targets": sorted(expected_targets - planned_targets),
        "extra_planned_targets": sorted(planned_targets - expected_targets),
        "shape_failures": [
            entry for entry in mapping if not bool(entry.get("shape_ok"))
        ],
        "operation_counts": {
            operation: sum(1 for entry in mapping if entry["operation"] == operation)
            for operation in sorted({entry["operation"] for entry in mapping})
        },
        "source_counts": {
            "total": len(source_metadata),
            "language_payload": len(payload_language_sources),
            "qscale": len(qscale_sources),
            "used_total": len(used_sources),
            "unused_language_payload": len(payload_language_sources - used_sources),
        },
        "unused_language_payload_keys": sorted(payload_language_sources - used_sources),
        "used_source_sample_first_80": sorted(used_sources)[:80],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a metadata-only Leanstral-to-DeepseekV2 weight remap plan."
    )
    parser.add_argument(
        "--source-dir",
        default="/srv/models-hdd/models/leanstral-nvfp4/upstream",
    )
    parser.add_argument(
        "--output-root",
        default="/srv/models-hdd/models/leanstral-nvfp4/probes",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{timestamp}-leanstral-weight-remap-plan"
    output_dir.mkdir(parents=True, exist_ok=False)

    params_path = source_dir / "params.json"
    index_path = source_dir / "consolidated.safetensors.index.json"
    params = load_json(params_path)
    index = load_json(index_path)
    source_metadata = read_source_metadata(source_dir, index)
    expected_metadata = build_expected_metadata(params, output_dir)
    mapping = build_mapping(expected_metadata, source_metadata, params)
    summary = summarize(mapping, expected_metadata, source_metadata)

    result = {
        "condition": {
            "name": "leanstral_weight_remap_plan",
            "label": "controlled_metadata_only_shape_plan",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "command": " ".join(sys.argv),
            "python_command": " ".join([sys.executable, *sys.argv]),
            "cwd": os.getcwd(),
            "started_at_utc": timestamp,
            "tensor_payload_read": False,
            "offline_env": {
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "notes": [
                "This plan uses safetensors metadata and a meta HF model only.",
                "It verifies shape coverage for AutoModel target-cache loading, not numerical correctness.",
            ],
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
        "summary": summary,
        "mapping": mapping,
    }
    dump_json(output_dir / "weight_remap_plan.json", result)
    print(
        json.dumps(
            {
                "status": "ok" if summary["all_expected_targets_planned"] and not summary["shape_failures"] else "needs_review",
                "output_dir": str(output_dir),
                "result_path": str(output_dir / "weight_remap_plan.json"),
                "expected_target_count": summary["expected_target_count"],
                "planned_target_count": summary["planned_target_count"],
                "shape_failures": len(summary["shape_failures"]),
                "unused_language_payload": summary["source_counts"]["unused_language_payload"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
