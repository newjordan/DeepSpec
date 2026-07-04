import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel

from scripts.leanstral.plan_leanstral_weight_remap import (
    build_expected_metadata,
    build_mapping,
    read_source_metadata,
    summarize,
)
from scripts.leanstral.probe_leanstral_target import dump_json, load_json


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_TARGETS = [
    "layers.0.self_attn.q_a_layernorm.weight",
    "layers.0.self_attn.kv_a_layernorm.weight",
    "layers.0.self_attn.q_a_proj.weight",
    "layers.0.self_attn.q_b_proj.weight",
    "layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "layers.0.self_attn.kv_b_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.mlp.gate.weight",
    "layers.0.mlp.shared_experts.gate_proj.weight",
    "layers.0.mlp.shared_experts.up_proj.weight",
    "layers.0.mlp.shared_experts.down_proj.weight",
]


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


def parse_csv_ints(raw: str):
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def dtype_from_name(name: str):
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    view = cpu.view(torch.uint8)
    return hashlib.sha256(view.numpy().tobytes()).hexdigest()


def tensor_stats(tensor: torch.Tensor):
    sample = tensor.detach().float()
    finite = torch.isfinite(sample)
    if bool(finite.all()):
        min_value = float(sample.min())
        max_value = float(sample.max())
        mean_value = float(sample.mean())
    else:
        valid = sample[finite]
        min_value = float(valid.min()) if valid.numel() else math.nan
        max_value = float(valid.max()) if valid.numel() else math.nan
        mean_value = float(valid.mean()) if valid.numel() else math.nan
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "finite": bool(finite.all()),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "sha256_raw_tensor_bytes": tensor_digest(tensor),
    }


class SourceTensorReader:
    def __init__(self, source_dir: Path, index):
        self.source_dir = source_dir
        self.index = index
        self._handles = {}

    def close(self):
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def _handle_for(self, key: str):
        shard = self.index["weight_map"][key]
        if shard not in self._handles:
            path = self.source_dir / shard
            handle = safe_open(str(path), framework="pt", device="cpu")
            handle.__enter__()
            self._handles[shard] = handle
        return self._handles[shard]

    def tensor(self, key: str):
        return self._handle_for(key).get_tensor(key)


def source_scale_key(source_key: str):
    if not source_key.endswith(".weight"):
        return None
    return source_key.removesuffix(".weight") + ".qscale_weight"


def load_direct_tensor(reader: SourceTensorReader, index, source_key: str, target_dtype):
    tensor = reader.tensor(source_key)
    scale_key = source_scale_key(source_key)
    scale_value = None
    dequantized = False
    if scale_key is not None and scale_key in index["weight_map"]:
        scale = reader.tensor(scale_key)
        scale_value = float(scale.float().reshape(-1)[0]) if scale.numel() == 1 else None
        tensor = tensor.float() * scale.float()
        dequantized = True
    return tensor.to(dtype=target_dtype).cpu(), {
        "source": source_key,
        "scale_source": scale_key if dequantized else None,
        "scale_value": scale_value,
        "source_dtype": str(reader.tensor(source_key).dtype).replace("torch.", ""),
        "dequantized": dequantized,
    }


def find_mapping(mapping, target_key: str):
    matches = [entry for entry in mapping if entry["target"] == target_key]
    if len(matches) != 1:
        raise KeyError(f"Expected one mapping for {target_key}, found {len(matches)}")
    return matches[0]


def materialize_direct_targets(reader, index, mapping, targets, target_dtype):
    tensors = {}
    records = {}
    for target_key in targets:
        entry = find_mapping(mapping, target_key)
        if entry["operation"] != "direct_dequant_or_copy":
            raise ValueError(
                f"{target_key} is {entry['operation']}; direct smoke targets only"
            )
        source_key = next(key for key in entry["sources"] if key.endswith(".weight"))
        tensor, load_record = load_direct_tensor(reader, index, source_key, target_dtype)
        tensors[target_key] = tensor
        records[target_key] = {
            "operation": entry["operation"],
            "sources": entry["sources"],
            "load_record": load_record,
            "stats": tensor_stats(tensor),
            "target_shape_expected": entry["target_shape"],
            "shape_ok": [int(dim) for dim in tensor.shape] == entry["target_shape"],
        }
    return tensors, records


def materialize_expert_samples(reader, index, layer_ids, expert_ids, target_dtype):
    tensors = {}
    records = {}
    for layer_id in layer_ids:
        for expert_id in expert_ids:
            w1_key = f"layers.{layer_id}.experts.{expert_id}.w1.weight"
            w2_key = f"layers.{layer_id}.experts.{expert_id}.w2.weight"
            w3_key = f"layers.{layer_id}.experts.{expert_id}.w3.weight"
            w1, w1_record = load_direct_tensor(reader, index, w1_key, target_dtype)
            w2, w2_record = load_direct_tensor(reader, index, w2_key, target_dtype)
            w3, w3_record = load_direct_tensor(reader, index, w3_key, target_dtype)
            gate_up = torch.cat([w1, w3], dim=0).contiguous()
            down = w2.contiguous()
            gate_up_key = f"expert_samples.layers.{layer_id}.experts.{expert_id}.gate_up_proj"
            down_key = f"expert_samples.layers.{layer_id}.experts.{expert_id}.down_proj"
            tensors[gate_up_key] = gate_up
            tensors[down_key] = down
            records[gate_up_key] = {
                "operation": "concat_single_expert_w1_w3_dim0",
                "sources": [w1_key, source_scale_key(w1_key), w3_key, source_scale_key(w3_key)],
                "load_records": [w1_record, w3_record],
                "stats": tensor_stats(gate_up),
                "shape_ok": gate_up.shape == (w1.shape[0] + w3.shape[0], w1.shape[1]),
            }
            records[down_key] = {
                "operation": "single_expert_w2",
                "sources": [w2_key, source_scale_key(w2_key)],
                "load_records": [w2_record],
                "stats": tensor_stats(down),
                "shape_ok": down.shape == w2.shape,
            }
    return tensors, records


def materialize_full_expert_packs(reader, index, mapping, layer_ids, target_dtype):
    tensors = {}
    records = {}
    for layer_id in layer_ids:
        gate_up_key = f"layers.{layer_id}.mlp.experts.gate_up_proj"
        down_key = f"layers.{layer_id}.mlp.experts.down_proj"
        gate_up_entry = find_mapping(mapping, gate_up_key)
        down_entry = find_mapping(mapping, down_key)
        gate_up_shape = tuple(int(dim) for dim in gate_up_entry["target_shape"])
        down_shape = tuple(int(dim) for dim in down_entry["target_shape"])
        num_experts = gate_up_shape[0]
        gate_up = torch.empty(gate_up_shape, dtype=target_dtype)
        down = torch.empty(down_shape, dtype=target_dtype)
        gate_up_load_records = []
        down_load_records = []
        for expert_id in range(num_experts):
            w1_key = f"layers.{layer_id}.experts.{expert_id}.w1.weight"
            w2_key = f"layers.{layer_id}.experts.{expert_id}.w2.weight"
            w3_key = f"layers.{layer_id}.experts.{expert_id}.w3.weight"
            w1, w1_record = load_direct_tensor(reader, index, w1_key, target_dtype)
            w2, w2_record = load_direct_tensor(reader, index, w2_key, target_dtype)
            w3, w3_record = load_direct_tensor(reader, index, w3_key, target_dtype)
            gate_up[expert_id].copy_(torch.cat([w1, w3], dim=0))
            down[expert_id].copy_(w2)
            if expert_id in (0, num_experts - 1):
                gate_up_load_records.extend([w1_record, w3_record])
                down_load_records.append(w2_record)
        tensors[gate_up_key] = gate_up.contiguous()
        tensors[down_key] = down.contiguous()
        records[gate_up_key] = {
            "operation": "stack_experts_concat_w1_w3_dim0",
            "sources_count": len(gate_up_entry["sources"]),
            "source_sample_records": gate_up_load_records,
            "stats": tensor_stats(tensors[gate_up_key]),
            "target_shape_expected": list(gate_up_shape),
            "shape_ok": [int(dim) for dim in gate_up.shape] == list(gate_up_shape),
        }
        records[down_key] = {
            "operation": "stack_experts_w2",
            "sources_count": len(down_entry["sources"]),
            "source_sample_records": down_load_records,
            "stats": tensor_stats(tensors[down_key]),
            "target_shape_expected": list(down_shape),
            "shape_ok": [int(dim) for dim in down.shape] == list(down_shape),
        }
    return tensors, records


def verify_safetensors(path: Path, expected_records):
    loaded = load_file(str(path), device="cpu")
    result = {}
    for key, record in expected_records.items():
        if key not in loaded:
            result[key] = {"present": False}
            continue
        tensor = loaded[key]
        result[key] = {
            "present": True,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "sha256_raw_tensor_bytes": tensor_digest(tensor),
            "matches_written_digest": (
                tensor_digest(tensor)
                == record["stats"]["sha256_raw_tensor_bytes"]
            ),
        }
    extra_keys = sorted(set(loaded) - set(expected_records))
    return {
        "all_present": all(item.get("present") for item in result.values()),
        "all_digest_match": all(
            item.get("matches_written_digest") for item in result.values()
        ),
        "extra_keys": extra_keys,
        "tensors": result,
    }


def partial_hf_load_probe(config_dir: Path, tensors):
    loadable_state = {
        key: value
        for key, value in tensors.items()
        if not key.startswith("expert_samples.")
    }
    try:
        config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
        with torch.device("meta"):
            model = AutoModel.from_config(config)
        incompatible = model.load_state_dict(loadable_state, strict=False, assign=True)
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "loadable_key_count": len(loadable_state),
        }
    return {
        "status": "ok",
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "loadable_key_count": len(loadable_state),
        "loaded_keys": sorted(loadable_state),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "missing_key_sample_first_40": list(incompatible.missing_keys)[:40],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Materialize a controlled real-tensor Leanstral remap smoke subset."
    )
    parser.add_argument(
        "--source-dir",
        default="/srv/models-hdd/models/leanstral-nvfp4/upstream",
    )
    parser.add_argument(
        "--output-root",
        default="/srv/models-hdd/models/leanstral-nvfp4/probes",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Exact HF target tensor to materialize. Repeatable. Defaults to layer-0 direct tensors.",
    )
    parser.add_argument("--expert-layers", default="0")
    parser.add_argument("--expert-ids", default="0,127")
    parser.add_argument(
        "--full-expert-layers",
        default="",
        help="Comma-separated layers for exact full HF routed-expert pack tensors.",
    )
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_root).resolve()
    target_dtype = dtype_from_name(args.dtype)
    direct_targets = list(args.target) if args.target else list(DEFAULT_TARGETS)
    expert_layers = parse_csv_ints(args.expert_layers)
    expert_ids = parse_csv_ints(args.expert_ids)
    full_expert_layers = parse_csv_ints(args.full_expert_layers)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{timestamp}-leanstral-remap-smoke"
    output_dir.mkdir(parents=True, exist_ok=False)

    params_path = source_dir / "params.json"
    index_path = source_dir / "consolidated.safetensors.index.json"
    params = load_json(params_path)
    index = load_json(index_path)
    source_metadata = read_source_metadata(source_dir, index)
    expected_metadata = build_expected_metadata(params, output_dir)
    mapping = build_mapping(expected_metadata, source_metadata, params)
    mapping_summary = summarize(mapping, expected_metadata, source_metadata)

    reader = SourceTensorReader(source_dir, index)
    try:
        direct_tensors, direct_records = materialize_direct_targets(
            reader, index, mapping, direct_targets, target_dtype
        )
        expert_tensors, expert_records = materialize_expert_samples(
            reader, index, expert_layers, expert_ids, target_dtype
        )
        full_expert_tensors, full_expert_records = materialize_full_expert_packs(
            reader, index, mapping, full_expert_layers, target_dtype
        )
    finally:
        reader.close()

    tensors = {**direct_tensors, **expert_tensors, **full_expert_tensors}
    records = {**direct_records, **expert_records, **full_expert_records}
    output_path = output_dir / "remapped_subset.safetensors"
    save_file(tensors, str(output_path))
    verification = verify_safetensors(output_path, records)
    hf_partial_load = partial_hf_load_probe(output_dir / "config", tensors)

    result = {
        "condition": {
            "name": "leanstral_weight_remap_real_tensor_smoke",
            "label": "controlled_real_tensor_subset",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "command": " ".join(sys.argv),
            "python_command": " ".join([sys.executable, *sys.argv]),
            "cwd": os.getcwd(),
            "started_at_utc": timestamp,
            "target_dtype": str(target_dtype).replace("torch.", ""),
            "direct_targets": direct_targets,
            "expert_layers": expert_layers,
            "expert_ids": expert_ids,
            "full_expert_layers": full_expert_layers,
            "offline_env": {
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "notes": [
                "This smoke reads real Leanstral tensor payloads for a small subset.",
                "FP8 dequant policy follows llama.cpp conversion/base.py: weight.float() * qscale_weight.float().",
                "Expert sample tensors prove concat/pack mechanics for selected experts only; they are not full HF expert-pack tensors.",
                "Full expert layers, when provided, materialize exact HF routed-expert pack tensors for those layers.",
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
        "mapping_summary": {
            "expected_target_count": mapping_summary["expected_target_count"],
            "planned_target_count": mapping_summary["planned_target_count"],
            "shape_failure_count": len(mapping_summary["shape_failures"]),
            "operation_counts": mapping_summary["operation_counts"],
        },
        "output": {
            "safetensors_path": str(output_path),
            "safetensors_sha256": sha256_file(output_path),
            "tensor_count": len(tensors),
            "bytes": output_path.stat().st_size,
        },
        "records": records,
        "verification": verification,
        "hf_partial_load_probe": hf_partial_load,
    }
    dump_json(output_dir / "remap_smoke_result.json", result)
    print(
        json.dumps(
            {
                "status": "ok"
                if verification["all_present"]
                and verification["all_digest_match"]
                and hf_partial_load["status"] == "ok"
                else "needs_review",
                "output_dir": str(output_dir),
                "result_path": str(output_dir / "remap_smoke_result.json"),
                "safetensors_path": str(output_path),
                "tensor_count": len(tensors),
                "safetensors_sha256": result["output"]["safetensors_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
