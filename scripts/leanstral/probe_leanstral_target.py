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
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Model


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
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_layer_ids(raw: str):
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("target layer list is empty")
    return values


def build_leanstral_deepseek_v2_config(params):
    moe = params["moe"]
    yarn = params.get("yarn", {})
    rope_parameters = {
        "rope_type": "yarn",
        "factor": float(yarn.get("factor", 1.0)),
        "original_max_position_embeddings": int(
            yarn.get(
                "original_max_position_embeddings",
                params.get("max_position_embeddings", 8192),
            )
        ),
    }
    if "beta" in yarn:
        rope_parameters["beta_fast"] = float(yarn["beta"])

    return {
        "architectures": ["DeepseekV2Model"],
        "model_type": "deepseek_v2",
        "vocab_size": int(params["vocab_size"]),
        "hidden_size": int(params["dim"]),
        "intermediate_size": int(params["hidden_dim"]),
        "moe_intermediate_size": int(moe["expert_hidden_dim"]),
        "num_hidden_layers": int(params["n_layers"]),
        "num_attention_heads": int(params["n_heads"]),
        "num_key_value_heads": int(params.get("n_kv_heads", params["n_heads"])),
        "head_dim": int(params["head_dim"]),
        "hidden_act": "silu",
        "max_position_embeddings": int(params["max_position_embeddings"]),
        "initializer_range": 0.02,
        "rms_norm_eps": float(params.get("norm_eps", 1e-6)),
        "use_cache": True,
        "pad_token_id": 11,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": bool(params.get("tied_embeddings", False)),
        "rope_parameters": rope_parameters,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "first_k_dense_replace": int(moe.get("first_k_dense_replace", 0)),
        "kv_lora_rank": int(params["kv_lora_rank"]),
        "q_lora_rank": int(params["q_lora_rank"]),
        "n_group": int(moe.get("num_expert_groups", 1)),
        "n_routed_experts": int(moe["num_experts"]),
        "n_shared_experts": int(moe.get("num_shared_experts", 0)),
        "qk_nope_head_dim": int(params["qk_nope_head_dim"]),
        "qk_rope_head_dim": int(params["qk_rope_head_dim"]),
        "routed_scaling_factor": float(moe.get("routed_scale", 1.0)),
        "topk_group": int(moe.get("num_expert_groups_per_tok", 1)),
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "v_head_dim": int(params["v_head_dim"]),
        "num_experts_per_tok": int(moe["num_experts_per_tok"]),
    }


def summarize_weight_keys(index):
    keys = sorted(index["weight_map"].keys())
    qscale_keys = [key for key in keys if key.endswith(".qscale_act") or key.endswith(".qscale_weight")]
    tensor_keys = [key for key in keys if key not in qscale_keys]
    families = {}
    for key in keys:
        if key.startswith("layers."):
            parts = key.split(".")
            family = ".".join(parts[:3])
        else:
            family = key.split(".")[0]
        families[family] = families.get(family, 0) + 1
    return {
        "total_keys": len(keys),
        "tensor_keys": len(tensor_keys),
        "qscale_keys": len(qscale_keys),
        "sample_first_80": keys[:80],
        "family_counts_sample": dict(sorted(families.items())[:80]),
    }


def build_expected_key_probe(num_hidden_layers: int):
    config = DeepseekV2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        head_dim=16,
        first_k_dense_replace=0,
        kv_lora_rank=8,
        q_lora_rank=16,
        n_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        routed_scaling_factor=1.0,
        topk_group=1,
        topk_method="greedy",
        norm_topk_prob=True,
        v_head_dim=16,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = DeepseekV2Model(config).eval()
    keys = sorted(model.state_dict().keys())
    return config, model, keys


def load_cache_hook_runner():
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.data.prepare_target_cache import run_target_forward_with_hooks

    return run_target_forward_with_hooks


def run_synthetic_hook_probe(target_layer_ids):
    hook_runner = load_cache_hook_runner()
    max_layer_id = max([layer_id for layer_id in target_layer_ids if layer_id >= 0], default=2)
    config, model, expected_keys = build_expected_key_probe(
        num_hidden_layers=max(3, max_layer_id + 1)
    )
    selected_layers = []
    for layer_id in target_layer_ids:
        if layer_id == -1:
            selected_layers.append(layer_id)
        elif 0 <= layer_id < config.num_hidden_layers:
            selected_layers.append(layer_id)
    if not selected_layers:
        selected_layers = [-1, 0, config.num_hidden_layers - 1]

    input_ids = torch.tensor([[1, 5, 6, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        result = hook_runner(
            target_model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_ids=selected_layers,
        )
    return {
        "status": "ok",
        "condition_label": "synthetic_deepseek_v2_api_shape_probe",
        "target_layer_ids": selected_layers,
        "input_shape": list(input_ids.shape),
        "target_hidden_states_shape": list(result.target_hidden_states.shape),
        "target_last_hidden_states_shape": list(result.target_last_hidden_states.shape),
        "hidden_size": int(config.hidden_size),
        "expected_key_sample_first_80": expected_keys[:80],
    }


def instantiate_real_meta_model(config_dir: Path):
    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    backbone = getattr(model, "model", model)
    layer_modules = getattr(backbone, "layers", None)
    return {
        "status": "ok",
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "model_type": str(config.model_type),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "has_embed_tokens": hasattr(backbone, "embed_tokens"),
        "has_layers": layer_modules is not None,
        "num_layers_observed": len(layer_modules) if layer_modules is not None else None,
        "has_forward": callable(getattr(model, "forward", None)),
        "device_type": "meta",
        "weight_load_attempted": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe Leanstral params against the DeepSpec target-cache contract."
    )
    parser.add_argument(
        "--source-dir",
        default="/srv/models-hdd/models/leanstral-nvfp4/upstream",
        help="Directory containing params.json, tekken.json, and safetensors index.",
    )
    parser.add_argument(
        "--output-root",
        default="/srv/models-hdd/models/leanstral-nvfp4/probes",
        help="Probe output root.",
    )
    parser.add_argument(
        "--target-layer-ids",
        default="1,9,17,25,33",
        help="Comma-separated target layers to validate for the cache contract.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_root = Path(args.output_root).resolve()
    target_layer_ids = parse_layer_ids(args.target_layer_ids)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{timestamp}-leanstral-deepseek2-target-probe"
    output_dir.mkdir(parents=True, exist_ok=False)

    params_path = source_dir / "params.json"
    index_path = source_dir / "consolidated.safetensors.index.json"
    tekken_path = source_dir / "tekken.json"
    params = load_json(params_path)
    index = load_json(index_path)
    leanstral_config = build_leanstral_deepseek_v2_config(params)
    config_dir = output_dir / "config"
    dump_json(config_dir / "config.json", leanstral_config)

    result = {
        "condition": {
            "name": "leanstral_deepseek2_target_loader_probe",
            "label": "controlled_metadata_and_api_probe",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "target_layer_ids": target_layer_ids,
            "command": " ".join(sys.argv),
            "python_command": " ".join([sys.executable, *sys.argv]),
            "cwd": os.getcwd(),
            "started_at_utc": timestamp,
            "seed": "none",
            "offline_env": {
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "weight_load_attempted": False,
            "notes": [
                "This probe does not load Leanstral weights.",
                "Synthetic forward verifies the DeepSpec hook contract for DeepseekV2 API shape only.",
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
            "tekken_json_sha256": sha256_file(tekken_path) if tekken_path.exists() else None,
        },
        "params_summary": {
            "dim": params.get("dim"),
            "n_layers": params.get("n_layers"),
            "n_heads": params.get("n_heads"),
            "n_kv_heads": params.get("n_kv_heads"),
            "vocab_size": params.get("vocab_size"),
            "max_position_embeddings": params.get("max_position_embeddings"),
            "q_lora_rank": params.get("q_lora_rank"),
            "kv_lora_rank": params.get("kv_lora_rank"),
            "qk_nope_head_dim": params.get("qk_nope_head_dim"),
            "qk_rope_head_dim": params.get("qk_rope_head_dim"),
            "v_head_dim": params.get("v_head_dim"),
            "moe": params.get("moe"),
            "yarn": params.get("yarn"),
            "quantization": params.get("quantization"),
        },
        "generated_config_path": str(config_dir / "config.json"),
        "generated_config": leanstral_config,
        "weight_index_summary": summarize_weight_keys(index),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(source_dir), local_files_only=True)
        result["tokenizer_probe"] = {
            "status": "ok",
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "len": int(len(tokenizer)),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    except Exception as exc:
        result["tokenizer_probe"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        result["real_meta_model_probe"] = instantiate_real_meta_model(config_dir)
    except Exception as exc:
        result["real_meta_model_probe"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        result["synthetic_hook_probe"] = run_synthetic_hook_probe(target_layer_ids)
    except Exception as exc:
        result["synthetic_hook_probe"] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    result["known_next_loader_gap"] = {
        "status": "open",
        "description": (
            "Leanstral safetensor keys use Mistral names plus FP8 qscale tensors; "
            "Transformers DeepseekV2 expects different module names. A weight "
            "remap and FP8/NVFP4 dequant policy are required before real target "
            "hidden-state extraction can run from the BF16/FP8 shard directory."
        ),
        "source_key_examples": result["weight_index_summary"]["sample_first_80"][:20],
        "expected_key_examples": (
            result.get("synthetic_hook_probe", {})
            .get("expected_key_sample_first_80", [])[:20]
        ),
    }

    dump_json(output_dir / "probe_result.json", result)
    print(json.dumps({
        "status": {
            "tokenizer": result["tokenizer_probe"]["status"],
            "real_meta_model": result["real_meta_model_probe"]["status"],
            "synthetic_hook": result["synthetic_hook_probe"]["status"],
        },
        "output_dir": str(output_dir),
        "result_path": str(output_dir / "probe_result.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
