import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, Qwen3Config

from deepspec.data.target_cache_dataset import CacheCollator, CacheDataset
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from scripts.leanstral.probe_leanstral_target import dump_json, sha256_file
from scripts.leanstral.smoke_leanstral_dspark_forward import (
    init_single_process_dist,
    run_git,
    tensor_summary,
)
from scripts.leanstral.train_leanstral_dspark_tiny import (
    dtype_from_name,
    grad_summary,
    load_shared_target_weights,
    scalar,
    write_jsonl,
)


DEFAULT_TARGET_CONFIG = (
    "/srv/models-hdd/models/leanstral-nvfp4/hf-remap/"
    "20260704T011143Z-layers-1-global/hf_model"
)
DEFAULT_SOURCE_DIR = "/srv/models-hdd/models/leanstral-nvfp4/upstream"
DEFAULT_CACHE_DIR = (
    "/srv/models-hdd/models/leanstral-nvfp4/target-cache-runs/"
    "20260704T015954Z-llama-prompt-cache-dataset"
)
DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/qwen3-dspark-tiny"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def choose_output_dir(output_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-leanstral-qwen3-dspark-tiny").resolve()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_qwen3_draft_config(target_config, args):
    head_dim = int(args.head_dim or target_config.hidden_size // target_config.num_attention_heads)
    config = Qwen3Config(
        vocab_size=int(target_config.vocab_size),
        hidden_size=int(target_config.hidden_size),
        intermediate_size=int(args.draft_intermediate_size),
        num_hidden_layers=int(args.num_draft_layers),
        num_attention_heads=int(target_config.num_attention_heads),
        num_key_value_heads=int(target_config.num_key_value_heads),
        head_dim=head_dim,
        max_position_embeddings=int(target_config.max_position_embeddings),
        rms_norm_eps=float(target_config.rms_norm_eps),
        bos_token_id=getattr(target_config, "bos_token_id", None),
        eos_token_id=getattr(target_config, "eos_token_id", None),
        pad_token_id=getattr(target_config, "pad_token_id", None),
        rope_parameters={
            "rope_type": "yarn",
            "factor": float(args.rope_scaling_factor),
            "original_max_position_embeddings": int(args.original_context_length),
            "rope_theta": float(args.rope_theta),
        },
    )
    config.architectures = ["Qwen3DSparkModel"]
    config.num_target_layers = int(target_config.num_hidden_layers)
    config.num_hidden_layers = int(args.num_draft_layers)
    config.block_size = int(args.block_size)
    config.tie_word_embeddings = False
    config.layer_types = ["full_attention"] * int(args.num_draft_layers)
    config._attn_implementation = "flex_attention"
    config.mask_token_id = int(args.mask_token_id)
    config.target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
    config.num_anchors = int(args.num_anchors)
    config.enable_confidence_head = float(args.confidence_head_alpha) > 0.0
    if config.enable_confidence_head:
        config.confidence_head_with_markov = bool(args.confidence_head_with_markov)
    config.markov_rank = int(args.markov_rank)
    if int(args.markov_rank) > 0:
        config.markov_head_type = str(args.markov_head_type)
    return config


def batch_to_device(batch, device):
    out = {key: value.to(device=device) for key, value in batch.items()}
    out["input_ids"] = out["input_ids"].long()
    return out


@torch.no_grad()
def evaluate_loss(model, batch, args, dtype, seed: int):
    torch.manual_seed(int(seed))
    if batch["input_ids"].is_cuda:
        torch.cuda.manual_seed_all(int(seed))
    model.eval()
    outputs = model(
        input_ids=batch["input_ids"],
        target_hidden_states=batch["target_hidden_states"].to(dtype=dtype),
        loss_mask=batch["loss_mask"],
        target_last_hidden_states=batch["target_last_hidden_states"].to(dtype=dtype),
    )
    loss = compute_dspark_loss(
        outputs=outputs,
        loss_decay_gamma=float(args.loss_decay_gamma),
        ce_loss_alpha=float(args.ce_loss_alpha),
        l1_loss_alpha=float(args.l1_loss_alpha),
        confidence_head_alpha=float(args.confidence_head_alpha),
    )
    return scalar(loss), {
        "draft_logits": tensor_summary(outputs.draft_logits),
        "target_ids": tensor_summary(outputs.target_ids),
        "eval_mask": tensor_summary(outputs.eval_mask),
        "block_keep_mask": tensor_summary(outputs.block_keep_mask),
        "aligned_target_logits": (
            tensor_summary(outputs.aligned_target_logits)
            if outputs.aligned_target_logits is not None
            else None
        ),
    }


def count_parameters(model: torch.nn.Module):
    total = 0
    trainable = 0
    frozen = 0
    by_prefix = {}
    for name, param in model.named_parameters():
        numel = int(param.numel())
        total += numel
        prefix = name.split(".", 1)[0]
        item = by_prefix.setdefault(prefix, {"total": 0, "trainable": 0, "frozen": 0})
        item["total"] += numel
        if param.requires_grad:
            trainable += numel
            item["trainable"] += numel
        else:
            frozen += numel
            item["frozen"] += numel
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "by_prefix": by_prefix,
    }


def state_dict_to_cpu(model: torch.nn.Module):
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }


def summarize_state_dict(state):
    by_prefix = {}
    total_numel = 0
    for name, tensor in state.items():
        total_numel += int(tensor.numel())
        prefix = name.split(".", 1)[0]
        item = by_prefix.setdefault(prefix, {"tensor_count": 0, "numel": 0})
        item["tensor_count"] += 1
        item["numel"] += int(tensor.numel())
    return {
        "tensor_count": len(state),
        "numel": total_numel,
        "by_prefix": by_prefix,
        "keys": sorted(state),
    }


def selected_tensor_summaries(state):
    selected = {}
    for name in (
        "embed_tokens.weight",
        "lm_head.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "fc.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
    ):
        tensor = state.get(name)
        if tensor is None:
            selected[name] = {"present": False}
            continue
        selected[name] = {
            "present": True,
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": int(tensor.numel()),
        }
    return selected


def copy_tokenizer_sidecars(source_dir: Path, export_dir: Path):
    copied = {}
    for name in ("tekken.json", "params.json", ".gitattributes"):
        source = source_dir / name
        if not source.exists():
            continue
        target = export_dir / name
        shutil.copy2(source, target)
        copied[name] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return copied


def write_weight_index(export_dir: Path, state):
    file_name = "model.safetensors"
    path = export_dir / "model.safetensors.index.json"
    payload = {
        "metadata": {"total_size": str((export_dir / file_name).stat().st_size)},
        "weight_map": {name: file_name for name in sorted(state)},
    }
    dump_json(path, payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "weight_count": len(payload["weight_map"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train and export a tiny Leanstral-targeted Qwen3/DSpark draft. "
            "This uses a DFlash-compatible draft body for llama.cpp DSpark GGUF."
        )
    )
    parser.add_argument("--target-config", default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument("--draft-intermediate-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, default=0)
    parser.add_argument("--target-layer-ids", default="1,9,17,25,33")
    parser.add_argument("--markov-rank", type=int, default=32)
    parser.add_argument("--markov-head-type", default="vanilla")
    parser.add_argument("--confidence-head-alpha", type=float, default=0.0)
    parser.add_argument("--confidence-head-with-markov", action="store_true")
    parser.add_argument("--ce-loss-alpha", type=float, default=0.1)
    parser.add_argument("--l1-loss-alpha", type=float, default=0.9)
    parser.add_argument("--loss-decay-gamma", type=float, default=4.0)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-scaling-factor", type=float, default=128.0)
    parser.add_argument("--original-context-length", type=int, default=8192)
    parser.add_argument("--loss-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output_dir = choose_output_dir(Path(args.output_root).resolve(), args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    export_dir = output_dir / "hf_model"
    export_dir.mkdir(parents=False, exist_ok=False)
    script_path = Path(__file__).resolve()
    target_config_path = Path(args.target_config).resolve()
    source_dir = Path(args.source_dir).resolve()
    source_index_path = source_dir / "consolidated.safetensors.index.json"
    cache_dir = Path(args.cache_dir).resolve()
    cache_manifest_path = cache_dir / "manifest.json"
    result = None

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
        dtype = dtype_from_name(args.dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(int(args.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed))

        dist_init_file = init_single_process_dist(output_dir)
        cache_manifest = load_json(cache_manifest_path)
        source_index = load_json(source_index_path)
        target_config = AutoConfig.from_pretrained(
            str(target_config_path),
            local_files_only=True,
        )
        draft_config = build_qwen3_draft_config(target_config, args)
        model = Qwen3DSparkModel(draft_config).to(device=device, dtype=dtype)
        shared_weight_records = load_shared_target_weights(
            model=model,
            source_dir=source_dir,
            index=source_index,
            dtype=dtype,
            device=device,
        )
        param_counts = count_parameters(model)

        dataset = CacheDataset(str(cache_dir))
        try:
            features = [dataset[idx] for idx in range(len(dataset))]
        finally:
            dataset.close()
        batch = batch_to_device(CacheCollator()(features), device)

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

        initial_eval_loss, initial_output_summary = evaluate_loss(
            model,
            batch,
            args,
            dtype,
            seed=int(args.seed) + 10_000,
        )
        rows = []
        train_log_path = output_dir / "train_log.jsonl"
        for step_idx in range(int(args.steps)):
            step_seed = int(args.seed) + step_idx
            torch.manual_seed(step_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(step_seed)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"].to(dtype=dtype),
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch["target_last_hidden_states"].to(
                    dtype=dtype
                ),
            )
            loss = compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=float(args.loss_decay_gamma),
                ce_loss_alpha=float(args.ce_loss_alpha),
                l1_loss_alpha=float(args.l1_loss_alpha),
                confidence_head_alpha=float(args.confidence_head_alpha),
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite loss at step {step_idx}: {loss}")
            loss.backward()
            grads = grad_summary(model)
            if not grads["finite"]:
                raise RuntimeError(f"Non-finite gradients at step {step_idx}.")
            optimizer.step()
            rows.append(
                {
                    "step": step_idx + 1,
                    "seed": step_seed,
                    "loss": scalar(loss),
                    "grad": grads,
                    "draft_logits": tensor_summary(outputs.draft_logits),
                    "eval_mask_true": int(outputs.eval_mask.sum().item()),
                    "cuda_memory_allocated": (
                        int(torch.cuda.memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                    "cuda_max_memory_allocated": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None
                    ),
                }
            )
            write_jsonl(train_log_path, rows)

        final_eval_loss, final_output_summary = evaluate_loss(
            model,
            batch,
            args,
            dtype,
            seed=int(args.seed) + 10_000,
        )

        config_path = export_dir / "config.json"
        draft_config.to_json_file(config_path)
        tokenizer_sidecars = copy_tokenizer_sidecars(source_dir, export_dir)
        full_state = state_dict_to_cpu(model)
        weights_path = export_dir / "model.safetensors"
        save_file(full_state, str(weights_path), metadata={"format": "pt"})
        index_info = write_weight_index(export_dir, full_state)
        readback_state = load_file(str(weights_path), device="cpu")
        if sorted(readback_state) != sorted(full_state):
            raise RuntimeError("Readback state keys do not match exported state.")

        verify_config = Qwen3Config.from_json_file(str(config_path))
        verify_config._attn_implementation = "flex_attention"
        verify_model = Qwen3DSparkModel(verify_config).to(device=device, dtype=dtype)
        verify_model.load_state_dict(readback_state, strict=True)
        readback_eval_loss, readback_output_summary = evaluate_loss(
            verify_model,
            batch,
            args,
            dtype,
            seed=int(args.seed) + 10_000,
        )
        readback_loss_abs_diff = abs(readback_eval_loss - final_eval_loss)
        if readback_loss_abs_diff > float(args.loss_tolerance):
            raise RuntimeError(
                "Readback loss mismatch: "
                f"{readback_eval_loss} vs {final_eval_loss} "
                f"(abs diff {readback_loss_abs_diff})"
            )

        state_summary = summarize_state_dict(full_state)
        manifest = {
            "format": "leanstral_qwen3_dspark_hf_export",
            "condition_label": "controlled_qwen3_body_train_export_readback_smoke",
            "target_cache_dir": str(cache_dir),
            "target_model_sha256": str(cache_manifest["target_model_sha256"]),
            "final_eval_loss": final_eval_loss,
            "readback_eval_loss": readback_eval_loss,
            "readback_loss_abs_diff": readback_loss_abs_diff,
            "weights_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
            "shared_target_weights": shared_weight_records,
            "state_summary": state_summary,
        }
        export_manifest_path = export_dir / "dspark_export_manifest.json"
        dump_json(export_manifest_path, manifest)

        result = {
            "status": "ok",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "condition": {
                "name": "leanstral_qwen3_body_dspark_tiny_train_export",
                "label": "controlled_qwen3_body_train_export_readback_smoke",
                "note": (
                    "Tiny Leanstral-targeted DSpark train/export using the "
                    "Qwen3/DFlash-compatible draft body. This is a mechanics "
                    "artifact intended to feed the existing llama.cpp DSpark "
                    "converter path; it is not a release-quality draft."
                ),
                "argv": list(sys.argv),
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
            "inputs": {
                "target_config": {
                    "path": str(target_config_path),
                    "config_json_sha256": sha256_file(target_config_path / "config.json"),
                },
                "source_dir": str(source_dir),
                "source_index": {
                    "path": str(source_index_path),
                    "sha256": sha256_file(source_index_path),
                },
                "shared_target_weights": shared_weight_records,
                "cache_dir": str(cache_dir),
                "cache_manifest": {
                    "path": str(cache_manifest_path),
                    "sha256": sha256_file(cache_manifest_path),
                    "num_samples": int(cache_manifest["num_samples"]),
                    "target_layer_ids": [
                        int(x) for x in cache_manifest["target_layer_ids"]
                    ],
                    "hidden_size": int(cache_manifest["hidden_size"]),
                    "max_length": int(cache_manifest["max_length"]),
                    "target_model_sha256": str(cache_manifest["target_model_sha256"]),
                },
            },
            "draft_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "architectures": list(draft_config.architectures),
                "model_type": str(draft_config.model_type),
                "hidden_size": int(draft_config.hidden_size),
                "intermediate_size": int(draft_config.intermediate_size),
                "num_hidden_layers": int(draft_config.num_hidden_layers),
                "num_attention_heads": int(draft_config.num_attention_heads),
                "num_key_value_heads": int(draft_config.num_key_value_heads),
                "head_dim": int(draft_config.head_dim),
                "vocab_size": int(draft_config.vocab_size),
                "block_size": int(draft_config.block_size),
                "num_anchors": int(draft_config.num_anchors),
                "target_layer_ids": [int(x) for x in draft_config.target_layer_ids],
                "markov_rank": int(draft_config.markov_rank),
                "markov_head_type": getattr(draft_config, "markov_head_type", None),
                "enable_confidence_head": bool(draft_config.enable_confidence_head),
                "attn_implementation": str(draft_config._attn_implementation),
            },
            "training": {
                "steps": int(args.steps),
                "optimizer": "AdamW",
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "loss_decay_gamma": float(args.loss_decay_gamma),
                "ce_loss_alpha": float(args.ce_loss_alpha),
                "l1_loss_alpha": float(args.l1_loss_alpha),
                "confidence_head_alpha": float(args.confidence_head_alpha),
                "initial_eval_loss": initial_eval_loss,
                "final_eval_loss": final_eval_loss,
                "step_log_path": str(train_log_path),
                "step_log_sha256": sha256_file(train_log_path),
                "step_losses": [float(row["loss"]) for row in rows],
                "param_counts": param_counts,
            },
            "export": {
                "output_dir": str(output_dir),
                "hf_model_dir": str(export_dir),
                "weights": {
                    "path": str(weights_path),
                    "sha256": sha256_file(weights_path),
                    "bytes": weights_path.stat().st_size,
                },
                "index": index_info,
                "manifest": {
                    "path": str(export_manifest_path),
                    "sha256": sha256_file(export_manifest_path),
                    "bytes": export_manifest_path.stat().st_size,
                },
                "tokenizer_sidecars": tokenizer_sidecars,
                "state_summary": state_summary,
                "selected_tensors": selected_tensor_summaries(full_state),
            },
            "verification": {
                "final_eval_loss": final_eval_loss,
                "readback_eval_loss": readback_eval_loss,
                "readback_loss_abs_diff": readback_loss_abs_diff,
                "loss_tolerance": float(args.loss_tolerance),
            },
            "batch": {
                key: tensor_summary(value)
                for key, value in batch.items()
                if key in (
                    "input_ids",
                    "attention_mask",
                    "loss_mask",
                    "target_hidden_states",
                    "target_last_hidden_states",
                )
            },
            "outputs": {
                "initial": initial_output_summary,
                "final": final_output_summary,
                "readback": readback_output_summary,
            },
            "runtime": {
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "seed": int(args.seed),
                "dist_backend": dist.get_backend() if dist.is_initialized() else None,
                "dist_init_file": str(dist_init_file) if dist_init_file else None,
                "cuda_max_memory_allocated": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            },
        }
    except Exception as exc:
        invalid = {
            "status": "invalid",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "condition": {
                "name": "leanstral_qwen3_body_dspark_tiny_train_export",
                "label": "invalid_qwen3_body_train_export_readback_smoke",
                "argv": list(sys.argv),
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
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        invalid_path = output_dir / "INVALID_RUN.json"
        dump_json(invalid_path, invalid)
        (output_dir / "INVALID_RUN.txt").write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    result_path = output_dir / "qwen3_tiny_train_export_result.json"
    dump_json(result_path, result)
    sha_path = output_dir / "qwen3_tiny_train_export_result.sha256"
    sha_path.write_text(
        f"{sha256_file(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    print(str(result_path))


if __name__ == "__main__":
    main()
