import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoConfig

from deepspec.data.target_cache_dataset import CacheCollator, CacheDataset
from deepspec.modeling.dspark.deepseek2 import DeepseekV2DSparkModel
from deepspec.modeling.dspark.deepseek2.config import build_draft_config
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.utils.config import ConfigNode
from scripts.leanstral.probe_leanstral_target import dump_json, sha256_file
from scripts.leanstral.smoke_leanstral_dspark_forward import (
    init_single_process_dist,
    run_git,
    tensor_summary,
)
from scripts.leanstral.smoke_leanstral_weight_remap import (
    SourceTensorReader,
    load_direct_tensor,
    tensor_digest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_CONFIG = (
    "/srv/models-hdd/models/leanstral-nvfp4/hf-remap/"
    "20260704T011143Z-layers-1-global/hf_model"
)
DEFAULT_SOURCE_DIR = "/srv/models-hdd/models/leanstral-nvfp4/upstream"
DEFAULT_CACHE_DIR = (
    "/srv/models-hdd/models/leanstral-nvfp4/target-cache-runs/"
    "20260704T015954Z-llama-prompt-cache-dataset"
)
DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/dspark-train-smokes"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def choose_output_dir(output_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-leanstral-dspark-tiny-train").resolve()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def build_model_args(args, target_layer_ids):
    payload = {
        "block_size": int(args.block_size),
        "num_draft_layers": int(args.num_draft_layers),
        "target_layer_ids": [int(x) for x in target_layer_ids],
        "mask_token_id": int(args.mask_token_id),
        "num_anchors": int(args.num_anchors),
        "markov_rank": int(args.markov_rank),
        "confidence_head_alpha": float(args.confidence_head_alpha),
        "ce_loss_alpha": float(args.ce_loss_alpha),
        "l1_loss_alpha": float(args.l1_loss_alpha),
        "loss_decay_gamma": float(args.loss_decay_gamma),
        "draft_intermediate_size": int(args.draft_intermediate_size),
    }
    if int(args.markov_rank) > 0:
        payload["markov_head_type"] = str(args.markov_head_type)
    if float(args.confidence_head_alpha) > 0.0:
        payload["confidence_head_with_markov"] = bool(
            args.confidence_head_with_markov
        )
    return ConfigNode(payload)


def count_parameters(model: torch.nn.Module):
    total = 0
    trainable = 0
    frozen = 0
    trainable_by_prefix = {}
    for name, param in model.named_parameters():
        numel = int(param.numel())
        total += numel
        top = name.split(".", 1)[0]
        if param.requires_grad:
            trainable += numel
            trainable_by_prefix[top] = trainable_by_prefix.get(top, 0) + numel
        else:
            frozen += numel
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_by_prefix": trainable_by_prefix,
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def grad_summary(model: torch.nn.Module):
    grad_sq = 0.0
    max_abs = 0.0
    grad_param_count = 0
    finite = True
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float()
        grad_param_count += 1
        finite = finite and bool(torch.isfinite(grad).all().item())
        grad_sq += float((grad * grad).sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return {
        "finite": finite,
        "param_count": grad_param_count,
        "global_norm": math.sqrt(grad_sq),
        "max_abs": max_abs,
    }


def trainable_state_dict(model: torch.nn.Module):
    named_params = dict(model.named_parameters())
    state = {}
    for name, param in named_params.items():
        if param.requires_grad:
            state[name] = param.detach().cpu().contiguous()
    return state


def load_shared_target_weights(
    *,
    model: DeepseekV2DSparkModel,
    source_dir: Path,
    index,
    dtype: torch.dtype,
    device: torch.device,
):
    reader = SourceTensorReader(source_dir, index)
    try:
        embed_tensor, embed_record = load_direct_tensor(
            reader,
            index,
            "tok_embeddings.weight",
            dtype,
        )
        output_tensor, output_record = load_direct_tensor(
            reader,
            index,
            "output.weight",
            dtype,
        )
    finally:
        reader.close()

    assert list(embed_tensor.shape) == list(model.embed_tokens.weight.shape), (
        f"Embedding shape mismatch: {tuple(embed_tensor.shape)} != "
        f"{tuple(model.embed_tokens.weight.shape)}"
    )
    assert list(output_tensor.shape) == list(model.lm_head.weight.shape), (
        f"Output head shape mismatch: {tuple(output_tensor.shape)} != "
        f"{tuple(model.lm_head.weight.shape)}"
    )
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            embed_tensor.to(device=device, dtype=dtype, non_blocking=False)
        )
        model.lm_head.weight.copy_(
            output_tensor.to(device=device, dtype=dtype, non_blocking=False)
        )
    model.set_embedding_head_trainable(False)
    return {
        "tok_embeddings.weight": {
            "load_record": embed_record,
            "shape": [int(dim) for dim in embed_tensor.shape],
            "dtype": str(embed_tensor.dtype).replace("torch.", ""),
            "sha256_raw_tensor_bytes": tensor_digest(embed_tensor),
        },
        "output.weight": {
            "load_record": output_record,
            "shape": [int(dim) for dim in output_tensor.shape],
            "dtype": str(output_tensor.dtype).replace("torch.", ""),
            "sha256_raw_tensor_bytes": tensor_digest(output_tensor),
        },
    }


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


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny controlled Leanstral DeepseekV2 DSpark training smoke. "
            "This is a mechanics/checkpoint gate, not a quality benchmark."
        )
    )
    parser.add_argument("--target-config", default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument("--draft-intermediate-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, default=0)
    parser.add_argument("--markov-rank", type=int, default=32)
    parser.add_argument("--markov-head-type", default="vanilla")
    parser.add_argument("--confidence-head-alpha", type=float, default=0.0)
    parser.add_argument("--confidence-head-with-markov", action="store_true")
    parser.add_argument("--ce-loss-alpha", type=float, default=0.1)
    parser.add_argument("--l1-loss-alpha", type=float, default=0.9)
    parser.add_argument("--loss-decay-gamma", type=float, default=4.0)
    args = parser.parse_args()

    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output_dir = choose_output_dir(Path(args.output_root).resolve(), args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    target_config_path = Path(args.target_config).resolve()
    source_dir = Path(args.source_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    source_index_path = source_dir / "consolidated.safetensors.index.json"
    source_params_path = source_dir / "params.json"
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))

    assert int(args.steps) > 0, "--steps must be positive."
    assert float(args.learning_rate) > 0, "--learning-rate must be positive."
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    dtype = dtype_from_name(args.dtype)

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
        torch.cuda.reset_peak_memory_stats(device)

    dist_init_file = init_single_process_dist(output_dir)
    rows = []
    train_log_path = output_dir / "train_log.jsonl"
    result = None
    try:
        dataset = CacheDataset(str(cache_dir))
        try:
            features = [dataset[idx] for idx in range(len(dataset))]
        finally:
            dataset.close()
        batch = CacheCollator()(features)
        batch = {key: value.to(device=device) for key, value in batch.items()}

        target_config = AutoConfig.from_pretrained(
            str(target_config_path),
            local_files_only=True,
        )
        model_args = build_model_args(args, cache_manifest["target_layer_ids"])
        draft_config = build_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        model = DeepseekV2DSparkModel(draft_config).to(device=device, dtype=dtype)
        shared_weight_records = load_shared_target_weights(
            model=model,
            source_dir=source_dir,
            index=source_index,
            dtype=dtype,
            device=device,
        )
        param_counts = count_parameters(model)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

        initial_eval_loss, eval_output_summary = evaluate_loss(
            model,
            batch,
            args,
            dtype,
            seed=int(args.seed) + 10_000,
        )
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
            row = {
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
            rows.append(row)
            write_jsonl(train_log_path, rows)

        final_eval_loss, final_eval_output_summary = evaluate_loss(
            model,
            batch,
            args,
            dtype,
            seed=int(args.seed) + 10_000,
        )

        draft_config_path = output_dir / "draft_config.json"
        draft_config.to_json_file(draft_config_path)
        checkpoint_path = output_dir / "draft_trainable_state.safetensors"
        save_file(trainable_state_dict(model), str(checkpoint_path))

        result = {
            "status": "ok",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "condition": {
                "name": "leanstral_deepseek2_dspark_tiny_train",
                "label": "controlled_mechanics_train_checkpoint_smoke",
                "note": (
                    "Tiny two-prompt DSpark training smoke using frozen Leanstral "
                    "target token embeddings and output head. This proves "
                    "backward, optimizer, and trainable checkpoint mechanics only; "
                    "it is not a quality benchmark and not a release draft."
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
                    "config_json_sha256": sha256_file(
                        target_config_path / "config.json"
                    ),
                },
                "source_dir": str(source_dir),
                "source_index": {
                    "path": str(source_index_path),
                    "sha256": sha256_file(source_index_path),
                },
                "source_params": {
                    "path": str(source_params_path),
                    "sha256": sha256_file(source_params_path),
                },
                "shared_target_weights": shared_weight_records,
                "cache_dir": str(cache_dir),
                "cache_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
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
                "path": str(draft_config_path),
                "sha256": sha256_file(draft_config_path),
                "architectures": list(draft_config.architectures),
                "hidden_size": int(draft_config.hidden_size),
                "intermediate_size": int(draft_config.intermediate_size),
                "num_hidden_layers": int(draft_config.num_hidden_layers),
                "num_attention_heads": int(draft_config.num_attention_heads),
                "num_key_value_heads": int(draft_config.num_key_value_heads),
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
                "initial_eval": eval_output_summary,
                "final_eval": final_eval_output_summary,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "format": "safetensors",
                "contents": (
                    "trainable parameters only; frozen target token embeddings "
                    "and output head are referenced in inputs.shared_target_weights"
                ),
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
                "name": "leanstral_deepseek2_dspark_tiny_train",
                "label": "invalid_mechanics_train_checkpoint_smoke",
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

    if result is None:
        raise RuntimeError("Training smoke did not produce a result payload.")
    result_path = output_dir / "train_smoke_result.json"
    dump_json(result_path, result)
    sha_path = output_dir / "train_smoke_result.sha256"
    sha_path.write_text(
        f"{sha256_file(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    print(str(result_path))


if __name__ == "__main__":
    main()
