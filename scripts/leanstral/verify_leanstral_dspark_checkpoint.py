import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

from deepspec.data.target_cache_dataset import CacheCollator, CacheDataset
from deepspec.modeling.dspark.deepseek2 import DeepseekV2DSparkModel
from scripts.leanstral.probe_leanstral_target import dump_json, sha256_file
from scripts.leanstral.smoke_leanstral_dspark_forward import (
    init_single_process_dist,
    run_git,
    tensor_summary,
)
from scripts.leanstral.train_leanstral_dspark_tiny import (
    dtype_from_name,
    evaluate_loss,
    load_shared_target_weights,
)


DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/dspark-checkpoint-verifies"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def choose_output_dir(output_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-leanstral-dspark-checkpoint-verify").resolve()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_missing_keys(model: torch.nn.Module):
    frozen = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen.append(name)
    return sorted(frozen)


def count_loaded_state(state):
    return {
        "tensor_count": len(state),
        "numel": int(sum(tensor.numel() for tensor in state.values())),
        "keys": sorted(state),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rehydrate a Leanstral DeepseekV2 DSpark tiny-train checkpoint and "
            "verify its deterministic eval loss."
        )
    )
    parser.add_argument("--train-run-dir", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--loss-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output_dir = choose_output_dir(Path(args.output_root).resolve(), args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()
    train_run_dir = Path(args.train_run_dir).resolve()
    train_result_path = train_run_dir / "train_smoke_result.json"
    result = None

    try:
        train_result = load_json(train_result_path)
        if train_result["status"] != "ok":
            raise ValueError(f"Training result status is not ok: {train_result['status']}")

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is false."
            )
        dtype = dtype_from_name(args.dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        dist_init_file = init_single_process_dist(output_dir)
        cache_dir = Path(train_result["inputs"]["cache_dir"]).resolve()
        source_dir = Path(train_result["inputs"]["source_dir"]).resolve()
        source_index_path = Path(
            train_result["inputs"]["source_index"]["path"]
        ).resolve()
        draft_config_path = Path(train_result["draft_config"]["path"]).resolve()
        checkpoint_path = Path(train_result["checkpoint"]["path"]).resolve()
        seed = int(train_result["runtime"]["seed"])
        expected_loss = float(train_result["training"]["final_eval_loss"])

        dataset = CacheDataset(str(cache_dir))
        try:
            features = [dataset[idx] for idx in range(len(dataset))]
        finally:
            dataset.close()
        batch = CacheCollator()(features)
        batch = {key: value.to(device=device) for key, value in batch.items()}

        draft_config = DeepseekV2Config.from_json_file(str(draft_config_path))
        draft_config._attn_implementation = str(
            train_result["draft_config"]["attn_implementation"]
        )
        model = DeepseekV2DSparkModel(draft_config).to(device=device, dtype=dtype)
        source_index = load_json(source_index_path)
        shared_weight_records = load_shared_target_weights(
            model=model,
            source_dir=source_dir,
            index=source_index,
            dtype=dtype,
            device=device,
        )
        state = load_file(str(checkpoint_path), device="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        missing_keys = sorted(incompatible.missing_keys)
        unexpected_keys = sorted(incompatible.unexpected_keys)
        expected_missing = expected_missing_keys(model)
        if missing_keys != expected_missing:
            raise RuntimeError(
                "Unexpected missing keys after trainable checkpoint load: "
                f"{missing_keys} != {expected_missing}"
            )
        if unexpected_keys:
            raise RuntimeError(
                "Unexpected keys after trainable checkpoint load: "
                f"{unexpected_keys}"
            )

        actual_loss, eval_output_summary = evaluate_loss(
            model,
            batch,
            argparse.Namespace(
                loss_decay_gamma=float(
                    train_result["training"].get("loss_decay_gamma", 4.0)
                ),
                ce_loss_alpha=float(
                    train_result["training"].get("ce_loss_alpha", 0.1)
                ),
                l1_loss_alpha=float(
                    train_result["training"].get("l1_loss_alpha", 0.9)
                ),
                confidence_head_alpha=float(
                    train_result["training"].get("confidence_head_alpha", 0.0)
                ),
            ),
            dtype,
            seed=seed + 10_000,
        )
        loss_abs_diff = abs(actual_loss - expected_loss)
        if loss_abs_diff > float(args.loss_tolerance):
            raise RuntimeError(
                f"Rehydrated loss mismatch: {actual_loss} vs {expected_loss} "
                f"(abs diff {loss_abs_diff})"
            )

        result = {
            "status": "ok",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "condition": {
                "name": "leanstral_deepseek2_dspark_checkpoint_rehydrate_verify",
                "label": "controlled_checkpoint_rehydrate_smoke",
                "note": (
                    "Reloads trainable DSpark checkpoint state plus referenced "
                    "frozen Leanstral embedding/output tensors and verifies the "
                    "same deterministic eval loss. This is not a quality benchmark."
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
                "train_run_dir": str(train_run_dir),
                "train_result": {
                    "path": str(train_result_path),
                    "sha256": sha256_file(train_result_path),
                },
                "cache_dir": str(cache_dir),
                "source_dir": str(source_dir),
                "source_index": {
                    "path": str(source_index_path),
                    "sha256": sha256_file(source_index_path),
                },
                "draft_config": {
                    "path": str(draft_config_path),
                    "sha256": sha256_file(draft_config_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                },
                "shared_target_weights": shared_weight_records,
            },
            "load": {
                "checkpoint_state": count_loaded_state(state),
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
            },
            "verification": {
                "expected_final_eval_loss": expected_loss,
                "actual_eval_loss": actual_loss,
                "loss_abs_diff": loss_abs_diff,
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
                "eval": eval_output_summary,
            },
            "runtime": {
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "seed": seed,
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
                "name": "leanstral_deepseek2_dspark_checkpoint_rehydrate_verify",
                "label": "invalid_checkpoint_rehydrate_smoke",
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

    result_path = output_dir / "checkpoint_verify_result.json"
    dump_json(result_path, result)
    sha_path = output_dir / "checkpoint_verify_result.sha256"
    sha_path.write_text(
        f"{sha256_file(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    print(str(result_path))


if __name__ == "__main__":
    main()
