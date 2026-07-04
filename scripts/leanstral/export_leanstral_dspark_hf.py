import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
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


DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/dspark-hf-exports"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def choose_output_dir(output_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-leanstral-dspark-hf-export").resolve()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def state_dict_to_cpu(model: torch.nn.Module):
    state = {}
    for name, tensor in model.state_dict().items():
        state[name] = tensor.detach().cpu().contiguous()
    return state


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


def write_weight_index(export_dir: Path, state):
    file_name = "model.safetensors"
    metadata = {
        "total_size": str((export_dir / file_name).stat().st_size),
    }
    index = {
        "metadata": metadata,
        "weight_map": {name: file_name for name in sorted(state)},
    }
    path = export_dir / "model.safetensors.index.json"
    dump_json(path, index)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "weight_count": len(index["weight_map"]),
    }


def build_loss_args(train_result):
    return argparse.Namespace(
        loss_decay_gamma=float(train_result["training"].get("loss_decay_gamma", 4.0)),
        ce_loss_alpha=float(train_result["training"].get("ce_loss_alpha", 0.1)),
        l1_loss_alpha=float(train_result["training"].get("l1_loss_alpha", 0.9)),
        confidence_head_alpha=float(
            train_result["training"].get("confidence_head_alpha", 0.0)
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a rehydrated Leanstral DeepseekV2 DSpark checkpoint as a "
            "full HF-style draft package and verify deterministic readback."
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
    export_dir = output_dir / "hf_model"
    export_dir.mkdir(parents=False, exist_ok=False)
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
        trainable_state = load_file(str(checkpoint_path), device="cpu")
        incompatible = model.load_state_dict(trainable_state, strict=False)
        unexpected_keys = sorted(incompatible.unexpected_keys)
        missing_keys = sorted(incompatible.missing_keys)
        expected_missing = ["embed_tokens.weight", "lm_head.weight"]
        if missing_keys != expected_missing:
            raise RuntimeError(
                f"Unexpected missing keys: {missing_keys} != {expected_missing}"
            )
        if unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")

        dataset = CacheDataset(str(cache_dir))
        try:
            features = [dataset[idx] for idx in range(len(dataset))]
        finally:
            dataset.close()
        batch = CacheCollator()(features)
        batch = {key: value.to(device=device) for key, value in batch.items()}

        pre_export_loss, pre_export_output_summary = evaluate_loss(
            model,
            batch,
            build_loss_args(train_result),
            dtype,
            seed=seed + 10_000,
        )
        pre_export_diff = abs(pre_export_loss - expected_loss)
        if pre_export_diff > float(args.loss_tolerance):
            raise RuntimeError(
                f"Pre-export loss mismatch: {pre_export_loss} vs {expected_loss} "
                f"(abs diff {pre_export_diff})"
            )

        config_path = export_dir / "config.json"
        draft_config.to_json_file(config_path)
        tokenizer_sidecars = copy_tokenizer_sidecars(source_dir, export_dir)

        full_state = state_dict_to_cpu(model)
        state_summary = summarize_state_dict(full_state)
        tensor_summaries = selected_tensor_summaries(full_state)
        weights_path = export_dir / "model.safetensors"
        save_file(full_state, str(weights_path), metadata={"format": "pt"})
        weights_info = {
            "path": str(weights_path),
            "sha256": sha256_file(weights_path),
            "bytes": weights_path.stat().st_size,
        }
        index_info = write_weight_index(export_dir, full_state)

        readback_state = load_file(str(weights_path), device="cpu")
        if sorted(readback_state) != sorted(full_state):
            raise RuntimeError("Readback state keys do not match exported state keys.")
        readback_summary = summarize_state_dict(readback_state)

        verify_config = DeepseekV2Config.from_json_file(str(config_path))
        verify_config._attn_implementation = str(
            train_result["draft_config"]["attn_implementation"]
        )
        verify_model = DeepseekV2DSparkModel(verify_config).to(
            device=device,
            dtype=dtype,
        )
        verify_model.load_state_dict(readback_state, strict=True)
        readback_loss, readback_output_summary = evaluate_loss(
            verify_model,
            batch,
            build_loss_args(train_result),
            dtype,
            seed=seed + 10_000,
        )
        readback_diff = abs(readback_loss - expected_loss)
        if readback_diff > float(args.loss_tolerance):
            raise RuntimeError(
                f"Readback loss mismatch: {readback_loss} vs {expected_loss} "
                f"(abs diff {readback_diff})"
            )

        manifest = {
            "format": "leanstral_deepseek2_dspark_hf_export",
            "source_train_run_dir": str(train_run_dir),
            "source_train_result_sha256": sha256_file(train_result_path),
            "condition_label": "controlled_hf_export_readback_smoke",
            "expected_final_eval_loss": expected_loss,
            "readback_eval_loss": readback_loss,
            "loss_abs_diff": readback_diff,
            "shared_target_weights": shared_weight_records,
            "state_summary": state_summary,
            "weights_sha256": weights_info["sha256"],
            "config_sha256": sha256_file(config_path),
        }
        manifest_path = export_dir / "dspark_export_manifest.json"
        dump_json(manifest_path, manifest)

        result = {
            "status": "ok",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "condition": {
                "name": "leanstral_deepseek2_dspark_hf_export",
                "label": "controlled_hf_export_readback_smoke",
                "note": (
                    "Exports the tiny Leanstral DSpark train checkpoint into a "
                    "full HF-style draft package and verifies deterministic "
                    "readback loss. This is still a tiny mechanics artifact, not "
                    "a release-quality draft or DSpark GGUF."
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
            "export": {
                "output_dir": str(output_dir),
                "hf_model_dir": str(export_dir),
                "config": {
                    "path": str(config_path),
                    "sha256": sha256_file(config_path),
                    "bytes": config_path.stat().st_size,
                    "architectures": list(verify_config.architectures),
                    "model_type": str(verify_config.model_type),
                },
                "weights": weights_info,
                "index": index_info,
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
                "tokenizer_sidecars": tokenizer_sidecars,
                "state_summary": state_summary,
                "selected_tensors": tensor_summaries,
                "readback_state_summary": readback_summary,
            },
            "verification": {
                "expected_final_eval_loss": expected_loss,
                "pre_export_eval_loss": pre_export_loss,
                "pre_export_loss_abs_diff": pre_export_diff,
                "readback_eval_loss": readback_loss,
                "readback_loss_abs_diff": readback_diff,
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
                "pre_export": pre_export_output_summary,
                "readback": readback_output_summary,
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
                "name": "leanstral_deepseek2_dspark_hf_export",
                "label": "invalid_hf_export_readback_smoke",
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

    result_path = output_dir / "hf_export_result.json"
    dump_json(result_path, result)
    sha_path = output_dir / "hf_export_result.sha256"
    sha_path.write_text(
        f"{sha256_file(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    print(str(result_path))


if __name__ == "__main__":
    main()
