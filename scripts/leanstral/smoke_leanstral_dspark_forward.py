import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from deepspec.data.target_cache_dataset import CacheCollator, CacheDataset
from deepspec.modeling.dspark.deepseek2 import DeepseekV2DSparkModel
from deepspec.modeling.dspark.deepseek2.config import build_draft_config
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.utils.config import ConfigNode
from scripts.leanstral.probe_leanstral_target import dump_json, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_CONFIG = (
    "/srv/models-hdd/models/leanstral-nvfp4/hf-remap/"
    "20260704T011143Z-layers-1-global/hf_model"
)
DEFAULT_CACHE_DIR = (
    "/srv/models-hdd/models/leanstral-nvfp4/target-cache-runs/"
    "20260704T015954Z-llama-prompt-cache-dataset"
)
DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/dspark-forward-smokes"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_git(args):
    import subprocess

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


def choose_output_dir(output_root: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-leanstral-dspark-forward-smoke").resolve()


def init_single_process_dist(output_dir: Path):
    if dist.is_available() and dist.is_initialized():
        return None
    init_dir = output_dir / "_dist"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_file = init_dir / "init"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    return init_file


def build_model_args(args, target_layer_ids):
    return ConfigNode(
        {
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
    )


def tensor_summary(tensor: torch.Tensor):
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    nonfinite_count = int(tensor.numel() - finite_count)
    if finite_count:
        values = tensor.detach().float()[finite]
        min_value = float(values.min().item())
        max_value = float(values.max().item())
        mean_value = float(values.mean().item())
    else:
        min_value = 0.0
        max_value = 0.0
        mean_value = 0.0
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite": finite_count,
        "nonfinite": nonfinite_count,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled DeepseekV2/Leanstral DSpark forward and loss "
            "smoke on a DeepSpec target-cache directory."
        )
    )
    parser.add_argument("--target-config", default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument("--draft-intermediate-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=2)
    parser.add_argument("--mask-token-id", type=int, default=0)
    parser.add_argument("--markov-rank", type=int, default=0)
    parser.add_argument("--confidence-head-alpha", type=float, default=0.0)
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
    manifest_path = cache_dir / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
        torch.cuda.reset_peak_memory_stats(device)

    dist_init_file = init_single_process_dist(output_dir)
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
        model.eval()

        with torch.no_grad():
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

        result = {
            "status": "ok",
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "condition": {
                "name": "leanstral_deepseek2_dspark_forward_smoke",
                "label": "controlled_forward_loss_smoke",
                "note": (
                    "This instantiates a random-initialized DeepseekV2 DSpark "
                    "draft and computes forward/loss on the controlled Leanstral "
                    "cache. It is not training and not a benchmark."
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
                "cache_dir": str(cache_dir),
                "cache_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "num_samples": int(cache_manifest["num_samples"]),
                    "target_layer_ids": [
                        int(x) for x in cache_manifest["target_layer_ids"]
                    ],
                    "hidden_size": int(cache_manifest["hidden_size"]),
                },
            },
            "draft_config": {
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
                "enable_confidence_head": bool(draft_config.enable_confidence_head),
                "attn_implementation": str(draft_config._attn_implementation),
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
                "draft_logits": tensor_summary(outputs.draft_logits),
                "target_ids": tensor_summary(outputs.target_ids),
                "eval_mask": tensor_summary(outputs.eval_mask),
                "block_keep_mask": tensor_summary(outputs.block_keep_mask),
                "aligned_target_logits": (
                    tensor_summary(outputs.aligned_target_logits)
                    if outputs.aligned_target_logits is not None
                    else None
                ),
                "loss": float(loss.detach().float().item()),
                "loss_finite": bool(torch.isfinite(loss).item()),
            },
            "runtime": {
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "seed": int(args.seed),
                "dist_backend": dist.get_backend() if dist.is_initialized() else None,
                "cuda_max_memory_allocated": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            },
        }
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    result_path = output_dir / "forward_smoke_result.json"
    dump_json(result_path, result)
    sha_path = output_dir / "forward_smoke_result.sha256"
    sha_path.write_text(
        f"{sha256_file(result_path)}  {result_path.name}\n",
        encoding="utf-8",
    )
    print(str(result_path))


if __name__ == "__main__":
    main()
