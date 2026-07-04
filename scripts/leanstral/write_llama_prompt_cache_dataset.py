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
from scripts.leanstral.write_llama_probe_cache_sample import (
    last_hidden_file_from_probe,
    layer_file_from_probe,
    load_json,
    parse_layers,
    read_f32_matrix,
    run_git,
    sha256_file,
    summarize_array,
    timestamp,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    "/home/frosty40/models/leanstral-nvfp4/artifacts/"
    "Leanstral-1.5-119B-A6B-NVFP4.gguf"
)
DEFAULT_PROBE_BIN = (
    "/home/frosty40/llama-nvfp4-dspark/build-dspark/bin/"
    "llama-dspark-cache-probe"
)
DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/target-cache-runs"


def choose_output_dir(output_root: Path, output_dir: str | None, label: str) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-{label}").resolve()


def load_prompt_records(paths, *, prompt_field: str, max_samples: int):
    records = []
    for path in paths:
        source_path = Path(path).resolve()
        with open(source_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if prompt_field not in item:
                    raise KeyError(
                        f"{source_path}:{line_number} missing prompt field "
                        f"{prompt_field!r}."
                    )
                prompt = item[prompt_field]
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(
                        f"{source_path}:{line_number} prompt field "
                        f"{prompt_field!r} must be a non-empty string."
                    )
                records.append(
                    {
                        "source_path": str(source_path),
                        "line_number": int(line_number),
                        "prompt": prompt,
                        "raw": item,
                    }
                )
                if max_samples > 0 and len(records) >= max_samples:
                    return records
    return records


def build_probe_command(args, *, prompt_file: Path, probe_dir: Path):
    command = [
        str(Path(args.probe_bin).resolve()),
        "-m",
        str(Path(args.model).resolve()),
        "--out-dir",
        str(probe_dir),
        "--prompt-file",
        str(prompt_file),
        "--layers",
        str(args.layers),
        "-ngl",
        str(args.n_gpu_layers),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads_batch),
    ]
    if args.ctx_size > 0:
        command.extend(["-c", str(args.ctx_size)])
    if args.batch_size > 0:
        command.extend(["-b", str(args.batch_size)])
    if args.ubatch_size > 0:
        command.extend(["-ub", str(args.ubatch_size)])
    return command


def run_probe(command, *, cwd: Path):
    started = datetime.now(timezone.utc)
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    finished = datetime.now(timezone.utc)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.decode("utf-8", errors="replace"),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_at_utc": finished.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": (finished - started).total_seconds(),
    }


def load_probe_tensors(probe_dir: Path, *, layer_ids):
    probe_json_path = probe_dir / "probe_result.json"
    probe_json = load_json(probe_json_path)
    seq_len = int(probe_json["tokenization"]["n_tokens"])
    tokens = np.asarray(probe_json["tokenization"]["tokens"], dtype=np.int32)
    hidden_size = int(probe_json["model"]["n_embd"])
    if tokens.shape != (seq_len,):
        raise ValueError(f"Token count mismatch in {probe_json_path}.")
    if probe_json["outputs"]["dtype"] != "float32":
        raise ValueError(f"Unsupported probe dtype: {probe_json['outputs']['dtype']}")
    if not bool(probe_json["outputs"]["row_major"]):
        raise ValueError("Probe tensors must be row-major.")

    layer_arrays = []
    layer_artifacts = {}
    for layer_id in layer_ids:
        layer_path = layer_file_from_probe(probe_dir, probe_json, layer_id)
        layer_array = read_f32_matrix(
            layer_path,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )
        layer_arrays.append(layer_array)
        layer_artifacts[f"layer_{layer_id:03d}_input_f32"] = {
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
    target_hidden = np.concatenate(layer_arrays, axis=1)
    return {
        "probe_json": probe_json,
        "probe_result_artifact": {
            "path": str(probe_json_path),
            "sha256": sha256_file(probe_json_path),
            "bytes": probe_json_path.stat().st_size,
        },
        "layer_artifacts": layer_artifacts,
        "target_last_hidden_artifact": {
            "path": str(last_hidden_path),
            "sha256": sha256_file(last_hidden_path),
            "bytes": last_hidden_path.stat().st_size,
            "stats": summarize_array(last_hidden),
        },
        "tokens": tokens,
        "hidden_size": hidden_size,
        "target_hidden": target_hidden,
        "target_last_hidden": last_hidden,
    }


def make_loss_mask(seq_len: int, policy: str):
    loss_mask = np.ones((seq_len,), dtype=np.uint8)
    if policy == "not-bos":
        loss_mask[0] = 0
    return loss_mask


def verify_cache(cache_dir: Path):
    dataset = CacheDataset(str(cache_dir))
    try:
        samples = [dataset[idx] for idx in range(len(dataset))]
        batch = CacheCollator()(samples)
        return {
            "status": "ok",
            "dataset_len": len(dataset),
            "sample_shapes": [
                {
                    "input_ids": [int(dim) for dim in sample["input_ids"].shape],
                    "loss_mask": [int(dim) for dim in sample["loss_mask"].shape],
                    "target_hidden_states": [
                        int(dim) for dim in sample["target_hidden_states"].shape
                    ],
                    "target_last_hidden_states": [
                        int(dim) for dim in sample["target_last_hidden_states"].shape
                    ],
                    "target_hidden_states_dtype": str(
                        sample["target_hidden_states"].dtype
                    ).replace("torch.", ""),
                    "target_last_hidden_states_dtype": str(
                        sample["target_last_hidden_states"].dtype
                    ).replace("torch.", ""),
                }
                for sample in samples
            ],
            "batch_shapes": {
                key: [int(dim) for dim in value.shape]
                for key, value in batch.items()
            },
        }
    finally:
        dataset.close()


def input_file_artifacts(paths):
    artifacts = []
    for path in paths:
        source_path = Path(path).resolve()
        artifacts.append(
            {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            }
        )
    return artifacts


def main():
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DeepSpec target-cache shard from JSONL prompts by "
            "running the llama.cpp Leanstral activation probe per prompt."
        )
    )
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--probe-bin", default=DEFAULT_PROBE_BIN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--layers", default="1,9,17,25,33")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--loss-mask",
        choices=("all", "not-bos"),
        default="all",
        help="Prompt-shard smoke policy. This is not a final training recipe.",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=1024**3)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--ctx-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--ubatch-size", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threads-batch", type=int, default=8)
    parser.add_argument(
        "--model-sha256",
        default="4668fd2a6d2764de250489852230e96238eb5c0d7495866eb3b9c6c4fac5c7da",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    model_path = Path(args.model).resolve()
    probe_bin = Path(args.probe_bin).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not probe_bin.exists():
        raise FileNotFoundError(probe_bin)

    layer_ids = parse_layers(args.layers)
    records = load_prompt_records(
        args.input_jsonl,
        prompt_field=args.prompt_field,
        max_samples=int(args.max_samples),
    )
    if not records:
        raise ValueError("No prompt records selected.")

    output_dir = choose_output_dir(
        output_root=Path(args.output_root).resolve(),
        output_dir=args.output_dir,
        label="llama-prompt-cache-dataset",
    )
    prepare_target_cache_output_dir(str(output_dir))
    prompt_dir = output_dir / "prompts"
    probe_root = output_dir / "probes"
    rank_dir = output_dir / "_tmp" / "rank_0"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    probe_root.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)

    writer = LocalTargetCacheWriter(
        rank_dir=str(rank_dir),
        max_shard_bytes=int(args.max_shard_bytes),
    )
    samples = []
    hidden_size = None
    try:
        for sample_id, record in enumerate(records):
            prompt_file = prompt_dir / f"sample-{sample_id:05d}.txt"
            with open(prompt_file, "w", encoding="utf-8") as handle:
                handle.write(record["prompt"])
            probe_dir = probe_root / f"sample-{sample_id:05d}"
            probe_dir.mkdir(parents=True, exist_ok=True)
            command = build_probe_command(
                args,
                prompt_file=prompt_file,
                probe_dir=probe_dir,
            )
            run_result = run_probe(command, cwd=REPO_ROOT)
            atomic_json_dump(run_result, str(probe_dir / "probe_command.json"))
            if run_result["returncode"] != 0:
                raise RuntimeError(
                    f"Probe failed for sample {sample_id} with return code "
                    f"{run_result['returncode']}. See {probe_dir / 'probe_command.json'}."
                )
            tensors = load_probe_tensors(probe_dir, layer_ids=layer_ids)
            if hidden_size is None:
                hidden_size = int(tensors["hidden_size"])
            elif int(hidden_size) != int(tensors["hidden_size"]):
                raise ValueError("Hidden size changed across probe samples.")

            seq_len = int(tensors["tokens"].shape[0])
            attention_mask = np.ones((seq_len,), dtype=np.uint8)
            loss_mask = make_loss_mask(seq_len, args.loss_mask)
            writer.write_sample(
                sample_id=sample_id,
                input_ids=torch.from_numpy(tensors["tokens"]),
                attention_mask=torch.from_numpy(attention_mask),
                loss_mask=torch.from_numpy(loss_mask),
                target_hidden_states=torch.from_numpy(tensors["target_hidden"]),
                target_last_hidden_states=torch.from_numpy(tensors["target_last_hidden"]),
            )
            samples.append(
                {
                    "sample_id": int(sample_id),
                    "source_path": record["source_path"],
                    "line_number": int(record["line_number"]),
                    "prompt_file": {
                        "path": str(prompt_file),
                        "sha256": sha256_file(prompt_file),
                        "bytes": prompt_file.stat().st_size,
                    },
                    "probe_dir": str(probe_dir),
                    "probe_command": {
                        "path": str(probe_dir / "probe_command.json"),
                        "sha256": sha256_file(probe_dir / "probe_command.json"),
                        "returncode": run_result["returncode"],
                        "elapsed_seconds": run_result["elapsed_seconds"],
                    },
                    "probe_result": tensors["probe_result_artifact"],
                    "layer_artifacts": tensors["layer_artifacts"],
                    "target_last_hidden_artifact": tensors[
                        "target_last_hidden_artifact"
                    ],
                    "seq_len": seq_len,
                    "loss_tokens": int(loss_mask.sum()),
                    "target_hidden_states": summarize_array(tensors["target_hidden"]),
                    "target_last_hidden_states": summarize_array(
                        tensors["target_last_hidden"]
                    ),
                }
            )
    finally:
        writer.close()

    assert hidden_size is not None
    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=len(samples),
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
            "condition_name": "llama_prompt_shard_to_deepspec_cache",
            "condition_label": "controlled_prompt_shard_cache_smoke",
            "source_jsonl_paths": [str(Path(path).resolve()) for path in args.input_jsonl],
            "prompt_field": str(args.prompt_field),
            "loss_mask_policy": str(args.loss_mask),
            "target_model_name_or_path": str(model_path),
            "target_model_sha256": str(args.model_sha256),
            "probe_binary": str(probe_bin),
            "probe_binary_sha256": sha256_file(probe_bin),
            "chat_template": None,
            "max_length": max(int(sample["seq_len"]) for sample in samples),
            "min_loss_tokens": min(int(sample["loss_tokens"]) for sample in samples),
            "git_sha": run_git(["rev-parse", "HEAD"])["stdout"],
        },
    )
    write_target_cache_manifest(output_dir=str(output_dir), manifest=manifest)
    cleanup_target_cache_tmp_dir(str(output_dir))

    verification = verify_cache(output_dir)
    output_artifacts = {}
    for name in ("manifest.json", "samples.idx"):
        path = output_dir / name
        output_artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    for shard in shards:
        path = output_dir / shard["file_name"]
        output_artifacts[shard["file_name"]] = {
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
            "name": "llama_prompt_shard_to_deepspec_cache",
            "label": "controlled_prompt_shard_cache_smoke",
            "note": (
                "This runs the frozen Leanstral NVFP4 llama.cpp activation "
                "probe per prompt and writes a DeepSpec cache shard. It is not "
                "a draft training run or benchmark."
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
        "inputs": {
            "jsonl": input_file_artifacts(args.input_jsonl),
            "model": {
                "path": str(model_path),
                "sha256": str(args.model_sha256),
            },
            "probe_binary": {
                "path": str(probe_bin),
                "sha256": sha256_file(probe_bin),
            },
        },
        "cache": {
            "dir": str(output_dir),
            "target_layer_ids": layer_ids,
            "hidden_size": int(hidden_size),
            "num_samples": int(num_samples),
            "num_shards": len(shards),
            "loss_mask_policy": str(args.loss_mask),
            "seq_lens": [int(sample["seq_len"]) for sample in samples],
            "loss_tokens": [int(sample["loss_tokens"]) for sample in samples],
        },
        "samples": samples,
        "output_artifacts": output_artifacts,
        "verification": verification,
    }
    result_path = output_dir / "cache_dataset_result.json"
    atomic_json_dump(result, str(result_path))
    result_sha_path = output_dir / "cache_dataset_result.sha256"
    with open(result_sha_path, "w", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(result_path)}  {result_path.name}\n")
    print(str(result_path))


if __name__ == "__main__":
    main()
