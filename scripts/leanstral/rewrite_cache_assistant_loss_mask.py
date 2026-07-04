#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

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


DEFAULT_OUTPUT_ROOT = "/srv/models-hdd/models/leanstral-nvfp4/target-cache-runs"
DEFAULT_TEKKEN_JSON = "/srv/models-hdd/models/leanstral-nvfp4/upstream/tekken.json"
ASSISTANT_MARKER = "<|im_start|>assistant\n"
CORE_MANIFEST_KEYS = {
    "version",
    "num_samples",
    "num_shards",
    "target_layer_ids",
    "hidden_dtype",
    "token_dtype",
    "mask_dtype",
    "index_record_size",
    "hidden_size",
    "shards",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_number"] = line_number
            rows.append(item)
    return rows


def run_git(repo: Path, args):
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
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


def choose_output_dir(output_root: Path, output_dir: str | None, label: str) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    return (output_root / f"{timestamp()}-{label}").resolve()


def load_cache_prompt_records(path: Path, expected_count: int):
    records = read_jsonl(path)
    if len(records) != int(expected_count):
        raise ValueError(
            f"{path} has {len(records)} records, expected {expected_count}."
        )
    by_sample_id = {}
    for index, record in enumerate(records):
        sample_id = int(record.get("sample_id", index))
        if sample_id in by_sample_id:
            raise ValueError(f"Duplicate sample_id {sample_id} in {path}.")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{path}:{record['_line_number']} missing prompt string.")
        marker_pos = prompt.find(ASSISTANT_MARKER)
        if marker_pos < 0:
            raise ValueError(
                f"{path}:{record['_line_number']} missing {ASSISTANT_MARKER!r}."
            )
        prefix_end = marker_pos + len(ASSISTANT_MARKER)
        by_sample_id[sample_id] = {
            "sample_id": sample_id,
            "line_number": int(record["_line_number"]),
            "id": record.get("id"),
            "domain": record.get("domain"),
            "source_name": record.get("source_name"),
            "source_path": record.get("source_path"),
            "source_line": record.get("source_line"),
            "prompt": prompt,
            "prefix_text": prompt[:prefix_end],
        }
    missing = [sample_id for sample_id in range(expected_count) if sample_id not in by_sample_id]
    if missing:
        raise ValueError(f"{path} is missing sample ids: {missing[:10]}")
    return by_sample_id


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
                    "loss_tokens": int(sample["loss_mask"].sum().item()),
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


def main():
    started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite an existing DeepSpec target cache so loss_mask covers only "
            "assistant-continuation tokens. Input ids and hidden states are copied "
            "from the parent cache."
        )
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-prompts-jsonl", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--tekken-json", default=DEFAULT_TEKKEN_JSON)
    parser.add_argument("--max-shard-bytes", type=int, default=1024**3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_prompts_jsonl = Path(args.cache_prompts_jsonl).resolve()
    tekken_json = Path(args.tekken_json).resolve()
    output_dir = choose_output_dir(
        Path(args.output_root).resolve(),
        args.output_dir,
        "assistant-loss-cache",
    )

    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)
    if not cache_prompts_jsonl.exists():
        raise FileNotFoundError(cache_prompts_jsonl)
    if not tekken_json.exists():
        raise FileNotFoundError(tekken_json)

    prepare_target_cache_output_dir(str(output_dir))
    rank_dir = output_dir / "_tmp" / "rank_0"
    rank_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tekkenizer.from_file(str(tekken_json))
    parent_manifest = load_json(cache_dir / "manifest.json")
    dataset = CacheDataset(str(cache_dir))
    records = load_cache_prompt_records(
        cache_prompts_jsonl,
        expected_count=len(dataset),
    )
    writer = LocalTargetCacheWriter(
        rank_dir=str(rank_dir),
        max_shard_bytes=int(args.max_shard_bytes),
    )

    samples = []
    try:
        for sample_id in range(len(dataset)):
            sample = dataset[sample_id]
            input_ids = sample["input_ids"].to(dtype=torch.int32).contiguous()
            seq_len = int(input_ids.shape[0])
            record = records[sample_id]
            prefix_ids = tokenizer.encode(
                record["prefix_text"],
                bos=True,
                eos=False,
            )
            if len(prefix_ids) >= seq_len:
                raise ValueError(
                    f"Sample {sample_id} prefix length {len(prefix_ids)} is not "
                    f"shorter than seq_len {seq_len}."
                )
            actual_prefix = input_ids[: len(prefix_ids)].tolist()
            if actual_prefix != prefix_ids:
                raise ValueError(
                    f"Sample {sample_id} Tekken prefix ids do not match parent cache."
                )

            attention_mask = torch.ones((seq_len,), dtype=torch.uint8)
            loss_mask = torch.zeros((seq_len,), dtype=torch.uint8)
            loss_mask[len(prefix_ids) :] = 1
            writer.write_sample(
                sample_id=sample_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                target_hidden_states=sample["target_hidden_states"],
                target_last_hidden_states=sample["target_last_hidden_states"],
            )
            samples.append(
                {
                    "sample_id": int(sample_id),
                    "id": record["id"],
                    "domain": record["domain"],
                    "source_name": record["source_name"],
                    "source_path": record["source_path"],
                    "source_line": record["source_line"],
                    "cache_prompt_line_number": int(record["line_number"]),
                    "seq_len": int(seq_len),
                    "prefix_token_count": int(len(prefix_ids)),
                    "loss_tokens": int(loss_mask.sum().item()),
                    "first_loss_token_id": int(input_ids[len(prefix_ids)].item()),
                }
            )
    finally:
        writer.close()
        dataset.close()

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

    prefix_counts = [int(sample["prefix_token_count"]) for sample in samples]
    loss_tokens = [int(sample["loss_tokens"]) for sample in samples]
    inherited_manifest_fields = {
        key: value
        for key, value in parent_manifest.items()
        if key not in CORE_MANIFEST_KEYS
    }
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=parent_manifest["target_layer_ids"],
        hidden_size=int(parent_manifest["hidden_size"]),
        extra_fields={
            **inherited_manifest_fields,
            "condition_name": "rewrite_target_cache_assistant_loss_mask",
            "condition_label": "controlled_assistant_continuation_loss_mask_rewrite",
            "parent_cache_dir": str(cache_dir),
            "parent_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "source_jsonl_paths": [str(cache_prompts_jsonl)],
            "source_jsonl_sha256": sha256_file(cache_prompts_jsonl),
            "loss_mask_policy": "assistant-continuation",
            "parent_loss_mask_policy": parent_manifest.get("loss_mask_policy"),
            "tekken_json": str(tekken_json),
            "tekken_json_sha256": sha256_file(tekken_json),
            "prefix_token_counts": prefix_counts,
            "loss_tokens": loss_tokens,
            "min_prefix_tokens": min(prefix_counts),
            "max_prefix_tokens": max(prefix_counts),
            "min_loss_tokens": min(loss_tokens),
            "max_loss_tokens": max(loss_tokens),
            "git_sha": run_git(repo_root, ["rev-parse", "HEAD"])["stdout"],
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
            "name": "rewrite_target_cache_assistant_loss_mask",
            "label": "controlled_assistant_continuation_loss_mask_rewrite",
            "argv": sys.argv,
            "working_directory": os.getcwd(),
            "source_script": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "git": {
                "head": run_git(repo_root, ["rev-parse", "HEAD"]),
                "status_short": run_git(repo_root, ["status", "--short"]),
            },
        },
        "inputs": {
            "parent_cache_dir": str(cache_dir),
            "parent_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "cache_prompts_jsonl": {
                "path": str(cache_prompts_jsonl),
                "sha256": sha256_file(cache_prompts_jsonl),
                "records": len(samples),
            },
            "tekken_json": {
                "path": str(tekken_json),
                "sha256": sha256_file(tekken_json),
            },
        },
        "cache": {
            "dir": str(output_dir),
            "num_samples": int(num_samples),
            "num_shards": len(shards),
            "target_layer_ids": [int(layer) for layer in parent_manifest["target_layer_ids"]],
            "hidden_size": int(parent_manifest["hidden_size"]),
            "loss_mask_policy": "assistant-continuation",
            "parent_loss_mask_policy": parent_manifest.get("loss_mask_policy"),
            "prefix_token_counts": prefix_counts,
            "loss_tokens": loss_tokens,
        },
        "samples": samples,
        "output_artifacts": output_artifacts,
        "verification": verification,
    }
    result_path = output_dir / "cache_rewrite_result.json"
    atomic_json_dump(result, str(result_path))
    result_sha_path = output_dir / "cache_rewrite_result.sha256"
    with result_sha_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{sha256_file(result_path)}  {result_path.name}\n")
    print(str(result_path))


if __name__ == "__main__":
    main()
