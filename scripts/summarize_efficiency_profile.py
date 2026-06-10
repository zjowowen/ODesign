#!/usr/bin/env python3
"""Summarize ODesign training-efficiency profiling artifacts.

The input is a record directory produced by the `.cluster_operator` efficiency
launch scripts. The script intentionally depends only on the Python standard
library so it can run inside the cluster runtime snapshot.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import statistics
from typing import Any


TIMING_KEYS = [
    "data_wait_sec",
    "to_device_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "optimizer_sec",
    "empty_cache_sec",
    "train_step_sec",
    "microbatch_total_sec",
]

MEMORY_KEYS = [
    "gpu_mem_allocated_mib",
    "gpu_mem_reserved_mib",
    "gpu_mem_max_allocated_mib",
    "gpu_mem_max_reserved_mib",
]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_int_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    return int(text)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize_values(values: list[float]) -> dict[str, float]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {}
    values_sorted = sorted(values)
    return {
        "mean": statistics.fmean(values_sorted),
        "p50": statistics.median(values_sorted),
        "p90": quantile(values_sorted, 0.9),
        "max": values_sorted[-1],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "line_count": len(rows),
        "optimizer_update_rows": sum(1 for row in rows if row.get("optimizer_update")),
    }
    if rows:
        info["global_step_first"] = rows[0].get("global_step")
        info["global_step_last"] = rows[-1].get("global_step")
        info["step_first"] = rows[0].get("step")
        info["step_last"] = rows[-1].get("step")

    losses = [row["loss"] for row in rows if finite_number(row.get("loss"))]
    if losses:
        info["loss_first"] = losses[0]
        info["loss_last"] = losses[-1]
        info["loss_min"] = min(losses)
        info["loss_max"] = max(losses)
        info["loss_mean"] = statistics.fmean(losses)

    for key in TIMING_KEYS:
        values = [row[key] for row in rows if finite_number(row.get(key))]
        stats = summarize_values(values)
        for stat_key, stat_value in stats.items():
            info[f"{key}_{stat_key}"] = stat_value

    for key in MEMORY_KEYS:
        values = [row[key] for row in rows if finite_number(row.get(key))]
        if values:
            info[key] = max(values)

    return info


def first_update_excluded(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for idx, row in enumerate(rows):
        if row.get("optimizer_update"):
            return rows[idx + 1 :]
    return rows[:]


def load_profile_rows(record_dir: Path) -> dict[str, list[dict[str, Any]]]:
    profile_rows: dict[str, list[dict[str, Any]]] = {}
    for path_str in sorted(glob.glob(str(record_dir / "profile_steps*.jsonl"))):
        path = Path(path_str)
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        profile_rows[path.name] = rows
    return profile_rows


def parse_gpu_memory(record_dir: Path) -> dict[str, Any]:
    path = record_dir / "gpu_memory.csv"
    result: dict[str, Any] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            idx = row[1].strip()
            used_text = row[3].replace(" MiB", "").strip()
            total_text = row[4].replace(" MiB", "").strip()
            util_text = row[5].replace(" %", "").strip()
            try:
                used = int(used_text)
            except ValueError:
                continue
            gpu = result.setdefault(
                idx,
                {
                    "memory_used_peak_mib": 0,
                    "memory_total_mib": int(total_text)
                    if total_text.isdigit()
                    else None,
                    "utilization_peak_pct": 0,
                },
            )
            gpu["memory_used_peak_mib"] = max(gpu["memory_used_peak_mib"], used)
            try:
                gpu["utilization_peak_pct"] = max(
                    gpu["utilization_peak_pct"], int(util_text)
                )
            except ValueError:
                pass
    return result


def build_summary(record_dir: Path) -> dict[str, Any]:
    start_epoch = read_int_file(record_dir / "start_epoch.txt")
    end_epoch = read_int_file(record_dir / "end_epoch.txt")
    returncode = read_int_file(record_dir / "returncode.txt")

    summary: dict[str, Any] = {
        "record_dir": str(record_dir),
        "returncode": returncode,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "wall_time_sec": end_epoch - start_epoch
        if start_epoch is not None and end_epoch is not None
        else None,
        "profile_files": {},
        "gpu_memory_peak_mib": parse_gpu_memory(record_dir),
    }

    profile_rows = load_profile_rows(record_dir)
    for name, rows in profile_rows.items():
        summary["profile_files"][name] = {
            "all_rows": summarize_rows(rows),
            "drop_first_row": summarize_rows(rows[1:]),
            "drop_first_accumulation_window": summarize_rows(first_update_excluded(rows)),
        }
    return summary


def fmt_seconds(value: Any) -> str:
    if not finite_number(value):
        return "-"
    return f"{float(value):.2f}s"


def markdown_table(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("| profile | rows | update rows | microbatch mean | forward mean | backward mean | data wait mean | empty cache mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, file_summary in summary.get("profile_files", {}).items():
        steady = file_summary.get("drop_first_row", {})
        lines.append(
            "| {name} | {rows} | {updates} | {micro} | {forward} | {backward} | {data_wait} | {empty_cache} |".format(
                name=name,
                rows=steady.get("line_count", 0),
                updates=steady.get("optimizer_update_rows", 0),
                micro=fmt_seconds(steady.get("microbatch_total_sec_mean")),
                forward=fmt_seconds(steady.get("forward_sec_mean")),
                backward=fmt_seconds(steady.get("backward_sec_mean")),
                data_wait=fmt_seconds(steady.get("data_wait_sec_mean")),
                empty_cache=fmt_seconds(steady.get("empty_cache_sec_mean")),
            )
        )
    lines.append("")
    lines.append("| GPU | peak used | total | peak util |")
    lines.append("| --- | ---: | ---: | ---: |")
    for gpu, gpu_summary in sorted(summary.get("gpu_memory_peak_mib", {}).items()):
        lines.append(
            "| {gpu} | {used} MiB | {total} MiB | {util}% |".format(
                gpu=gpu,
                used=gpu_summary.get("memory_used_peak_mib", "-"),
                total=gpu_summary.get("memory_total_mib", "-"),
                util=gpu_summary.get("utilization_peak_pct", "-"),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ODesign efficiency profile record directories."
    )
    parser.add_argument("record_dir", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <record_dir>/summary_steady.json.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional Markdown summary table output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_dir = args.record_dir.resolve()
    summary = build_summary(record_dir)

    output_json = args.output_json or (record_dir / "summary_steady.json")
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = markdown_table(summary)
    if args.output_md:
        args.output_md.write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
