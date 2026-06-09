#!/usr/bin/env python3
"""Summarize ODesign module-level profiling JSONL files.

The script intentionally uses only the Python standard library so it can run in
the cluster runtime snapshot without installing plotting dependencies.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import statistics
from typing import Any


STAGE_KEYS = [
    "forward_sec",
    "backward_sec",
    "loss_sec",
    "microbatch_total_sec",
]

FORWARD_MODULE_KEYS = [
    "module_profile_prepare_inputs_sec",
    "module_profile_pairformer_sec",
    "module_profile_pairwise_head_sec",
    "module_profile_diffusion_sec",
    "module_profile_permutation_sec",
]

BACKWARD_MODULE_KEYS = [
    "module_profile_backward_pairformer_sec",
    "module_profile_backward_msa_sec",
    "module_profile_backward_pairwise_head_sec",
    "module_profile_backward_diffusion_module_sec",
]

SUMMARY_KEYS = STAGE_KEYS + FORWARD_MODULE_KEYS + BACKWARD_MODULE_KEYS


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_profile_rows(record_dir: Path) -> dict[str, list[dict[str, Any]]]:
    profile_rows: dict[str, list[dict[str, Any]]] = {}
    for path_str in sorted(glob.glob(str(record_dir / "profile_steps*.jsonl"))):
        path = Path(path_str)
        profile_rows[path.name] = read_jsonl(path)
    return profile_rows


def quantile(sorted_values: list[float], q: float) -> float:
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
    values = [float(value) for value in values if math.isfinite(float(value))]
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
    summary: dict[str, Any] = {"row_count": len(rows)}
    for key in SUMMARY_KEYS:
        values = [row[key] for row in rows if finite_number(row.get(key))]
        if values:
            summary[key] = summarize_values(values)
    losses = [row["loss"] for row in rows if finite_number(row.get("loss"))]
    if losses:
        summary["loss"] = summarize_values(losses)
    return summary


def first_update_excluded(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for idx, row in enumerate(rows):
        if row.get("optimizer_update"):
            return rows[idx + 1 :]
    return rows[:]


def build_summary(record_dir: Path) -> dict[str, Any]:
    profile_rows = load_profile_rows(record_dir)
    combined_all: list[dict[str, Any]] = []
    combined_drop_first: list[dict[str, Any]] = []
    combined_drop_first_update: list[dict[str, Any]] = []
    files: dict[str, Any] = {}

    for name, rows in profile_rows.items():
        drop_first = rows[1:]
        drop_first_update = first_update_excluded(rows)
        files[name] = {
            "all_rows": summarize_rows(rows),
            "drop_first_row": summarize_rows(drop_first),
            "drop_first_accumulation_window": summarize_rows(drop_first_update),
        }
        combined_all.extend(rows)
        combined_drop_first.extend(drop_first)
        combined_drop_first_update.extend(drop_first_update)

    return {
        "record_dir": str(record_dir),
        "profile_files": files,
        "combined": {
            "all_rows": summarize_rows(combined_all),
            "drop_first_row": summarize_rows(combined_drop_first),
            "drop_first_accumulation_window": summarize_rows(
                combined_drop_first_update
            ),
        },
    }


def fmt_seconds(value: Any) -> str:
    if not finite_number(value):
        return "-"
    return f"{float(value):.3f}s"


def mean(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key, {})
    if isinstance(value, dict) and finite_number(value.get("mean")):
        return float(value["mean"])
    return None


def table_for_keys(
    title: str,
    section_summary: dict[str, Any],
    keys: list[str],
    denominator_key: str | None,
) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| field | mean | p50 | p90 | max | share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    denominator = mean(section_summary, denominator_key) if denominator_key else None
    for key in keys:
        stats = section_summary.get(key, {})
        if not isinstance(stats, dict) or not stats:
            continue
        mean_value = stats.get("mean")
        share = "-"
        if finite_number(mean_value) and denominator and denominator > 0:
            share = f"{float(mean_value) / denominator * 100.0:.1f}%"
        lines.append(
            "| {key} | {mean} | {p50} | {p90} | {max} | {share} |".format(
                key=key,
                mean=fmt_seconds(mean_value),
                p50=fmt_seconds(stats.get("p50")),
                p90=fmt_seconds(stats.get("p90")),
                max=fmt_seconds(stats.get("max")),
                share=share,
            )
        )
    lines.append("")
    return lines


def markdown_table(summary: dict[str, Any]) -> str:
    steady = summary["combined"]["drop_first_row"]
    lines = [
        "## Combined Drop-First-Row Summary",
        "",
        f"Rows: `{steady.get('row_count', 0)}`",
        "",
    ]
    lines.extend(table_for_keys("Stage Timers", steady, STAGE_KEYS, "microbatch_total_sec"))
    lines.extend(
        table_for_keys(
            "Forward Modules", steady, FORWARD_MODULE_KEYS, "forward_sec"
        )
    )
    lines.extend(
        table_for_keys(
            "Backward Hook Modules", steady, BACKWARD_MODULE_KEYS, "backward_sec"
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def svg_bar_chart(summary: dict[str, Any]) -> str:
    steady = summary["combined"]["drop_first_row"]
    rows: list[tuple[str, float, str]] = []
    for key in FORWARD_MODULE_KEYS:
        value = mean(steady, key)
        if value is not None:
            rows.append((key.replace("module_profile_", ""), value, "#2563eb"))
    for key in BACKWARD_MODULE_KEYS:
        value = mean(steady, key)
        if value is not None:
            rows.append((key.replace("module_profile_backward_", "bwd_"), value, "#dc2626"))

    width = 1200
    row_h = 46
    top = 110
    left = 300
    chart_w = 720
    height = top + max(len(rows), 1) * row_h + 90
    max_value = max((value for _, value, _ in rows), default=1.0)
    scale = chart_w / max_value if max_value > 0 else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="ODesign module-level profile">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="44" font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700" fill="#111827">ODesign Module-Level Training Profile</text>',
        '<text x="40" y="76" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#4b5563">Combined drop-first-row means across profile ranks. Blue = forward module timer; red = backward hook timer.</text>',
    ]
    for idx, (label, value, color) in enumerate(rows):
        y = top + idx * row_h
        bar_w = max(1.0, value * scale)
        safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'<text x="40" y="{y + 29}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#111827">{safe_label}</text>')
        lines.append(f'<rect x="{left}" y="{y + 8}" width="{bar_w:.1f}" height="28" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{left + bar_w + 10:.1f}" y="{y + 29}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#111827">{value:.3f}s</text>')
    lines.append(f'<text x="40" y="{height - 30}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#6b7280">Boundary: backward module timings use PyTorch full backward hooks and approximate autograd module spans, not CUDA kernel-level attribution.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ODesign module-level profile record directories."
    )
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_dir = args.record_dir.resolve()
    summary = build_summary(record_dir)
    output_json = args.output_json or (record_dir / "module_profile_summary.json")
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = markdown_table(summary)
    if args.output_md:
        args.output_md.write_text(md, encoding="utf-8")
    if args.output_svg:
        args.output_svg.write_text(svg_bar_chart(summary), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
