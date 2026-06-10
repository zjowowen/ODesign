#!/usr/bin/env python3
"""Summarize selected PyTorch ops by recorded input type and shape.

This complements ``summarize_torch_trace.py`` for traces exported with
``record_shapes=True``.  It uses a lightweight line parser for PyTorch Chrome
traces and avoids loading multi-GB JSON files into memory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


NAME_RE = re.compile(r'"name":\s*"((?:\\.|[^"\\])*)"')
CAT_RE = re.compile(r'"cat":\s*"((?:\\.|[^"\\])*)"')
DUR_RE = re.compile(r'"dur":\s*([0-9]+(?:\.[0-9]+)?)')
PH_X_RE = re.compile(r'"ph":\s*"X"')


DEFAULT_OPS = (
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::is_nonzero",
)


@dataclass
class BucketStats:
    count: int = 0
    total_us: float = 0.0
    max_us: float = 0.0

    def add(self, dur_us: float) -> None:
        self.count += 1
        self.total_us += dur_us
        self.max_us = max(self.max_us, dur_us)

    def as_dict(self) -> dict[str, float | int]:
        mean_us = self.total_us / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_us": self.total_us,
            "total_ms": self.total_us / 1000.0,
            "mean_us": mean_us,
            "max_us": self.max_us,
        }


def decode_json_string(value: str) -> str:
    return json.loads(f'"{value}"')


def maybe_decode(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    return decode_json_string(match.group(1))


def maybe_json_value_after_key(line: str, key: str) -> Any:
    marker = f'"{key}":'
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    while start < len(line) and line[start].isspace():
        start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(line[start:])
    except json.JSONDecodeError:
        return None
    return value


def compact(value: Any, max_chars: int = 220) -> str:
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def summarize_ops(
    trace_path: Path,
    *,
    ops: set[str],
    category: str = "cpu_op",
    top_limit: int = 40,
) -> dict[str, Any]:
    by_op: defaultdict[str, BucketStats] = defaultdict(BucketStats)
    by_signature: defaultdict[tuple[str, str, str], BucketStats] = defaultdict(
        BucketStats
    )
    by_dims: defaultdict[tuple[str, str], BucketStats] = defaultdict(BucketStats)

    current_name: str | None = None
    current_cat: str | None = None
    current_dur: float | None = None
    current_types: Any = None
    current_dims: Any = None
    collecting_x_event = False
    total_target_events = 0

    def flush() -> None:
        nonlocal current_name, current_cat, current_dur, current_types, current_dims
        nonlocal collecting_x_event, total_target_events
        if (
            current_name is not None
            and current_dur is not None
            and current_name in ops
            and (not category or current_cat == category)
        ):
            types_key = compact(current_types)
            dims_key = compact(current_dims)
            total_target_events += 1
            by_op[current_name].add(current_dur)
            by_signature[(current_name, types_key, dims_key)].add(current_dur)
            by_dims[(current_name, dims_key)].add(current_dur)

        current_name = None
        current_cat = None
        current_dur = None
        current_types = None
        current_dims = None
        collecting_x_event = False

    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            starts_event = bool(PH_X_RE.search(line))
            if starts_event and collecting_x_event:
                flush()
            if starts_event:
                collecting_x_event = True

            if not collecting_x_event:
                continue

            if current_name is None:
                current_name = maybe_decode(NAME_RE.search(line))
            if current_cat is None:
                current_cat = maybe_decode(CAT_RE.search(line))
            if current_dur is None:
                dur_match = DUR_RE.search(line)
                if dur_match is not None:
                    current_dur = float(dur_match.group(1))
            if current_types is None:
                current_types = maybe_json_value_after_key(line, "Input type")
            if current_dims is None:
                current_dims = maybe_json_value_after_key(line, "Input Dims")

        if collecting_x_event:
            flush()

    def sorted_rows(mapping: dict[Any, BucketStats]) -> list[tuple[Any, BucketStats]]:
        return sorted(
            mapping.items(),
            key=lambda item: (item[1].total_us, item[1].count),
            reverse=True,
        )

    return {
        "trace_path": str(trace_path),
        "ops": sorted(ops),
        "category": category,
        "total_target_events": total_target_events,
        "by_op": [
            {"name": name, **stats.as_dict()}
            for name, stats in sorted_rows(by_op)[:top_limit]
        ],
        "by_signature": [
            {
                "name": key[0],
                "input_type": key[1],
                "input_dims": key[2],
                **stats.as_dict(),
            }
            for key, stats in sorted_rows(by_signature)[:top_limit]
        ],
        "by_dims": [
            {"name": key[0], "input_dims": key[1], **stats.as_dict()}
            for key, stats in sorted_rows(by_dims)[:top_limit]
        ],
    }


def fmt_ms(value_us: float) -> str:
    return f"{value_us / 1000.0:.3f}ms"


def markdown_summary(summary: dict[str, Any], *, top_limit: int) -> str:
    lines = [
        "# PyTorch Trace Op Shape Summary",
        "",
        f"- Trace: `{summary['trace_path']}`",
        f"- Category: `{summary['category'] or 'all'}`",
        f"- Target events: `{summary['total_target_events']}`",
        f"- Ops: `{', '.join(summary['ops'])}`",
        "",
        "## By Op",
        "",
        "| name | count | total | mean | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["by_op"][:top_limit]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | "
            f"{fmt_ms(float(row['total_us']))} | "
            f"{fmt_ms(float(row['mean_us']))} | "
            f"{fmt_ms(float(row['max_us']))} |"
        )

    lines.extend(
        [
            "",
            "## By Input Type And Shape",
            "",
            "| name | input type | input dims | count | total | mean | max |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["by_signature"][:top_limit]:
        input_type = str(row["input_type"]).replace("|", "\\|")
        input_dims = str(row["input_dims"]).replace("|", "\\|")
        lines.append(
            f"| `{row['name']}` | `{input_type}` | `{input_dims}` | "
            f"{row['count']} | {fmt_ms(float(row['total_us']))} | "
            f"{fmt_ms(float(row['mean_us']))} | "
            f"{fmt_ms(float(row['max_us']))} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--ops", nargs="*", default=list(DEFAULT_OPS))
    parser.add_argument("--category", default="cpu_op")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = summarize_ops(
        args.trace_path,
        ops=set(args.ops),
        category=args.category,
        top_limit=args.top,
    )
    if args.output_json:
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        args.output_md.write_text(
            markdown_summary(summary, top_limit=args.top),
            encoding="utf-8",
        )
    if not args.output_json and not args.output_md:
        print(markdown_summary(summary, top_limit=args.top))


if __name__ == "__main__":
    main()
