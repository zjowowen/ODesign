#!/usr/bin/env python3
"""Attribute selected PyTorch ops to enclosing ODesign trace ranges."""

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
TS_RE = re.compile(r'"ts":\s*([0-9]+(?:\.[0-9]+)?)')
PH_X_RE = re.compile(r'"ph":\s*"X"')

DEFAULT_OPS = (
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::is_nonzero",
)
OUTSIDE_RANGE = "<outside_odesign_range>"


@dataclass
class TraceEvent:
    name: str
    category: str
    ts: float
    dur: float
    input_type: Any = None
    input_dims: Any = None

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur


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


def maybe_number(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return float(match.group(1))


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


def parse_relevant_events(
    trace_path: Path,
    *,
    ops: set[str],
    op_category: str,
    range_prefix: str,
    range_category: str,
) -> tuple[list[TraceEvent], list[TraceEvent]]:
    ranges: list[TraceEvent] = []
    op_events: list[TraceEvent] = []

    current_name: str | None = None
    current_cat: str | None = None
    current_ts: float | None = None
    current_dur: float | None = None
    current_types: Any = None
    current_dims: Any = None
    collecting_x_event = False

    def flush() -> None:
        nonlocal current_name, current_cat, current_ts, current_dur
        nonlocal current_types, current_dims, collecting_x_event
        if current_name is None or current_cat is None:
            pass
        elif current_ts is None or current_dur is None:
            pass
        elif current_cat == range_category and current_name.startswith(range_prefix):
            ranges.append(
                TraceEvent(
                    current_name,
                    current_cat,
                    current_ts,
                    current_dur,
                    current_types,
                    current_dims,
                )
            )
        elif current_cat == op_category and current_name in ops:
            op_events.append(
                TraceEvent(
                    current_name,
                    current_cat,
                    current_ts,
                    current_dur,
                    current_types,
                    current_dims,
                )
            )

        current_name = None
        current_cat = None
        current_ts = None
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
            if current_ts is None:
                current_ts = maybe_number(TS_RE.search(line))
            if current_dur is None:
                current_dur = maybe_number(DUR_RE.search(line))
            if current_types is None:
                current_types = maybe_json_value_after_key(line, "Input type")
            if current_dims is None:
                current_dims = maybe_json_value_after_key(line, "Input Dims")

        if collecting_x_event:
            flush()

    return ranges, op_events


def summarize_op_ranges(
    trace_path: Path,
    *,
    ops: set[str],
    op_category: str = "cpu_op",
    range_prefix: str = "odesign.",
    range_category: str = "user_annotation",
    top_limit: int = 60,
) -> dict[str, Any]:
    ranges, op_events = parse_relevant_events(
        trace_path,
        ops=ops,
        op_category=op_category,
        range_prefix=range_prefix,
        range_category=range_category,
    )
    ranges.sort(key=lambda event: event.ts)
    op_events.sort(key=lambda event: event.ts)

    by_range_op: defaultdict[tuple[str, str], BucketStats] = defaultdict(BucketStats)
    by_range_op_dims: defaultdict[tuple[str, str, str], BucketStats] = defaultdict(
        BucketStats
    )
    by_range: defaultdict[str, BucketStats] = defaultdict(BucketStats)

    active: list[TraceEvent] = []
    range_idx = 0
    for op_event in op_events:
        while range_idx < len(ranges) and ranges[range_idx].ts <= op_event.ts:
            active.append(ranges[range_idx])
            range_idx += 1
        active = [event for event in active if event.end_ts >= op_event.ts]
        enclosing = [
            event
            for event in active
            if event.ts <= op_event.ts and event.end_ts >= op_event.end_ts
        ]
        if enclosing:
            range_name = min(enclosing, key=lambda event: event.dur).name
        else:
            range_name = OUTSIDE_RANGE

        by_range_op[(range_name, op_event.name)].add(op_event.dur)
        by_range_op_dims[
            (range_name, op_event.name, compact(op_event.input_dims))
        ].add(op_event.dur)
        by_range[range_name].add(op_event.dur)

    def sorted_rows(mapping: dict[Any, BucketStats]) -> list[tuple[Any, BucketStats]]:
        return sorted(
            mapping.items(),
            key=lambda item: (item[1].total_us, item[1].count),
            reverse=True,
        )

    return {
        "trace_path": str(trace_path),
        "ops": sorted(ops),
        "op_category": op_category,
        "range_prefix": range_prefix,
        "range_category": range_category,
        "range_count": len(ranges),
        "op_event_count": len(op_events),
        "by_range": [
            {"range": key, **stats.as_dict()}
            for key, stats in sorted_rows(by_range)[:top_limit]
        ],
        "by_range_op": [
            {"range": key[0], "op": key[1], **stats.as_dict()}
            for key, stats in sorted_rows(by_range_op)[:top_limit]
        ],
        "by_range_op_dims": [
            {
                "range": key[0],
                "op": key[1],
                "input_dims": key[2],
                **stats.as_dict(),
            }
            for key, stats in sorted_rows(by_range_op_dims)[:top_limit]
        ],
    }


def fmt_ms(value_us: float) -> str:
    return f"{value_us / 1000.0:.3f}ms"


def markdown_rows(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" if field in {"range", "op", "input_dims"} else "---:" for field in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if field in {"range", "op", "input_dims"}:
                text = str(value).replace("|", "\\|")
                if len(text) > 160:
                    text = text[:157] + "..."
                values.append(f"`{text}`")
            elif field in {"total_us", "mean_us", "max_us"}:
                values.append(fmt_ms(float(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def markdown_summary(summary: dict[str, Any], *, top_limit: int) -> str:
    lines = [
        "# PyTorch Trace Op Range Summary",
        "",
        f"- Trace: `{summary['trace_path']}`",
        f"- Range category/prefix: `{summary['range_category']}` / `{summary['range_prefix']}`",
        f"- Op category: `{summary['op_category']}`",
        f"- Range events: `{summary['range_count']}`",
        f"- Target op events: `{summary['op_event_count']}`",
        "",
        "## By Range",
        "",
    ]
    lines.extend(
        markdown_rows(
            summary["by_range"][:top_limit],
            ["range", "count", "total_us", "mean_us", "max_us"],
        )
    )
    lines.extend(["", "## By Range And Op", ""])
    lines.extend(
        markdown_rows(
            summary["by_range_op"][:top_limit],
            ["range", "op", "count", "total_us", "mean_us", "max_us"],
        )
    )
    lines.extend(["", "## By Range, Op, And Input Dims", ""])
    lines.extend(
        markdown_rows(
            summary["by_range_op_dims"][:top_limit],
            ["range", "op", "input_dims", "count", "total_us", "mean_us", "max_us"],
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--ops", nargs="*", default=list(DEFAULT_OPS))
    parser.add_argument("--op-category", default="cpu_op")
    parser.add_argument("--range-prefix", default="odesign.")
    parser.add_argument("--range-category", default="user_annotation")
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = summarize_op_ranges(
        args.trace_path,
        ops=set(args.ops),
        op_category=args.op_category,
        range_prefix=args.range_prefix,
        range_category=args.range_category,
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
