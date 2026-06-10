#!/usr/bin/env python3
"""Summarize selected events from a PyTorch Chrome trace.

The exported ``*.pt.trace.json`` files can be multiple GB.  This script uses a
line-oriented parser for the formatting emitted by PyTorch profiler so the
trace does not need to be loaded as one JSON object.
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


@dataclass
class EventStats:
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


def summarize_trace(
    trace_path: Path,
    *,
    range_prefix: str = "odesign.",
    range_category: str = "user_annotation",
    top_limit: int = 40,
) -> dict[str, Any]:
    ranges: defaultdict[str, EventStats] = defaultdict(EventStats)
    by_name: defaultdict[str, EventStats] = defaultdict(EventStats)
    by_category_name: defaultdict[tuple[str, str], EventStats] = defaultdict(EventStats)
    category_counts: defaultdict[str, int] = defaultdict(int)

    pending_name: str | None = None
    pending_cat: str | None = None
    collecting_x_event = False
    total_x_events = 0
    matched_range_events = 0

    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not collecting_x_event:
                if not PH_X_RE.search(line):
                    continue
                collecting_x_event = True

            if pending_name is None:
                pending_name = maybe_decode(NAME_RE.search(line))
            if pending_cat is None:
                pending_cat = maybe_decode(CAT_RE.search(line))
            dur_match = DUR_RE.search(line)

            if dur_match is None:
                continue

            if pending_name is None:
                continue
            dur_us = float(dur_match.group(1))
            name = pending_name
            cat = pending_cat or ""
            pending_name = None
            pending_cat = None
            collecting_x_event = False

            total_x_events += 1
            category_counts[cat] += 1
            by_name[name].add(dur_us)
            by_category_name[(cat, name)].add(dur_us)
            category_matches = not range_category or cat == range_category
            if name.startswith(range_prefix) and category_matches:
                matched_range_events += 1
                ranges[name].add(dur_us)

    def sorted_items(mapping: dict[Any, EventStats], field: str) -> list[tuple[Any, EventStats]]:
        if field == "count":
            return sorted(
                mapping.items(),
                key=lambda item: (item[1].count, item[1].total_us),
                reverse=True,
            )
        return sorted(
            mapping.items(),
            key=lambda item: (item[1].total_us, item[1].count),
            reverse=True,
        )

    top_by_total = [
        {"name": name, **stats.as_dict()}
        for name, stats in sorted_items(by_name, "total_us")[:top_limit]
    ]
    top_by_count = [
        {"name": name, **stats.as_dict()}
        for name, stats in sorted_items(by_name, "count")[:top_limit]
    ]
    top_by_category_total = [
        {"category": key[0], "name": key[1], **stats.as_dict()}
        for key, stats in sorted_items(by_category_name, "total_us")[:top_limit]
    ]

    return {
        "trace_path": str(trace_path),
        "range_prefix": range_prefix,
        "range_category": range_category,
        "total_x_events_with_duration": total_x_events,
        "matched_range_events": matched_range_events,
        "category_counts": dict(sorted(category_counts.items())),
        "ranges_by_total": [
            {"name": name, **stats.as_dict()}
            for name, stats in sorted_items(ranges, "total_us")
        ],
        "ranges_by_count": [
            {"name": name, **stats.as_dict()}
            for name, stats in sorted_items(ranges, "count")
        ],
        "top_by_total": top_by_total,
        "top_by_count": top_by_count,
        "top_by_category_total": top_by_category_total,
    }


def fmt_ms(value_us: float) -> str:
    return f"{value_us / 1000.0:.3f}ms"


def markdown_table_rows(rows: list[dict[str, Any]], include_category: bool) -> list[str]:
    header = "| category | name | count | total | mean | max |" if include_category else "| name | count | total | mean | max |"
    align = "| --- | --- | ---: | ---: | ---: | ---: |" if include_category else "| --- | ---: | ---: | ---: | ---: |"
    lines = [header, align]
    for row in rows:
        name = str(row["name"]).replace("|", "\\|")
        if len(name) > 140:
            name = name[:137] + "..."
        common = (
            f"{int(row['count'])} | {fmt_ms(float(row['total_us']))} | "
            f"{fmt_ms(float(row['mean_us']))} | {fmt_ms(float(row['max_us']))} |"
        )
        if include_category:
            lines.append(f"| `{row.get('category', '')}` | `{name}` | {common}")
        else:
            lines.append(f"| `{name}` | {common}")
    return lines


def markdown_summary(summary: dict[str, Any], *, top_limit: int) -> str:
    lines = [
        "# PyTorch Trace Summary",
        "",
        f"- Trace: `{summary['trace_path']}`",
        f"- Range prefix: `{summary['range_prefix']}`",
        f"- Range category: `{summary['range_category'] or 'all'}`",
        f"- X events with duration: `{summary['total_x_events_with_duration']}`",
        f"- Matched range events: `{summary['matched_range_events']}`",
        "",
        "## ODesign Ranges By Total CPU Span",
        "",
    ]
    lines.extend(markdown_table_rows(summary["ranges_by_total"][:top_limit], False))
    lines.extend(["", "## ODesign Ranges By Count", ""])
    lines.extend(markdown_table_rows(summary["ranges_by_count"][:top_limit], False))
    lines.extend(["", "## Top Events By Total CPU Span", ""])
    lines.extend(markdown_table_rows(summary["top_by_category_total"][:top_limit], True))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--range-prefix", default="odesign.")
    parser.add_argument(
        "--range-category",
        default="user_annotation",
        help="Category to use for prefixed range tables. Use empty string for all categories.",
    )
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = summarize_trace(
        args.trace_path,
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
