import importlib.util
import sys
import tempfile
from pathlib import Path
import unittest


def load_op_ranges_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "summarize_torch_trace_op_ranges.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_torch_trace_op_ranges", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TorchTraceOpRangesSummaryTests(unittest.TestCase):
    def test_summarize_op_ranges_uses_innermost_range(self) -> None:
        module = load_op_ranges_module()
        trace_text = """
{
  "traceEvents": [
  {
    "ph": "X", "cat": "user_annotation", "name": "odesign.outer",
    "ts": 0, "dur": 100
  },
  {
    "ph": "X", "cat": "user_annotation", "name": "odesign.inner",
    "ts": 10, "dur": 20
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::item",
    "ts": 12, "dur": 5,
    "args": {"Input Dims": [[]]}
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::copy_",
    "ts": 90, "dur": 5,
    "args": {"Input Dims": [[2, 3], [2, 3], []]}
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::item",
    "ts": 200, "dur": 7,
    "args": {"Input Dims": [[]]}
  }
  ]
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(trace_text, encoding="utf-8")
            summary = module.summarize_op_ranges(
                path,
                ops={"aten::item", "aten::copy_"},
                top_limit=10,
            )

        self.assertEqual(summary["range_count"], 2)
        self.assertEqual(summary["op_event_count"], 3)
        by_range_op = {
            (row["range"], row["op"]): row for row in summary["by_range_op"]
        }
        self.assertEqual(by_range_op[("odesign.inner", "aten::item")]["count"], 1)
        self.assertEqual(by_range_op[("odesign.outer", "aten::copy_")]["count"], 1)
        self.assertEqual(
            by_range_op[(module.OUTSIDE_RANGE, "aten::item")]["total_us"], 7.0
        )

    def test_markdown_summary_contains_sections(self) -> None:
        module = load_op_ranges_module()
        summary = {
            "trace_path": "/tmp/trace.json",
            "range_category": "user_annotation",
            "range_prefix": "odesign.",
            "op_category": "cpu_op",
            "range_count": 1,
            "op_event_count": 1,
            "by_range": [
                {
                    "range": "odesign.inner",
                    "count": 1,
                    "total_us": 5.0,
                    "mean_us": 5.0,
                    "max_us": 5.0,
                }
            ],
            "by_range_op": [
                {
                    "range": "odesign.inner",
                    "op": "aten::item",
                    "count": 1,
                    "total_us": 5.0,
                    "mean_us": 5.0,
                    "max_us": 5.0,
                }
            ],
            "by_range_op_dims": [
                {
                    "range": "odesign.inner",
                    "op": "aten::item",
                    "input_dims": "[[]]",
                    "count": 1,
                    "total_us": 5.0,
                    "mean_us": 5.0,
                    "max_us": 5.0,
                }
            ],
        }

        markdown = module.markdown_summary(summary, top_limit=10)

        self.assertIn("By Range And Op", markdown)
        self.assertIn("odesign.inner", markdown)


if __name__ == "__main__":
    unittest.main()
