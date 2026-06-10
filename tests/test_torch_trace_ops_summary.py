import importlib.util
import sys
import tempfile
from pathlib import Path
import unittest


def load_ops_summary_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "summarize_torch_trace_ops.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_torch_trace_ops", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TorchTraceOpsSummaryTests(unittest.TestCase):
    def test_summarize_ops_groups_by_input_signature(self) -> None:
        module = load_ops_summary_module()
        trace_text = """
{
  "traceEvents": [
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::_to_copy",
    "ts": 1, "dur": 100,
    "args": {"Input type": ["float", "Scalar"], "Input Dims": [[2, 3], []]}
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::_to_copy",
    "ts": 2, "dur": 300,
    "args": {"Input type": ["float", "Scalar"], "Input Dims": [[2, 3], []]}
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::item",
    "ts": 3, "dur": 50,
    "args": {"Input type": ["float"], "Input Dims": [[]]}
  },
  {
    "ph": "X", "cat": "kernel", "name": "aten::_to_copy",
    "ts": 4, "dur": 999
  }
  ]
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(trace_text, encoding="utf-8")
            summary = module.summarize_ops(
                path,
                ops={"aten::_to_copy", "aten::item"},
                category="cpu_op",
                top_limit=10,
            )

        self.assertEqual(summary["total_target_events"], 3)
        self.assertEqual(summary["by_op"][0]["name"], "aten::_to_copy")
        self.assertEqual(summary["by_op"][0]["count"], 2)
        self.assertEqual(summary["by_op"][0]["total_us"], 400.0)
        top_signature = summary["by_signature"][0]
        self.assertEqual(top_signature["input_type"], '["float","Scalar"]')
        self.assertEqual(top_signature["input_dims"], "[[2,3],[]]")

    def test_markdown_summary_contains_signature_section(self) -> None:
        module = load_ops_summary_module()
        summary = {
            "trace_path": "/tmp/trace.json",
            "category": "cpu_op",
            "total_target_events": 1,
            "ops": ["aten::item"],
            "by_op": [
                {
                    "name": "aten::item",
                    "count": 1,
                    "total_us": 100.0,
                    "mean_us": 100.0,
                    "max_us": 100.0,
                }
            ],
            "by_signature": [
                {
                    "name": "aten::item",
                    "input_type": '["float"]',
                    "input_dims": "[[]]",
                    "count": 1,
                    "total_us": 100.0,
                    "mean_us": 100.0,
                    "max_us": 100.0,
                }
            ],
        }

        markdown = module.markdown_summary(summary, top_limit=10)

        self.assertIn("By Input Type And Shape", markdown)
        self.assertIn("aten::item", markdown)


if __name__ == "__main__":
    unittest.main()
