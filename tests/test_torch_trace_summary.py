import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest


def load_trace_summary_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_torch_trace.py"
    spec = importlib.util.spec_from_file_location("summarize_torch_trace", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TorchTraceSummaryTests(unittest.TestCase):
    def test_summarize_trace_counts_prefixed_ranges_and_durations(self) -> None:
        module = load_trace_summary_module()
        trace = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "odesign.pairformer_block.tri_mul_out",
                    "ts": 1,
                    "dur": 100,
                },
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "odesign.pairformer_block.tri_mul_out",
                    "ts": 2,
                    "dur": 300,
                },
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "sm90_xmma_gemm",
                    "ts": 3,
                    "dur": 50,
                },
                {
                    "ph": "f",
                    "cat": "fwdbwd",
                    "name": "odesign.ignored_flow_event",
                    "ts": 4,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

            summary = module.summarize_trace(path, range_prefix="odesign.", top_limit=10)

        self.assertEqual(summary["total_x_events_with_duration"], 3)
        self.assertEqual(summary["matched_range_events"], 2)
        self.assertEqual(summary["category_counts"]["user_annotation"], 2)
        self.assertEqual(summary["category_counts"]["kernel"], 1)
        first_range = summary["ranges_by_total"][0]
        self.assertEqual(first_range["name"], "odesign.pairformer_block.tri_mul_out")
        self.assertEqual(first_range["count"], 2)
        self.assertEqual(first_range["total_us"], 400.0)
        self.assertEqual(first_range["mean_us"], 200.0)

    def test_markdown_summary_contains_range_sections(self) -> None:
        module = load_trace_summary_module()
        summary = {
            "trace_path": "/tmp/trace.json",
            "range_prefix": "odesign.",
            "range_category": "user_annotation",
            "total_x_events_with_duration": 1,
            "matched_range_events": 1,
            "ranges_by_total": [
                {
                    "name": "odesign.transition.projections",
                    "count": 1,
                    "total_us": 1000.0,
                    "mean_us": 1000.0,
                    "max_us": 1000.0,
                }
            ],
            "ranges_by_count": [
                {
                    "name": "odesign.transition.projections",
                    "count": 1,
                    "total_us": 1000.0,
                    "mean_us": 1000.0,
                    "max_us": 1000.0,
                }
            ],
            "top_by_category_total": [
                {
                    "category": "cpu_op",
                    "name": "odesign.transition.projections",
                    "count": 1,
                    "total_us": 1000.0,
                    "mean_us": 1000.0,
                    "max_us": 1000.0,
                }
            ],
        }

        markdown = module.markdown_summary(summary, top_limit=5)

        self.assertIn("ODesign Ranges By Total CPU Span", markdown)
        self.assertIn("odesign.transition.projections", markdown)


if __name__ == "__main__":
    unittest.main()
