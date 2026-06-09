import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


def load_summary_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_module_profile.py"
    spec = importlib.util.spec_from_file_location("summarize_module_profile", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ModuleProfileSummaryTests(unittest.TestCase):
    def test_build_summary_combines_drop_first_rows(self) -> None:
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            record_dir = Path(tmpdir)
            rows_by_rank = {
                "profile_steps_rank00.jsonl": [
                    {"forward_sec": 100.0, "module_profile_pairformer_sec": 90.0},
                    {
                        "forward_sec": 20.0,
                        "backward_sec": 30.0,
                        "module_profile_pairformer_sec": 12.0,
                        "module_profile_diffusion_sec": 3.0,
                        "module_profile_backward_pairformer_sec": 8.0,
                        "module_profile_backward_msa_sec": 2.0,
                    },
                ],
                "profile_steps_rank01.jsonl": [
                    {"forward_sec": 200.0, "module_profile_pairformer_sec": 180.0},
                    {
                        "forward_sec": 24.0,
                        "backward_sec": 32.0,
                        "module_profile_pairformer_sec": 14.0,
                        "module_profile_diffusion_sec": 5.0,
                        "module_profile_backward_pairformer_sec": 10.0,
                        "module_profile_backward_msa_sec": 4.0,
                    },
                ],
            }
            for name, rows in rows_by_rank.items():
                with (record_dir / name).open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            summary = module.build_summary(record_dir)
            combined = summary["combined"]["drop_first_row"]

        self.assertEqual(combined["row_count"], 2)
        self.assertEqual(combined["module_profile_pairformer_sec"]["mean"], 13.0)
        self.assertEqual(combined["module_profile_diffusion_sec"]["mean"], 4.0)
        self.assertEqual(combined["module_profile_backward_pairformer_sec"]["mean"], 9.0)
        self.assertEqual(combined["module_profile_backward_msa_sec"]["mean"], 3.0)
        self.assertEqual(combined["forward_sec"]["mean"], 22.0)
        self.assertEqual(combined["backward_sec"]["mean"], 31.0)

    def test_markdown_table_includes_forward_and_backward_sections(self) -> None:
        module = load_summary_module()
        summary = {
            "combined": {
                "drop_first_row": {
                    "row_count": 2,
                    "forward_sec": {"mean": 22.0},
                    "backward_sec": {"mean": 31.0},
                    "module_profile_pairformer_sec": {"mean": 13.0},
                    "module_profile_backward_pairformer_sec": {"mean": 9.0},
                }
            }
        }

        markdown = module.markdown_table(summary)

        self.assertIn("Forward Modules", markdown)
        self.assertIn("Backward Hook Modules", markdown)
        self.assertIn("module_profile_pairformer_sec", markdown)
        self.assertIn("module_profile_backward_pairformer_sec", markdown)


if __name__ == "__main__":
    unittest.main()
