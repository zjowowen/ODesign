import unittest
from unittest import mock

from src.model.odesign import ODesign


class ODesignModuleProfilingTests(unittest.TestCase):
    def test_module_profile_stage_records_elapsed_time(self) -> None:
        model = object.__new__(ODesign)
        model._odesign_profile_enabled = True
        model._odesign_profile_record = {}

        with mock.patch.object(ODesign, "_profile_now", side_effect=[10.0, 12.5]):
            start = ODesign._profile_stage_start(model)
            ODesign._profile_stage_end(model, "pairformer", start)

        self.assertEqual(model._odesign_profile_record["module_profile_pairformer_sec"], 2.5)

    def test_module_profile_stage_is_noop_without_record(self) -> None:
        model = object.__new__(ODesign)
        model._odesign_profile_enabled = True
        model._odesign_profile_record = None

        with mock.patch.object(ODesign, "_profile_now") as profile_now:
            self.assertIsNone(ODesign._profile_stage_start(model))
            ODesign._profile_stage_end(model, "pairformer", None)

        profile_now.assert_not_called()


if __name__ == "__main__":
    unittest.main()
