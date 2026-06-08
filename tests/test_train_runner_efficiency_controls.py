import unittest

from src.utils.train.train_runner import TrainRunner


class TrainRunnerEfficiencyControlTests(unittest.TestCase):
    def test_empty_cache_policy_defaults_to_microbatch(self) -> None:
        self.assertEqual(TrainRunner._normalize_empty_cache_policy(None), "microbatch")
        self.assertEqual(TrainRunner._normalize_empty_cache_policy(""), "microbatch")

    def test_empty_cache_policy_controls_when_cache_is_cleared(self) -> None:
        self.assertIs(TrainRunner._should_empty_cache_for_policy("microbatch", False), True)
        self.assertIs(TrainRunner._should_empty_cache_for_policy("microbatch", True), True)
        self.assertIs(
            TrainRunner._should_empty_cache_for_policy("optimizer_update", False), False
        )
        self.assertIs(
            TrainRunner._should_empty_cache_for_policy("optimizer_update", True), True
        )
        self.assertIs(TrainRunner._should_empty_cache_for_policy("never", False), False)
        self.assertIs(TrainRunner._should_empty_cache_for_policy("never", True), False)

    def test_empty_cache_policy_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "ODESIGN_EMPTY_CACHE_POLICY"):
            TrainRunner._normalize_empty_cache_policy("sometimes")

if __name__ == "__main__":
    unittest.main()
