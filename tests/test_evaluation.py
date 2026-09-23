import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from slope_core.evaluation import BestModelCallback, best_checkpoint_paths
from slope_core.learner import EvaluationResult


class EvaluationTests(unittest.TestCase):
    def test_best_paths_do_not_replace_latest_checkpoint(self):
        best, metadata = best_checkpoint_paths("runs/slope")
        self.assertEqual(best, Path("runs/slope_best.zip"))
        self.assertEqual(metadata, Path("runs/slope_best.json"))

    def test_callback_saves_and_records_better_deterministic_result(self):
        with TemporaryDirectory() as directory:
            env = Mock()
            callback = BestModelCallback(
                lambda: env,
                Path(directory) / "agent",
                eval_every=100,
                episodes=3,
                control_hz=20,
            )
            model = Mock(num_timesteps=100)
            model.save.side_effect = lambda path: Path(path).write_bytes(b"model")
            callback.model = model
            result = EvaluationResult((40, 60, 100), (-0.6, -0.4, 0.0))
            with patch("slope_core.evaluation.evaluate_deterministic", return_value=result):
                self.assertTrue(callback._on_step())

            self.assertTrue(callback.best_model_path.is_file())
            metadata = json.loads(callback.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["best_timestep"], 100)
            self.assertEqual(metadata["best_median_seconds"], 3.0)
            self.assertEqual(metadata["last_evaluation_timestep"], 100)
            callback._on_training_end()
            env.close.assert_called_once_with()

    def test_callback_does_not_evaluate_before_global_boundary(self):
        callback = BestModelCallback(Mock(), "agent", eval_every=100)
        callback.model = Mock(num_timesteps=99)
        with patch("slope_core.evaluation.evaluate_deterministic") as evaluate:
            self.assertTrue(callback._on_step())
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

