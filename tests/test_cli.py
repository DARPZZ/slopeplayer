from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slope_core.cli import parse_args, prepare_training_artifacts
from slope_core.evaluation import best_checkpoint_paths
from slope_core.learner import checkpoint_paths


class CliTests(unittest.TestCase):
    def test_training_supports_explicit_headless_and_headed_modes(self) -> None:
        self.assertFalse(parse_args(["train"]).headed)
        self.assertFalse(parse_args(["train", "--headless"]).headed)
        self.assertTrue(parse_args(["train", "--headed"]).headed)
        with self.assertRaises(SystemExit):
            parse_args(["train", "--headless", "--headed"])

    def test_training_evaluates_ten_episodes_by_default(self) -> None:
        args = parse_args(["train"])
        self.assertEqual(args.eval_episodes, 10)

    def test_training_uses_one_browser_by_default(self) -> None:
        self.assertEqual(parse_args(["train"]).envs, 1)
        self.assertEqual(parse_args(["train", "--envs", "3"]).envs, 3)

    def test_browser_count_is_bounded(self) -> None:
        for count in ("0", "9"):
            with self.subTest(count=count), self.assertRaises(SystemExit):
                parse_args(["train", "--envs", count])

    def test_parallel_browsers_cannot_share_one_cdp_chrome(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["train", "--envs", "2", "--cdp-url", "http://localhost:9222"])

    def test_fresh_run_rejects_any_latest_or_best_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "agent"
            latest = checkpoint_paths(run)
            best_model, best_metadata = best_checkpoint_paths(run)
            for artifact in (
                latest.model,
                latest.replay_buffer,
                best_model,
                best_metadata,
            ):
                with self.subTest(artifact=artifact.name):
                    for candidate in (
                        latest.model,
                        latest.replay_buffer,
                        best_model,
                        best_metadata,
                    ):
                        candidate.unlink(missing_ok=True)
                    artifact.write_bytes(b"old")
                    with self.assertRaisesRegex(
                        SystemExit, "Fresh training would overwrite"
                    ):
                        prepare_training_artifacts(
                            run, resume=False, overwrite=False
                        )

    def test_overwrite_removes_latest_and_best_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "agent"
            latest = checkpoint_paths(run)
            best_model, best_metadata = best_checkpoint_paths(run)
            artifacts = (
                latest.model,
                latest.replay_buffer,
                best_model,
                best_metadata,
            )
            for artifact in artifacts:
                artifact.write_bytes(b"old")

            returned = prepare_training_artifacts(
                run, resume=False, overwrite=True
            )

            self.assertEqual(returned, latest)
            self.assertTrue(all(not artifact.exists() for artifact in artifacts))

    def test_resume_does_not_delete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "agent"
            latest = checkpoint_paths(run)
            latest.model.write_bytes(b"model")
            latest.replay_buffer.write_bytes(b"replay")
            prepare_training_artifacts(run, resume=True, overwrite=False)
            self.assertTrue(latest.model.exists())
            self.assertTrue(latest.replay_buffer.exists())

    def test_resume_and_overwrite_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(SystemExit, "mutually exclusive"):
            prepare_training_artifacts("agent", resume=True, overwrite=True)


if __name__ == "__main__":
    unittest.main()
