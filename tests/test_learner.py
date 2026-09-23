from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import QRDQN

from slope_core.learner import (
    EvaluationResult,
    TrainingProgressCallback,
    Uint8VectorExtractor,
    build_qrdqn,
    checkpoint_paths,
    create_or_resume_qrdqn,
    evaluate_deterministic,
    is_better_evaluation,
    load_training_state,
    save_training_state,
)


class TinyEpisodeEnv(gym.Env[np.ndarray, int]):
    def __init__(self, lengths: list[int]) -> None:
        self.observation_space = spaces.Box(0, 255, shape=(8,), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.lengths = lengths
        self.episode_index = -1
        self.step_index = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.episode_index += 1
        self.step_index = 0
        return np.zeros(8, dtype=np.uint8), {}

    def step(self, action: int):
        if not 0 <= action < 3:
            raise AssertionError(f"invalid action {action}")
        self.step_index += 1
        terminated = self.step_index >= self.lengths[self.episode_index]
        return np.zeros(8, dtype=np.uint8), 1.0, terminated, False, {}


class LearnerTests(unittest.TestCase):
    def test_build_qrdqn_uses_sample_efficient_configuration(self) -> None:
        env = TinyEpisodeEnv([1])
        sentinel = object()
        with patch("slope_core.learner.QRDQN", return_value=sentinel) as qrdqn:
            result = build_qrdqn(env, device="cuda", seed=19)

        self.assertIs(result, sentinel)
        self.assertEqual(qrdqn.call_args.args[0], "MlpPolicy")
        self.assertIs(qrdqn.call_args.args[1], env)
        kwargs = qrdqn.call_args.kwargs
        self.assertEqual(kwargs["buffer_size"], 150_000)
        self.assertEqual(kwargs["learning_starts"], 10_000)
        self.assertEqual(kwargs["batch_size"], 256)
        self.assertAlmostEqual(kwargs["gamma"], 0.997)
        self.assertEqual(kwargs["n_steps"], 5)
        self.assertEqual(kwargs["train_freq"], (4, "step"))
        self.assertEqual(kwargs["gradient_steps"], 2)
        self.assertEqual(kwargs["target_update_interval"], 5_000)
        self.assertEqual(kwargs["exploration_initial_eps"], 1.0)
        self.assertEqual(kwargs["exploration_final_eps"], 0.03)
        self.assertEqual(kwargs["exploration_fraction"], 0.30)
        self.assertFalse(kwargs["optimize_memory_usage"])
        self.assertEqual(
            kwargs["replay_buffer_kwargs"], {"handle_timeout_termination": True}
        )
        policy_kwargs = kwargs["policy_kwargs"]
        self.assertIs(policy_kwargs["features_extractor_class"], Uint8VectorExtractor)
        self.assertFalse(policy_kwargs["normalize_images"])
        self.assertEqual(policy_kwargs["n_quantiles"], 100)
        self.assertEqual(policy_kwargs["net_arch"], [512, 512])
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["seed"], 19)

    def test_build_qrdqn_constructs_real_model(self) -> None:
        model = build_qrdqn(TinyEpisodeEnv([1]), device="cpu", seed=3)
        try:
            self.assertEqual(model.n_steps, 5)
            self.assertEqual(model.buffer_size, 150_000)
            self.assertEqual(model.policy.n_quantiles, 100)
            self.assertIsNotNone(model.replay_buffer)
            self.assertTrue(model.replay_buffer.handle_timeout_termination)
            self.assertEqual(model.replay_buffer.observations.dtype, np.uint8)
            action, _ = model.predict(
                np.full(8, 127, dtype=np.uint8), deterministic=True
            )
            self.assertIn(int(np.asarray(action).reshape(-1)[0]), (0, 1, 2))
        finally:
            model.get_env().close()

    def test_uint8_extractor_dequantizes_to_unit_interval(self) -> None:
        observation_space = spaces.Box(0, 255, shape=(4,), dtype=np.uint8)
        extractor = Uint8VectorExtractor(observation_space)
        encoded = torch.tensor([[0, 64, 128, 255]], dtype=torch.float32)
        decoded = extractor(encoded).detach().numpy()
        np.testing.assert_allclose(
            decoded,
            np.asarray([[0.0, 64 / 255, 128 / 255, 1.0]], dtype=np.float32),
        )

    def test_custom_extractor_survives_model_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = build_qrdqn(TinyEpisodeEnv([1]), device="cpu", seed=5)
            path = Path(directory) / "model.zip"
            try:
                model.save(path)
            finally:
                model.get_env().close()
            loaded = QRDQN.load(path, env=TinyEpisodeEnv([1]), device="cpu")
            try:
                self.assertIsInstance(
                    loaded.policy.quantile_net.features_extractor,
                    Uint8VectorExtractor,
                )
                action, _ = loaded.predict(
                    np.full(8, 255, dtype=np.uint8), deterministic=True
                )
                self.assertIn(int(np.asarray(action).reshape(-1)[0]), (0, 1, 2))
            finally:
                loaded.get_env().close()

    def test_checkpoint_paths_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for requested in ("agent", "agent.zip"):
                with self.subTest(requested=requested):
                    paths = checkpoint_paths(root / requested)
                    self.assertEqual(paths.model, root / "agent.zip")
                    self.assertEqual(paths.replay_buffer, root / "agent.replay.pkl")

    def test_save_training_state_writes_both_halves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Mock()
            model.save.side_effect = lambda path: Path(path).write_bytes(b"model")
            model.save_replay_buffer.side_effect = lambda path: Path(path).write_bytes(
                b"replay"
            )

            paths = save_training_state(model, root / "nested" / "agent")

            self.assertEqual(paths.model.read_bytes(), b"model")
            self.assertEqual(paths.replay_buffer.read_bytes(), b"replay")
            self.assertEqual(list((root / "nested").glob(".*.tmp.*")), [])

    def test_resume_requires_model_and_replay(self) -> None:
        for present in ((), ("model",), ("replay",)):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = checkpoint_paths(root / "agent")
                if "model" in present:
                    paths.model.write_bytes(b"model")
                if "replay" in present:
                    paths.replay_buffer.write_bytes(b"replay")

                with self.assertRaisesRegex(
                    FileNotFoundError, "both the model and replay buffer"
                ):
                    load_training_state(TinyEpisodeEnv([1]), root / "agent")

    def test_load_training_state_restores_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = checkpoint_paths(root / "agent")
            paths.model.write_bytes(b"model")
            paths.replay_buffer.write_bytes(b"replay")
            loaded = Mock()
            with patch("slope_core.learner.QRDQN.load", return_value=loaded) as load:
                result = load_training_state(
                    TinyEpisodeEnv([1]), root / "agent", device="cpu"
                )

            self.assertIs(result, loaded)
            self.assertEqual(load.call_args.kwargs["device"], "cpu")
            loaded.load_replay_buffer.assert_called_once_with(paths.replay_buffer)

    def test_fresh_start_does_not_require_checkpoint(self) -> None:
        env = TinyEpisodeEnv([1])
        fresh = Mock()
        with patch("slope_core.learner.build_qrdqn", return_value=fresh) as build:
            result = create_or_resume_qrdqn(
                env, "does-not-exist", resume=False, device="cpu", seed=23
            )
        self.assertIs(result, fresh)
        build.assert_called_once_with(env, device="cpu", seed=23)

    def test_deterministic_evaluation_returns_lengths_rewards_and_rank(self) -> None:
        env = TinyEpisodeEnv([3, 5, 4, 8])
        model = Mock()
        model.predict.return_value = (np.asarray(1), None)

        result = evaluate_deterministic(model, env, episodes=4, seed=100)

        self.assertEqual(result.episode_lengths, (3, 5, 4, 8))
        self.assertEqual(result.episode_rewards, (3.0, 5.0, 4.0, 8.0))
        self.assertEqual(result.median_length, 4.5)
        self.assertAlmostEqual(result.p25_length, 3.75)
        self.assertTrue(
            all(call.kwargs["deterministic"] is True for call in model.predict.call_args_list)
        )

    def test_evaluation_ranking_uses_median_then_p25(self) -> None:
        safer = EvaluationResult((4, 6, 6, 8), (0.0,) * 4)
        brittle = EvaluationResult((1, 6, 6, 9), (100.0,) * 4)
        self.assertEqual(safer.median_length, brittle.median_length)
        self.assertGreater(safer.p25_length, brittle.p25_length)
        self.assertTrue(is_better_evaluation(safer, brittle))
        self.assertFalse(is_better_evaluation(brittle, safer))
        self.assertTrue(is_better_evaluation(brittle, None))

    def test_progress_callback_records_episode_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest"
            callback = TrainingProgressCallback(
                checkpoint,
                control_hz=20,
                report_every=2,
                save_every=2,
                save_on_end=False,
            )
            model = Mock()
            model.num_timesteps = 0
            callback.init_callback(model)
            callback.on_training_start({}, {})
            model.num_timesteps = 2
            callback.update_locals(
                {
                    "rewards": np.asarray([0.7]),
                    "dones": np.asarray([True]),
                    "infos": [{"episode": {"l": 40, "r": 1.2}}],
                }
            )
            with (
                patch("slope_core.learner.save_training_state") as save,
                patch("builtins.print") as printer,
            ):
                self.assertTrue(callback.on_step())

            self.assertEqual(tuple(callback.recent_lengths), (40,))
            self.assertEqual(tuple(callback.recent_survival_seconds), (2.0,))
            self.assertEqual(tuple(callback.recent_rewards), (1.2,))
            save.assert_called_once_with(model, checkpoint)
            output = "\n".join(str(call.args[0]) for call in printer.call_args_list)
            self.assertIn("median_survival=2.00s", output)
            self.assertIn("throughput=", output)


if __name__ == "__main__":
    unittest.main()
