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
from stable_baselines3.common.vec_env import DummyVecEnv

from slope_core.learner import (
    EvaluationResult,
    TrainingProgressCallback,
    SlopeQRDQN,
    FrameStackExtractor,
    build_qrdqn,
    checkpoint_paths,
    create_or_resume_qrdqn,
    evaluate_deterministic,
    is_better_evaluation,
    load_training_state,
    save_training_state,
)


# TinyEpisodeEnv observations are two history entries of a one-pixel frame
# plus a three-value action.
TINY_FRAME = (1, 1, 1)


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



class RecordingEnv(TinyEpisodeEnv):
    def __init__(self, lengths: list[int]) -> None:
        super().__init__(lengths)
        self.actions: list[int] = []

    def step(self, action: int):
        self.actions.append(int(action))
        return super().step(action)


def run_lengths(values: list) -> list[int]:
    runs = [1]
    for previous, current in zip(values, values[1:]):
        if current == previous:
            runs[-1] += 1
        else:
            runs.append(1)
    return runs


class LearnerTests(unittest.TestCase):
    def test_build_qrdqn_uses_sample_efficient_configuration(self) -> None:
        env = TinyEpisodeEnv([1])
        sentinel = Mock()
        with patch("slope_core.learner.SlopeQRDQN", return_value=sentinel) as qrdqn:
            result = build_qrdqn(
                env, device="cuda", seed=19, exploration_steps=12_345
            )

        self.assertIs(result, sentinel)
        self.assertEqual(qrdqn.call_args.args[0], "MlpPolicy")
        self.assertIs(qrdqn.call_args.args[1], env)
        kwargs = qrdqn.call_args.kwargs
        self.assertEqual(kwargs["buffer_size"], 100_000)
        self.assertEqual(kwargs["learning_starts"], 10_000)
        self.assertEqual(kwargs["batch_size"], 256)
        self.assertAlmostEqual(kwargs["gamma"], 0.997)
        self.assertEqual(kwargs["n_steps"], 5)
        self.assertEqual(kwargs["train_freq"], (4, "step"))
        self.assertEqual(kwargs["gradient_steps"], -1)
        self.assertEqual(kwargs["target_update_interval"], 5_000)
        self.assertEqual(kwargs["exploration_initial_eps"], 1.0)
        self.assertEqual(kwargs["exploration_final_eps"], 0.01)
        self.assertFalse(kwargs["optimize_memory_usage"])
        self.assertEqual(
            kwargs["replay_buffer_kwargs"], {"handle_timeout_termination": True}
        )
        policy_kwargs = kwargs["policy_kwargs"]
        self.assertIs(policy_kwargs["features_extractor_class"], FrameStackExtractor)
        self.assertEqual(
            policy_kwargs["features_extractor_kwargs"], {"frame_shape": (2, 40, 64)}
        )
        self.assertFalse(policy_kwargs["normalize_images"])
        self.assertEqual(policy_kwargs["n_quantiles"], 100)
        self.assertEqual(policy_kwargs["net_arch"], [512])
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["seed"], 19)
        self.assertEqual(sentinel.exploration_steps, 12_345)

    def test_build_qrdqn_constructs_real_model(self) -> None:
        model = build_qrdqn(TinyEpisodeEnv([1]), device="cpu", frame_shape=TINY_FRAME, seed=3)
        try:
            self.assertEqual(model.n_steps, 5)
            self.assertEqual(model.buffer_size, 100_000)
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

    def test_exploration_schedule_survives_save_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = build_qrdqn(
                TinyEpisodeEnv([1]), device="cpu", frame_shape=TINY_FRAME, seed=5, exploration_steps=1_000
            )
            path = Path(directory) / "model.zip"
            try:
                self.assertAlmostEqual(model.exploration_at(0), 1.0)
                self.assertAlmostEqual(model.exploration_at(500), 0.505)
                self.assertAlmostEqual(model.exploration_at(5_000), 0.01)
                model.save(path)
            finally:
                model.get_env().close()
            loaded = SlopeQRDQN.load(
                path, env=TinyEpisodeEnv([1_000]), device="cpu"
            )
            try:
                self.assertEqual(loaded.exploration_steps, 1_000)
                loaded.num_timesteps = 500
                loaded.learn(total_timesteps=10, reset_num_timesteps=False)
                self.assertGreaterEqual(loaded.num_timesteps, 510)
                self.assertAlmostEqual(
                    loaded.exploration_rate,
                    loaded.exploration_at(loaded.num_timesteps),
                )
                self.assertLess(loaded.exploration_rate, loaded.exploration_at(500))
            finally:
                loaded.get_env().close()

    def test_random_exploration_holds_each_action_for_several_steps(self) -> None:
        env = RecordingEnv([10_000])
        model = build_qrdqn(env, device="cpu", frame_shape=TINY_FRAME, seed=11)
        try:
            model.learn(total_timesteps=400)
            runs = run_lengths(env.actions)
            shortest, longest = model.exploration_hold
            # The final run may be cut off by the step budget.
            self.assertGreaterEqual(min(runs[:-1]), shortest)
            self.assertGreater(np.mean(runs[:-1]), shortest)
            self.assertEqual(sorted(set(env.actions)), [0, 1, 2])
        finally:
            model.env.close()

    def test_exploration_hold_is_cleared_at_episode_end(self) -> None:
        model = build_qrdqn(TinyEpisodeEnv([3, 100]), device="cpu", frame_shape=TINY_FRAME, seed=13)
        model.exploration_hold = (50, 50)
        try:
            # One four-step rollout: steps 1-3 use one 50-step hold, which is
            # 47 steps from finishing when the episode ends.  Step 4 must start
            # a fresh hold (49 remaining) instead of continuing the old one (46).
            model.learn(total_timesteps=4)
            self.assertEqual(model.num_timesteps, 4)
            self.assertEqual(model._exploration_holds.tolist(), [49])
        finally:
            model.env.close()

    def test_exploitation_uses_greedy_policy_after_hold_expires(self) -> None:
        model = build_qrdqn(TinyEpisodeEnv([1]), device="cpu", frame_shape=TINY_FRAME, seed=17)
        try:
            model.exploration_rate = 0.0
            model._last_obs = np.zeros((1, 8), dtype=np.uint8)
            model._exploration_holds = np.array([1])
            model._held_actions = np.array([0])
            with patch.object(model.policy, "predict", return_value=(np.array([2]), None)):
                held, _ = model._sample_action(learning_starts=0)
                greedy, stored = model._sample_action(learning_starts=0)
            self.assertEqual(held.tolist(), [0])
            self.assertEqual(greedy.tolist(), [2])
            self.assertEqual(stored.tolist(), [2])
        finally:
            model.env.close()

    def test_frame_stack_extractor_splits_frames_and_actions(self) -> None:
        # Two history entries of a 2x2 two-channel frame and a one-hot action.
        entry = [0, 51, 102, 153, 204, 255, 0, 255] + [0, 255, 0]
        observation_space = spaces.Box(0, 255, shape=(2 * len(entry),), dtype=np.uint8)
        extractor = FrameStackExtractor(
            observation_space, frame_shape=(2, 2, 2), image_features=16
        )
        self.assertEqual(extractor.history, 2)
        self.assertEqual(extractor.features_dim, 16 + 6)
        self.assertEqual(extractor.cnn[0].in_channels, 4)

        features = extractor(torch.tensor([entry * 2], dtype=torch.float32))

        self.assertEqual(tuple(features.shape), (1, 22))
        np.testing.assert_allclose(
            features[0, 16:].detach().numpy(), [0, 1, 0, 0, 1, 0], atol=1e-6
        )

    def test_frame_stack_extractor_rejects_a_mismatched_layout(self) -> None:
        observation_space = spaces.Box(0, 255, shape=(10,), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "whole number"):
            FrameStackExtractor(observation_space, frame_shape=(1, 2, 2))

    def test_frame_stack_extractor_reads_the_default_observation(self) -> None:
        # Default 4-frame observation of 2x40x64 images plus actions.
        size = 4 * (2 * 40 * 64 + 3)
        extractor = FrameStackExtractor(
            spaces.Box(0, 255, shape=(size,), dtype=np.uint8)
        )
        features = extractor(torch.zeros((3, size)))
        self.assertEqual(tuple(features.shape), (3, 512 + 12))

    def test_custom_extractor_survives_model_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = build_qrdqn(TinyEpisodeEnv([1]), device="cpu", frame_shape=TINY_FRAME, seed=5)
            path = Path(directory) / "model.zip"
            try:
                model.save(path)
            finally:
                model.get_env().close()
            loaded = QRDQN.load(path, env=TinyEpisodeEnv([1]), device="cpu")
            try:
                self.assertIsInstance(
                    loaded.policy.quantile_net.features_extractor,
                    FrameStackExtractor,
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
            loaded = Mock(n_envs=1)
            loaded.replay_buffer.n_envs = 1
            with patch("slope_core.learner.SlopeQRDQN.load", return_value=loaded) as load:
                result = load_training_state(
                    TinyEpisodeEnv([1]), root / "agent", device="cpu"
                )

            self.assertIs(result, loaded)
            self.assertEqual(load.call_args.kwargs["device"], "cpu")
            loaded.load_replay_buffer.assert_called_once_with(paths.replay_buffer)

    def test_parallel_envs_keep_one_update_per_transition(self) -> None:
        env = DummyVecEnv([lambda: TinyEpisodeEnv([4] * 100) for _ in range(3)])
        model = build_qrdqn(env, device="cpu", frame_shape=TINY_FRAME, seed=3)
        model.learning_starts = 0
        model.batch_size = 8
        try:
            with patch.object(model, "train") as train:
                model.learn(total_timesteps=24)
            updates = sum(call.kwargs["gradient_steps"] for call in train.call_args_list)
            self.assertEqual(model.num_timesteps, 24)
            self.assertEqual(updates, 24)
        finally:
            env.close()

    def test_resume_rejects_a_different_browser_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent"
            single = build_qrdqn(TinyEpisodeEnv([2] * 50), device="cpu", frame_shape=TINY_FRAME, seed=3)
            single.learn(total_timesteps=10)
            save_training_state(single, path)

            parallel = DummyVecEnv([lambda: TinyEpisodeEnv([2] * 50) for _ in range(3)])
            try:
                with self.assertRaisesRegex(ValueError, "resume with --envs 1"):
                    load_training_state(parallel, path, device="cpu")
                resumed = load_training_state(
                    TinyEpisodeEnv([2] * 50), path, device="cpu"
                )
                self.assertEqual(resumed.replay_buffer.n_envs, 1)
            finally:
                parallel.close()

    def test_fresh_start_does_not_require_checkpoint(self) -> None:
        env = TinyEpisodeEnv([1])
        fresh = Mock()
        with patch("slope_core.learner.build_qrdqn", return_value=fresh) as build:
            result = create_or_resume_qrdqn(
                env, "does-not-exist", resume=False, device="cpu", seed=23
            )
        self.assertIs(result, fresh)
        build.assert_called_once_with(
            env, device="cpu", seed=23, exploration_steps=50_000
        )

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
