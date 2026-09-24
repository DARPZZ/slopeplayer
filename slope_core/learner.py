"""QR-DQN construction, persistence, progress reporting, and evaluation.

This module deliberately knows nothing about the browser environment. The
training environment only needs a three-action Discrete action space and a
byte-packed flat Box observation made of history entries, each one frame image
followed by the one-hot action that produced it.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .vision import FRAME_SHAPE

# NNPACK fails to initialize on CPUs it does not support (common in containers)
# and warns on every convolution before falling back to the default kernel.
torch.backends.nnpack.set_flags(False)


class FrameStackExtractor(BaseFeaturesExtractor):
    """CNN over the stacked frame images, joined with the recent actions.

    The flat uint8 observation holds ``history`` entries, each a
    ``frame_shape`` image followed by a one-hot action.  Keeping it as bytes
    cuts replay memory by four; conversion to [0, 1] floats happens only for
    sampled batches on the model's device.  The frames are stacked as channels
    so the first layer sees motion across the whole history.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        action_dim: int = 3,
        image_features: int = 512,
    ) -> None:
        if len(observation_space.shape) != 1:
            raise ValueError("FrameStackExtractor requires a flat observation")
        if observation_space.dtype != np.uint8:
            raise ValueError("FrameStackExtractor requires uint8 observations")
        channels, height, width = frame_shape
        frame_size = channels * height * width
        entry_size = frame_size + action_dim
        total = int(observation_space.shape[0])
        if total % entry_size:
            raise ValueError(
                f"observation length {total} is not a whole number of "
                f"{frame_shape} frames with {action_dim} action values"
            )
        history = total // entry_size
        super().__init__(observation_space, features_dim=image_features + history * action_dim)
        self.history = history
        self.frame_shape = (channels, height, width)
        self.frame_size = frame_size
        self.entry_size = entry_size
        self.cnn = nn.Sequential(
            nn.Conv2d(history * channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, history * channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(flat, image_features), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        count = observations.shape[0]
        entries = observations.reshape(count, self.history, self.entry_size).float() / 255.0
        channels, height, width = self.frame_shape
        frames = entries[:, :, : self.frame_size].reshape(
            count, self.history * channels, height, width
        )
        actions = entries[:, :, self.frame_size :].reshape(count, -1)
        return torch.cat((self.linear(self.cnn(frames)), actions), dim=1)


DEFAULT_EXPLORATION_STEPS = 50_000
DEFAULT_EXPLORATION_HOLD = (2, 8)


class SlopeQRDQN(QRDQN):
    """QR-DQN with resume-safe, temporally extended epsilon-greedy exploration.

    SB3 anneals epsilon over a fraction of the ``total_timesteps`` given to
    each ``learn()`` call, and a resume adds that request to the steps already
    taken.  Every resume therefore stretches the schedule and raises epsilon
    again.  ``exploration_steps`` is saved with the model, so the schedule set
    at the start of a run survives any number of resumes.

    A random steering action lasting one control step barely moves the ball,
    and independent random steps average out to driving straight.  Each random
    action is therefore held for a uniformly drawn ``exploration_hold`` number
    of steps (the "ez-greedy" scheme of Dabney et al., 2021), which explores
    genuinely different lines through turns.
    """

    exploration_steps: int = DEFAULT_EXPLORATION_STEPS
    exploration_hold: tuple[int, int] = DEFAULT_EXPLORATION_HOLD

    def exploration_at(self, timesteps: int) -> float:
        progress = min(1.0, timesteps / max(1, self.exploration_steps))
        start, end = self.exploration_initial_eps, self.exploration_final_eps
        return start + progress * (end - start)

    def _on_step(self) -> None:
        super()._on_step()
        self.exploration_rate = self.exploration_at(self.num_timesteps)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Any = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        holds = getattr(self, "_exploration_holds", None)
        if holds is None or len(holds) != n_envs:
            self._exploration_holds = np.zeros(n_envs, dtype=np.int64)
            self._held_actions = np.zeros(n_envs, dtype=np.int64)
        warmup = self.num_timesteps < learning_starts
        shortest, longest = self.exploration_hold
        actions = np.empty(n_envs, dtype=np.int64)
        greedy: np.ndarray | None = None
        for index in range(n_envs):
            if self._exploration_holds[index] > 0:
                self._exploration_holds[index] -= 1
            elif warmup or np.random.rand() < self.exploration_rate:
                self._held_actions[index] = int(self.action_space.sample())
                self._exploration_holds[index] = np.random.randint(shortest, longest + 1) - 1
            else:
                if greedy is None:
                    assert self._last_obs is not None, "self._last_obs was not set"
                    predicted, _ = self.policy.predict(self._last_obs, deterministic=True)
                    greedy = np.asarray(predicted).reshape(-1)
                actions[index] = greedy[index]
                continue
            actions[index] = self._held_actions[index]
        return actions, actions

    def _store_transition(
        self,
        replay_buffer: Any,
        buffer_action: np.ndarray,
        new_obs: Any,
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        holds = getattr(self, "_exploration_holds", None)
        if holds is not None:
            # A random hold never carries into the next episode.
            holds[np.asarray(dones, dtype=bool).reshape(-1)] = 0

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_exploration_holds", "_held_actions"]


@dataclass(frozen=True)
class CheckpointPaths:
    """Files that together make one resumable off-policy checkpoint."""

    model: Path
    replay_buffer: Path


def checkpoint_paths(path: str | Path) -> CheckpointPaths:
    """Return predictable model and replay-buffer paths for ``path``.

    ``models/slope`` and ``models/slope.zip`` both resolve to
    ``models/slope.zip`` plus ``models/slope.replay.pkl``.
    """

    requested = Path(path)
    model_path = requested if requested.suffix.lower() == ".zip" else Path(f"{requested}.zip")
    replay_path = model_path.with_name(f"{model_path.name[:-4]}.replay.pkl")
    return CheckpointPaths(model=model_path, replay_buffer=replay_path)


def build_qrdqn(
    env: Any,
    device: str = "auto",
    seed: int = 7,
    exploration_steps: int = DEFAULT_EXPLORATION_STEPS,
    frame_shape: tuple[int, int, int] = FRAME_SHAPE,
) -> SlopeQRDQN:
    """Build the sample-efficient default learner for Slope."""

    if exploration_steps < 1:
        raise ValueError("exploration_steps must be positive")
    model = SlopeQRDQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        # Each 4-frame observation is ~20 KB and the replay stores it twice
        # (current and next), so 100k transitions use ~4.1 GB of host RAM.
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.997,
        train_freq=(4, "step"),
        # One update per collected transition, whatever the number of
        # browsers (-1 means train_freq * n_envs).  Browser steps are
        # expensive and the game clock is paused during updates, so extra
        # replay only costs wall time.
        gradient_steps=-1,
        n_steps=5,
        target_update_interval=5_000,
        exploration_initial_eps=1.0,
        # Held random actions last five steps on average, so 1% epsilon still
        # leaves about 5% of late-training steps exploratory.
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        replay_buffer_kwargs={"handle_timeout_termination": True},
        max_grad_norm=10.0,
        policy_kwargs={
            "features_extractor_class": FrameStackExtractor,
            "features_extractor_kwargs": {"frame_shape": tuple(frame_shape)},
            "normalize_images": False,
            "n_quantiles": 100,
            "net_arch": [512],
        },
        verbose=1,
        device=device,
        seed=seed,
    )
    model.exploration_steps = int(exploration_steps)
    return model


def save_training_state(model: Any, path: str | Path) -> CheckpointPaths:
    """Save the model and replay buffer needed for an exact training resume.

    Both objects are first written to temporary sibling files.  This prevents
    an interrupted serialization from replacing either last usable file.
    """

    paths = checkpoint_paths(path)
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_model = paths.model.with_name(f".{paths.model.stem}.{token}.tmp.zip")
    temporary_replay = paths.replay_buffer.with_name(
        f".{paths.replay_buffer.stem}.{token}.tmp.pkl"
    )
    try:
        model.save(temporary_model)
        model.save_replay_buffer(temporary_replay)
        os.replace(temporary_replay, paths.replay_buffer)
        os.replace(temporary_model, paths.model)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_replay.unlink(missing_ok=True)
    return paths


def load_training_state(
    env: Any,
    path: str | Path,
    *,
    device: str = "auto",
) -> QRDQN:
    """Load a QR-DQN model and its matching replay buffer.

    A model alone can play, but it cannot faithfully resume off-policy
    training.  Missing either half is therefore an explicit error.
    """

    paths = checkpoint_paths(path)
    missing = [
        candidate
        for candidate in (paths.model, paths.replay_buffer)
        if not candidate.is_file()
    ]
    if missing:
        listed = ", ".join(str(candidate) for candidate in missing)
        raise FileNotFoundError(
            "Cannot resume training without both the model and replay buffer; "
            f"missing: {listed}"
        )
    try:
        model = SlopeQRDQN.load(paths.model, env=env, device=device)
    except ValueError as error:
        # A checkpoint from an older observation layout cannot be resumed.
        if "spaces do not match" not in str(error):
            raise
        raise ValueError(
            f"{paths.model} was trained with a different observation or action "
            f"layout and cannot be resumed ({error}). Start a new run with "
            "another --model path, or pass --overwrite to replace it."
        ) from error
    model.load_replay_buffer(paths.replay_buffer)
    # The replay keeps one transition sequence per browser, so it can only
    # continue with the browser count it was recorded with.
    recorded = model.replay_buffer.n_envs
    if recorded != model.n_envs:
        raise ValueError(
            f"{paths.replay_buffer} was recorded with {recorded} browser(s); "
            f"resume with --envs {recorded} or start a new --model with "
            f"--envs {model.n_envs}"
        )
    return model


def create_or_resume_qrdqn(
    env: Any,
    path: str | Path,
    *,
    resume: bool,
    device: str = "auto",
    seed: int = 7,
    exploration_steps: int = DEFAULT_EXPLORATION_STEPS,
) -> SlopeQRDQN:
    """Create a fresh learner or require and restore a complete checkpoint.

    ``exploration_steps`` applies only to a fresh learner; a resumed one keeps
    the schedule stored in its checkpoint.
    """

    if resume:
        return load_training_state(env, path, device=device)
    return build_qrdqn(
        env, device=device, seed=seed, exploration_steps=exploration_steps
    )


class TrainingProgressCallback(BaseCallback):
    """Report completed-run survival and periodically save resumable state."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        control_hz: float = 20.0,
        report_every: int = 2_000,
        save_every: int = 25_000,
        window: int = 100,
        save_on_end: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if report_every < 0 or save_every < 0:
            raise ValueError("report_every and save_every cannot be negative")
        if window < 1:
            raise ValueError("window must be at least one")

        self.checkpoint = Path(checkpoint)
        self.control_hz = float(control_hz)
        self.report_every = int(report_every)
        self.save_every = int(save_every)
        self.save_on_end = save_on_end
        self.recent_lengths: deque[int] = deque(maxlen=window)
        self.recent_survival_seconds: deque[float] = deque(maxlen=window)
        self.recent_rewards: deque[float] = deque(maxlen=window)
        self._episode_lengths = np.zeros(0, dtype=np.int64)
        self._episode_returns = np.zeros(0, dtype=np.float64)
        self._started_at = 0.0
        self._started_timesteps = 0
        self._next_report: int | None = None
        self._next_save: int | None = None
        self._last_saved_timestep = -1

    @staticmethod
    def _next_boundary(current: int, interval: int) -> int | None:
        if interval == 0:
            return None
        return (current // interval + 1) * interval

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()
        self._started_timesteps = int(self.model.num_timesteps)
        self._next_report = self._next_boundary(self._started_timesteps, self.report_every)
        self._next_save = self._next_boundary(self._started_timesteps, self.save_every)

    def _ensure_accumulators(self, count: int) -> None:
        if len(self._episode_lengths) == count:
            return
        self._episode_lengths = np.zeros(count, dtype=np.int64)
        self._episode_returns = np.zeros(count, dtype=np.float64)

    def _record_completed_episodes(self) -> None:
        infos = list(self.locals.get("infos", ()))
        dones = np.asarray(self.locals.get("dones", ()), dtype=bool).reshape(-1)
        rewards = np.asarray(self.locals.get("rewards", ()), dtype=np.float64).reshape(-1)
        count = max(len(infos), len(dones), len(rewards))
        if count == 0:
            return
        self._ensure_accumulators(count)
        if len(rewards) == count:
            self._episode_returns += rewards
        self._episode_lengths += 1

        for index in range(count):
            done = bool(dones[index]) if index < len(dones) else False
            if not done:
                continue
            info = infos[index] if index < len(infos) else {}
            episode = info.get("episode", {}) if isinstance(info, dict) else {}
            length = int(episode.get("l", self._episode_lengths[index]))
            episode_reward = float(episode.get("r", self._episode_returns[index]))
            survival_seconds = float(
                info.get("survival_seconds", length / self.control_hz)
                if isinstance(info, dict)
                else length / self.control_hz
            )
            self.recent_lengths.append(length)
            self.recent_rewards.append(episode_reward)
            self.recent_survival_seconds.append(survival_seconds)
            self._episode_lengths[index] = 0
            self._episode_returns[index] = 0.0

    def _print_progress(self, now: float) -> None:
        elapsed = max(now - self._started_at, 1e-9)
        collected = int(self.model.num_timesteps) - self._started_timesteps
        throughput = collected / elapsed
        if self.recent_survival_seconds:
            survival = np.asarray(self.recent_survival_seconds, dtype=np.float64)
            rewards = np.asarray(self.recent_rewards, dtype=np.float64)
            summary = (
                f"episodes={len(survival)} | median_survival={np.median(survival):.2f}s "
                f"| p25_survival={np.percentile(survival, 25):.2f}s "
                f"| mean_reward={np.mean(rewards):.3f}"
            )
        else:
            summary = "episodes=0 | median_survival=n/a | p25_survival=n/a | mean_reward=n/a"
        print(
            f"Progress: steps={int(self.model.num_timesteps)} | {summary} "
            f"| throughput={throughput:.2f} steps/s",
            flush=True,
        )

    def _on_step(self) -> bool:
        self._record_completed_episodes()
        current = int(self.model.num_timesteps)
        if self._next_report is not None and current >= self._next_report:
            self._print_progress(time.perf_counter())
            while self._next_report <= current:
                self._next_report += self.report_every
        if self._next_save is not None and current >= self._next_save:
            paths = save_training_state(self.model, self.checkpoint)
            self._last_saved_timestep = current
            print(
                f"Saved model and replay buffer to {paths.model} and {paths.replay_buffer}",
                flush=True,
            )
            while self._next_save <= current:
                self._next_save += self.save_every
        return True

    def _on_training_end(self) -> None:
        current = int(self.model.num_timesteps)
        if self.save_on_end and current != self._last_saved_timestep:
            save_training_state(self.model, self.checkpoint)
            self._last_saved_timestep = current


@dataclass(frozen=True)
class EvaluationResult:
    """Raw deterministic episode results plus the checkpoint ranking values."""

    episode_lengths: tuple[int, ...]
    episode_rewards: tuple[float, ...]

    @property
    def median_length(self) -> float:
        return float(np.median(self.episode_lengths))

    @property
    def p25_length(self) -> float:
        return float(np.percentile(self.episode_lengths, 25))

    @property
    def rank(self) -> tuple[float, float]:
        """Higher is better: median survival first, then lower-quartile survival."""

        return self.median_length, self.p25_length


def evaluate_deterministic(
    model: Any,
    env: Any,
    *,
    episodes: int = 10,
    max_steps: int | None = None,
    seed: int | None = None,
) -> EvaluationResult:
    """Evaluate a policy without exploration in an ordinary Gymnasium env."""

    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be at least one when supplied")

    lengths: list[int] = []
    rewards: list[float] = []
    for episode_index in range(episodes):
        reset_seed = None if seed is None else seed + episode_index
        observation, _ = env.reset(seed=reset_seed)
        episode_reward = 0.0
        episode_length = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            scalar_action = int(np.asarray(action).reshape(-1)[0])
            observation, reward, terminated, truncated, _ = env.step(scalar_action)
            episode_reward += float(reward)
            episode_length += 1
            capped = max_steps is not None and episode_length >= max_steps
            if terminated or truncated or capped:
                break
        lengths.append(episode_length)
        rewards.append(episode_reward)
    return EvaluationResult(tuple(lengths), tuple(rewards))


def is_better_evaluation(
    candidate: EvaluationResult, incumbent: EvaluationResult | None
) -> bool:
    """Rank checkpoints lexicographically by median and then p25 survival."""

    return incumbent is None or candidate.rank > incumbent.rank
