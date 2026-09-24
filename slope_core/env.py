"""Gymnasium environment with an explicit, testable Slope objective."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .browser import BrowserConfig, BrowserError, SlopeBrowser
from .vision import FrameFeatures, VisionEncoder


class GameIO(Protocol):
    """The small browser boundary used by the environment and its tests."""

    def reset(self, reason: str = "initial") -> np.ndarray: ...

    def step(self, direction: int, dt: float) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EnvConfig:
    """Physics and objective settings that must match in train/eval/play."""

    fps: int = 20
    history: int = 4
    max_episode_seconds: int = 180
    alive_reward_per_second: float = 0.20
    death_penalty: float = -1.0
    action_change_penalty: float = 0.001
    # Slope's camera never stops while the ball is alive: on real captures,
    # live frames changed by at least 6.6 grey levels per 1/12 s step.  About a
    # second after a crash the scene freezes (changes below 2), and AGAIN only
    # appears ~2.6 s later.  Ending the episode at the freeze puts the death
    # penalty that much closer to its cause.  Zero disables.
    death_freeze_seconds: float = 0.25
    freeze_pixel_change: float = 2.0

    def __post_init__(self) -> None:
        if not 10 <= self.fps <= 60:
            raise ValueError("fps must be between 10 and 60")
        if self.history < 2:
            raise ValueError("history must be at least 2")
        if self.max_episode_seconds < 1:
            raise ValueError("max_episode_seconds must be positive")
        if self.death_penalty >= 0:
            raise ValueError("death_penalty must be negative")
        if self.death_freeze_seconds < 0 or self.freeze_pixel_change < 0:
            raise ValueError("freeze limits cannot be negative")


class SlopeEnv(gym.Env[np.ndarray, int]):
    """Visual Slope environment whose reward is survival, not a steering heuristic.

    Each observation is a byte-packed history of small red/green frame images
    plus the action that produced each frame. No hand-authored desired
    direction is included. One browser step always means one fixed amount of
    simulated time.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        browser_config: BrowserConfig | None = None,
        config: EnvConfig | None = None,
        *,
        browser: GameIO | None = None,
        encoder: VisionEncoder | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.metadata = {**self.metadata, "render_fps": self.config.fps}
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        if browser is None and browser_config is None:
            raise ValueError("browser_config is required when browser is not supplied")
        self.browser: GameIO = browser or SlopeBrowser(browser_config)  # type: ignore[arg-type]
        self.encoder = encoder or VisionEncoder()
        self.action_space = spaces.Discrete(3)
        self.frame_feature_dim = self.encoder.feature_dim + 3
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_feature_dim * self.config.history,),
            dtype=np.uint8,
        )
        self._history: deque[np.ndarray] = deque(maxlen=self.config.history)
        self._last_frame: np.ndarray | None = None
        self._last_features: FrameFeatures | None = None
        self._previous_action = 1
        self._episode_steps = 0
        self._still_steps = 0
        self._reset_reason = "initial"
        self._needs_reset = True

    @property
    def frame_period(self) -> float:
        return 1.0 / self.config.fps

    @property
    def max_episode_steps(self) -> int:
        return self.config.fps * self.config.max_episode_seconds

    @property
    def freeze_steps(self) -> int:
        return math.ceil(self.config.death_freeze_seconds * self.config.fps)

    @staticmethod
    def _action_vector(action: int) -> np.ndarray:
        vector = np.zeros(3, dtype=np.float32)
        vector[int(action)] = 1.0
        return vector

    def _feature_with_action(self, features: FrameFeatures, action: int) -> np.ndarray:
        vector = np.concatenate((features.vector, self._action_vector(action)))
        normalized = np.clip(vector.astype(np.float32, copy=False), 0.0, 1.0)
        return np.rint(normalized * 255.0).astype(np.uint8)

    def _observation(self) -> np.ndarray:
        if len(self._history) != self.config.history:
            raise RuntimeError("observation history is not initialized")
        return np.concatenate(tuple(self._history)).astype(np.uint8, copy=False)

    def _info(self, features: FrameFeatures) -> dict[str, Any]:
        return {
            "survival_steps": self._episode_steps,
            "survival_seconds": self._episode_steps / self.config.fps,
            "game_over": features.game_over,
            "game_over_confidence": features.game_over_confidence,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        frame = self.browser.reset(self._reset_reason)
        self.encoder.reset()
        features = self.encoder.encode(frame)
        if features.game_over:
            frame = self.browser.reset("terminated")
            features = self.encoder.encode(frame)
            if features.game_over:
                raise BrowserError("game-over overlay remained visible after restart")

        self._episode_steps = 0
        self._still_steps = 0
        self._previous_action = 1
        self._last_frame = frame
        self._last_features = features
        initial = self._feature_with_action(features, self._previous_action)
        self._history.clear()
        self._history.extend(initial.copy() for _ in range(self.config.history))
        self._reset_reason = "manual"
        self._needs_reset = False
        return self._observation(), self._info(features)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        direction = (-1, 0, 1)[int(action)]
        try:
            frame = self.browser.step(direction, self.frame_period)
        except BrowserError as exc:
            self._needs_reset = True
            self._reset_reason = "truncated"
            info = self._info(self._last_features) if self._last_features else {}
            info.update({"browser_error": str(exc), "TimeLimit.truncated": True})
            return self._observation(), 0.0, False, True, info

        features = self.encoder.encode(frame)
        self._episode_steps += 1
        self._history.append(self._feature_with_action(features, int(action)))
        self._track_stillness(frame)

        frozen = self.freeze_steps > 0 and self._still_steps >= self.freeze_steps
        terminated = bool(features.game_over) or frozen
        truncated = self._episode_steps >= self.max_episode_steps and not terminated
        if terminated:
            reward = self.config.death_penalty
        else:
            reward = self.config.alive_reward_per_second * self.frame_period
            if int(action) != self._previous_action:
                reward -= self.config.action_change_penalty

        self._last_frame = frame
        self._last_features = features
        self._previous_action = int(action)
        info = self._info(features)
        if terminated:
            info["death_signal"] = "retry_screen" if features.game_over else "frozen_screen"
        if truncated:
            info["target_survival_reached"] = True
        if terminated or truncated:
            self._needs_reset = True
            self._reset_reason = "terminated" if terminated else "truncated"
        return self._observation(), float(reward), terminated, truncated, info

    def _track_stillness(self, frame: np.ndarray) -> None:
        previous = self._last_frame
        if previous is None or previous.shape != frame.shape:
            self._still_steps = 0
            return
        change = float(
            np.mean(
                cv2.absdiff(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
                )
            )
        )
        if change < self.config.freeze_pixel_change:
            self._still_steps += 1
        else:
            self._still_steps = 0

    def render(self) -> np.ndarray | None:
        if self._last_frame is None:
            return None
        return cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.browser.close()
        super().close()
