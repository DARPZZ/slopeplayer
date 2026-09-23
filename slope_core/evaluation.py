"""Periodic deterministic evaluation and best-policy promotion."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback

from .learner import EvaluationResult, evaluate_deterministic, is_better_evaluation


def best_checkpoint_paths(training_path: str | Path) -> tuple[Path, Path]:
    requested = Path(training_path)
    model = requested if requested.suffix.lower() == ".zip" else Path(f"{requested}.zip")
    best = model.with_name(f"{model.stem}_best.zip")
    return best, best.with_suffix(".json")


class BestModelCallback(BaseCallback):
    """Rank policies by median survival, with p25 survival as tie-breaker."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        training_path: str | Path,
        *,
        eval_every: int = 50_000,
        episodes: int = 5,
        control_hz: float = 20.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_every < 1:
            raise ValueError("eval_every must be positive")
        if episodes < 1:
            raise ValueError("episodes must be positive")
        self.env_factory = env_factory
        self.best_model_path, self.metadata_path = best_checkpoint_paths(training_path)
        self.eval_every = int(eval_every)
        self.episodes = int(episodes)
        self.control_hz = float(control_hz)
        self.eval_env: Any | None = None
        self.best_result: EvaluationResult | None = None
        self.best_timestep = 0
        self.last_evaluation_timestep = 0
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self.metadata_path.is_file():
            return
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            lengths = tuple(int(value) for value in data["best_episode_lengths"])
            rewards = tuple(float(value) for value in data["best_episode_rewards"])
            self.best_result = EvaluationResult(lengths, rewards)
            self.best_timestep = int(data["best_timestep"])
            self.last_evaluation_timestep = int(
                data.get("last_evaluation_timestep", self.best_timestep)
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid evaluation metadata: {self.metadata_path}"
            ) from exc

    def _save_best_model(self) -> None:
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.best_model_path.with_name(
            f".{self.best_model_path.stem}.{uuid.uuid4().hex}.tmp.zip"
        )
        try:
            self.model.save(temporary)
            os.replace(temporary, self.best_model_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_metadata(self, latest: EvaluationResult) -> None:
        assert self.best_result is not None
        data = {
            "selection_metric": "median_survival_steps_then_p25",
            "control_hz": self.control_hz,
            "best_timestep": self.best_timestep,
            "best_episode_lengths": self.best_result.episode_lengths,
            "best_episode_rewards": self.best_result.episode_rewards,
            "best_median_seconds": self.best_result.median_length / self.control_hz,
            "best_p25_seconds": self.best_result.p25_length / self.control_hz,
            "last_evaluation_timestep": self.last_evaluation_timestep,
            "last_episode_lengths": latest.episode_lengths,
            "last_episode_rewards": latest.episode_rewards,
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.metadata_path)

    def _evaluate(self) -> None:
        if self.eval_env is None:
            self.eval_env = self.env_factory()
        result = evaluate_deterministic(
            self.model,
            self.eval_env,
            episodes=self.episodes,
        )
        self.last_evaluation_timestep = int(self.model.num_timesteps)
        median_seconds = result.median_length / self.control_hz
        p25_seconds = result.p25_length / self.control_hz
        print(
            f"Evaluation: steps={self.last_evaluation_timestep} | "
            f"median_survival={median_seconds:.2f}s | "
            f"p25_survival={p25_seconds:.2f}s | episodes={self.episodes}",
            flush=True,
        )
        if is_better_evaluation(result, self.best_result):
            self.best_result = result
            self.best_timestep = self.last_evaluation_timestep
            self._save_best_model()
            print(f"New best policy saved to {self.best_model_path}", flush=True)
        self._write_metadata(result)

    def _on_step(self) -> bool:
        current = int(self.model.num_timesteps)
        next_evaluation = (
            self.last_evaluation_timestep // self.eval_every + 1
        ) * self.eval_every
        if current >= next_evaluation:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

