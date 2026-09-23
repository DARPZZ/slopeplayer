"""Command line for environment validation, training, evaluation, and play."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList

from .browser import BrowserConfig
from .env import EnvConfig, SlopeEnv
from .evaluation import BestModelCallback, best_checkpoint_paths
from .learner import (
    CheckpointPaths,
    SlopeQRDQN,
    TrainingProgressCallback,
    checkpoint_paths,
    create_or_resume_qrdqn,
    evaluate_deterministic,
    save_training_state,
)


DEFAULT_URL = "https://da.y8.com/games/slope"
DEFAULT_CHECKPOINT = Path("runs/slope_qrdqn")
EXPLORATION_FRACTION = 0.30


def prepare_training_artifacts(
    path: str | Path, *, resume: bool, overwrite: bool
) -> CheckpointPaths:
    """Protect or clear every latest/best artifact for a training run."""

    if overwrite and resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive")
    paths = checkpoint_paths(path)
    best_model, best_metadata = best_checkpoint_paths(path)
    artifacts = (paths.model, paths.replay_buffer, best_model, best_metadata)
    if resume:
        return paths
    existing = [artifact for artifact in artifacts if artifact.exists()]
    if existing and not overwrite:
        listed = ", ".join(str(artifact) for artifact in existing)
        raise SystemExit(
            "Fresh training would overwrite existing run artifacts: "
            f"{listed}. Use --resume, choose another --model path, or "
            "explicitly pass --overwrite."
        )
    if overwrite:
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
    return paths


def make_env(
    args: argparse.Namespace,
    *,
    headless: bool,
    max_episode_seconds: int | None = None,
) -> SlopeEnv:
    width = args.capture_width
    browser = BrowserConfig(
        url=args.url,
        channel=args.browser_channel,
        headless=headless,
        key_layout=args.keys,
        viewport_width=width,
        viewport_height=round(width * 2 / 3),
        screenshot_format=args.screenshot_format,
    )
    environment = EnvConfig(
        fps=args.fps,
        history=args.history,
        max_episode_seconds=max_episode_seconds or args.episode_seconds,
    )
    return SlopeEnv(browser_config=browser, config=environment)


def model_file(path: str | Path) -> Path:
    requested = Path(path)
    return requested if requested.suffix.lower() == ".zip" else Path(f"{requested}.zip")


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable; training will use the CPU"
    return f"CUDA ready: {torch.cuda.get_device_name(0)} ({torch.__version__})"


def doctor(args: argparse.Namespace) -> None:
    """Exercise real reset/action/death behavior before allowing a long run."""

    print(cuda_summary(), flush=True)
    env = make_env(args, headless=not args.headed, max_episode_seconds=15)
    policies = (("hold-left", 0), ("hold-right", 2), ("neutral", 1))
    detected_deaths = 0
    diagnostic_directory = Path("artifacts/doctor")
    diagnostic_directory.mkdir(parents=True, exist_ok=True)
    try:
        for label, action in policies:
            observation, _ = env.reset()
            if env._last_frame is not None:
                cv2.imwrite(
                    str(diagnostic_directory / f"{label}_reset.png"), env._last_frame
                )
            if not env.observation_space.contains(observation):
                raise RuntimeError("encoder produced an observation outside its declared space")
            started = time.perf_counter()
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                observation, _, terminated, truncated, info = env.step(action)
            elapsed = max(time.perf_counter() - started, 1e-9)
            steps = int(info.get("survival_steps", 0))
            detected_deaths += int(terminated)
            if env._last_frame is not None:
                cv2.imwrite(
                    str(diagnostic_directory / f"{label}_final.png"), env._last_frame
                )
            print(
                f"Doctor {label}: {'death detected' if terminated else 'time limit'} "
                f"after {steps / args.fps:.2f}s | {steps / elapsed:.2f} steps/s | "
                f"death_confidence={info.get('game_over_confidence', 0.0):.3f}",
                flush=True,
            )
        if detected_deaths < 2:
            raise RuntimeError(
                "Environment validation failed: fewer than two forced trajectories "
                "produced a game-over signal. Do not train until detection is fixed."
            )
        print(
            f"Environment validation passed. Observation size: "
            f"{env.observation_space.shape[0]}",
            flush=True,
        )
    finally:
        env.close()


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    prepare_training_artifacts(
        args.model, resume=args.resume, overwrite=args.overwrite
    )

    print(cuda_summary(), flush=True)
    env = make_env(args, headless=not args.headed)
    model: SlopeQRDQN | None = None
    try:
        model = create_or_resume_qrdqn(
            env,
            args.model,
            resume=args.resume,
            device=args.device,
            seed=args.seed,
            # Anneal over the first 30% of the planned run. Stored in the
            # checkpoint, so later resumes keep this schedule.
            exploration_steps=max(1, round(args.steps * EXPLORATION_FRACTION)),
        )
        progress = TrainingProgressCallback(
            args.model,
            control_hz=args.fps,
            report_every=args.report_every,
            save_every=args.save_every,
            save_on_end=False,
        )
        callbacks: list[Any] = [progress]
        if args.eval_every > 0:
            callbacks.append(
                BestModelCallback(
                    lambda: make_env(args, headless=True),
                    args.model,
                    eval_every=args.eval_every,
                    episodes=args.eval_episodes,
                    control_hz=args.fps,
                )
            )
        print(
            "Training QR-DQN. Press Ctrl+C to stop safely and save model + replay buffer.",
            flush=True,
        )
        model.learn(
            total_timesteps=args.steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not args.resume,
            log_interval=4,
        )
    except KeyboardInterrupt:
        print("Stop requested; saving resumable state...", flush=True)
    finally:
        pending_error = sys.exc_info()[1]
        try:
            if model is not None:
                saved = save_training_state(model, args.model)
                print(
                    f"Saved latest model to {saved.model} and replay to "
                    f"{saved.replay_buffer}",
                    flush=True,
                )
        finally:
            env.close()
        if pending_error is not None and not isinstance(pending_error, KeyboardInterrupt):
            raise pending_error


def load_play_model(path: str | Path, env: SlopeEnv, device: str) -> SlopeQRDQN:
    load_path = model_file(path)
    if not load_path.is_file():
        raise SystemExit(f"Model not found: {load_path}")
    return SlopeQRDQN.load(load_path, env=env, device=device)


def play(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    env = make_env(args, headless=args.headless)
    try:
        model = load_play_model(args.model, env, args.device)
        observation, _ = env.reset()
        print("Deterministic policy is playing. Press Ctrl+C to stop.", flush=True)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(
                int(np.asarray(action).reshape(-1)[0])
            )
            if terminated or truncated:
                print(
                    f"Episode: {info['survival_seconds']:.2f}s "
                    f"({'death' if terminated else 'target reached'})",
                    flush=True,
                )
                observation, _ = env.reset()
    except KeyboardInterrupt:
        print("Playback stopped.", flush=True)
    finally:
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    env = make_env(args, headless=True)
    try:
        model = load_play_model(args.model, env, args.device)
        result = evaluate_deterministic(
            model,
            env,
            episodes=args.episodes,
            seed=args.seed,
        )
        seconds = np.asarray(result.episode_lengths, dtype=np.float64) / args.fps
        print(
            f"Evaluation ({args.episodes} episodes): median={np.median(seconds):.2f}s | "
            f"p25={np.percentile(seconds, 25):.2f}s | "
            f"mean={np.mean(seconds):.2f}s | max={np.max(seconds):.2f}s",
            flush=True,
        )
        print("Episode seconds: " + ", ".join(f"{value:.2f}" for value in seconds))
    finally:
        env.close()


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--browser-channel",
        choices=("bundled", "chrome", "msedge"),
        default="bundled",
    )
    parser.add_argument("--keys", choices=("arrows", "ad"), default="arrows")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--episode-seconds", type=int, default=180)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument(
        "--screenshot-format", choices=("jpeg", "png"), default="jpeg"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fresh visual QR-DQN agent for browser Slope."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor_parser = commands.add_parser("doctor", help="validate the real environment")
    add_environment_arguments(doctor_parser)
    doctor_display = doctor_parser.add_mutually_exclusive_group()
    doctor_display.add_argument(
        "--headed", dest="headed", action="store_true", help="show the browser window"
    )
    doctor_display.add_argument(
        "--headless",
        dest="headed",
        action="store_false",
        help="hide the browser window (default)",
    )
    doctor_parser.set_defaults(headed=False)

    train_parser = commands.add_parser("train", help="train QR-DQN")
    add_environment_arguments(train_parser)
    train_parser.add_argument("--model", default=str(DEFAULT_CHECKPOINT))
    train_parser.add_argument("--steps", type=int, default=500_000)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--overwrite", action="store_true")
    train_display = train_parser.add_mutually_exclusive_group()
    train_display.add_argument(
        "--headed", dest="headed", action="store_true", help="show the browser window"
    )
    train_display.add_argument(
        "--headless",
        dest="headed",
        action="store_false",
        help="hide the browser window (default)",
    )
    train_parser.set_defaults(headed=False)
    train_parser.add_argument("--report-every", type=int, default=500)
    train_parser.add_argument("--save-every", type=int, default=50_000)
    train_parser.add_argument("--eval-every", type=int, default=50_000)
    train_parser.add_argument("--eval-episodes", type=int, default=10)

    play_parser = commands.add_parser("play", help="watch a deterministic model")
    add_environment_arguments(play_parser)
    play_parser.add_argument("--model", default="runs/slope_qrdqn_best")
    play_parser.add_argument("--headless", action="store_true")

    evaluate_parser = commands.add_parser("evaluate", help="measure survival")
    add_environment_arguments(evaluate_parser)
    evaluate_parser.add_argument("--model", default="runs/slope_qrdqn_best")
    evaluate_parser.add_argument("--episodes", type=int, default=20)

    args = parser.parse_args(argv)
    if args.fps < 10 or args.fps > 60:
        parser.error("--fps must be between 10 and 60")
    if args.history < 2:
        parser.error("--history must be at least 2")
    if args.episode_seconds < 1 or args.capture_width < 320:
        parser.error("episode duration and capture width must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.command == "train" and (
        args.steps < 1
        or args.report_every < 1
        or args.save_every < 1
        or args.eval_every < 0
        or args.eval_episodes < 1
    ):
        parser.error("training counts must be positive; --eval-every may be zero")
    if args.command == "evaluate" and args.episodes < 1:
        parser.error("--episodes must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "doctor":
        doctor(args)
    elif args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)
    else:
        evaluate(args)


__all__ = ["main", "make_env", "parse_args", "prepare_training_artifacts"]
