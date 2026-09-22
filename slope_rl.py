"""Train or run a PPO agent against a browser-based Slope game.

Screenshots are observations and keyboard events are actions. The legacy mode
controls one visible desktop game; URL mode owns isolated, clock-controlled
browser pages so several environments can collect experience in parallel.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

try:
    import cv2
    import gymnasium as gym
    import mss
    import numpy as np
    from gymnasium import spaces
    from pynput import keyboard, mouse
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecFrameStack,
        VecMonitor,
    )
except ImportError as exc:
    missing = getattr(exc, "name", "a reinforcement-learning package")
    raise SystemExit(
        f"Missing {missing}. Install the RL dependencies with:\n"
        f"  python -m pip install -r requirements-rl.txt"
    ) from exc

from slope_ai import KeyController, Perception, Region, load_config


MODEL_DIR = Path(__file__).with_name("models")


def restart_key(name: str) -> Any:
    if name == "space":
        return keyboard.Key.space
    if name == "enter":
        return keyboard.Key.enter
    return name


class BrowserSlopeEnv(gym.Env):
    """A Gymnasium environment backed by desktop or isolated-page game I/O."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        region: Region | None,
        fps: int = 12,
        layout: str = "arrows",
        restart: str = "space",
        restart_click: bool = False,
        restart_wait: float = 1.5,
        episode_seconds: int = 180,
        browser_url: str | None = None,
        browser_channel: str = "chrome",
        browser_headless: bool = True,
        browser_load_timeout: float = 120.0,
        browser_realtime: bool = False,
        worker_id: int = 0,
    ) -> None:
        super().__init__()
        if region is None and browser_url is None:
            raise ValueError("A screen region or browser URL is required")
        self.region = region
        self.frame_period = 1.0 / fps
        self.missing_track_limit = max(3, round(fps * 2 / 3))
        self.static_screen_limit = max(5, fps)
        self.restart_button = restart_key(restart)
        self.restart_click = restart_click
        self.restart_wait = restart_wait
        self.max_episode_steps = episode_seconds * fps
        self.action_space = spaces.Discrete(3)  # 0 left, 1 straight, 2 right
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.capture: Any | None = None
        self.keys: KeyController | None = None
        self.browser_session: Any | None = None
        if browser_url is None:
            self.capture = mss.mss()
            self.keys = KeyController(layout)
        else:
            from slope_browser import PlaywrightSlopeSession

            self.browser_session = PlaywrightSlopeSession(
                url=browser_url,
                worker_id=worker_id,
                layout=layout,
                browser_channel=browser_channel,
                headless=browser_headless,
                load_timeout=browser_load_timeout,
                realtime=browser_realtime,
            )
        self.perception = Perception()
        self.input = keyboard.Controller() if browser_url is None else None
        self.mouse = mouse.Controller() if browser_url is None else None
        self.episode_step = 0
        self.low_track_frames = 0
        self.still_frames = 0
        self.previous_small: np.ndarray | None = None
        self.last_step_at = 0.0
        self.has_reset = False
        self.last_terminated = False
        self.last_truncated = False

    def _capture(self) -> tuple[np.ndarray, np.ndarray]:
        if self.capture is None or self.region is None:
            raise RuntimeError("Desktop capture is unavailable in URL mode")
        frame = np.asarray(self.capture.grab(self.region.as_mss()))[:, :, :3]
        return frame, self._observation(frame)

    @staticmethod
    def _observation(frame: np.ndarray) -> np.ndarray:
        # Models expect RGB, while OpenCV and MSS provide BGR here.
        small = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _press_restart(self) -> None:
        if self.input is None or self.mouse is None:
            raise RuntimeError("Desktop restart controls are unavailable in URL mode")
        if self.restart_click:
            position = self.mouse.position
            print(f"Clicking restart at ({int(position[0])}, {int(position[1])})...", flush=True)
            # Unity WebGL can miss a near-instant synthetic click. Holding the
            # button briefly ensures at least one rendered frame sees it down.
            self.mouse.press(mouse.Button.left)
            time.sleep(0.12)
            self.mouse.release(mouse.Button.left)
            return
        self.input.press(self.restart_button)
        time.sleep(0.08)
        self.input.release(self.restart_button)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.browser_session is not None:
            if not self.has_reset:
                reason = "initial"
            elif self.last_truncated:
                reason = "truncated"
            elif self.last_terminated:
                reason = "terminated"
            else:
                reason = "manual"
            frame = self.browser_session.reset(reason, self.restart_wait)
            observation = self._observation(frame)
        else:
            assert self.keys is not None
            self.keys.release()
            self._press_restart()
            time.sleep(self.restart_wait)
            frame, observation = self._capture()
        self.perception = Perception()
        self.episode_step = 0
        self.low_track_frames = 0
        self.still_frames = 0
        self.previous_small = None
        self.last_step_at = time.perf_counter()
        self.has_reset = True
        self.last_terminated = False
        self.last_truncated = False
        result = self.perception.analyse(frame)
        return observation, {"track_confidence": result.confidence}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        direction = (-1, 0, 1)[int(action)]
        if self.browser_session is not None:
            frame = self.browser_session.advance(direction, self.frame_period)
            observation = self._observation(frame)
        else:
            assert self.keys is not None
            self.keys.set_direction(direction)

            deadline = self.last_step_at + self.frame_period
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self.last_step_at = time.perf_counter()
            frame, observation = self._capture()
        result = self.perception.analyse(frame)
        self.episode_step += 1

        motion = 0.0
        current_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.previous_small is not None:
            motion = float(cv2.absdiff(current_gray, self.previous_small).mean())
            if motion < 0.45:
                self.still_frames += 1
            else:
                self.still_frames = max(0, self.still_frames - 3)
        self.previous_small = current_gray

        green_ratio = float(np.count_nonzero(result.green_mask)) / result.green_mask.size
        if green_ratio < 0.0025 or result.confidence < 0.006:
            self.low_track_frames += 1
        else:
            self.low_track_frames = max(0, self.low_track_frames - 2)

        # Some retry screens remove the track, while Y8's Unity version can
        # leave it visible behind the overlay. In the latter case the canvas is
        # effectively static. Requiring consecutive frames avoids false deaths.
        missing_track = self.low_track_frames >= self.missing_track_limit
        static_screen = self.still_frames >= self.static_screen_limit
        terminated = missing_track or static_screen
        truncated = self.episode_step >= self.max_episode_steps

        target_error = abs(result.target_x - result.ball[0]) / max(1.0, frame.shape[1] * 0.35)
        alignment = max(-1.0, 1.0 - target_error)
        reward = 0.035 + 0.025 * alignment
        if result.obstacle is not None:
            reward -= 0.01
        if terminated:
            reward = -2.0

        info = {
            "track_confidence": result.confidence,
            "green_ratio": green_ratio,
            "target_error": target_error,
            "screen_motion": motion,
            "still_frames": self.still_frames,
            "episode_steps": self.episode_step,
        }
        if terminated or truncated:
            if self.keys is not None:
                self.keys.release()
            self.last_terminated = terminated
            self.last_truncated = truncated
        if terminated:
            reason = "missing track" if missing_track else "static screen"
            print(f"Death detected ({reason}, motion={motion:.3f}); restarting...", flush=True)
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.browser_session is not None:
            self.browser_session.close()
        if self.keys is not None:
            self.keys.release()
        if self.capture is not None:
            self.capture.close()
        super().close()


class StopOnEmergencyKey(BaseCallback):
    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__()
        self.stop_event = stop_event

    def _on_step(self) -> bool:
        return not self.stop_event.is_set()


def install_stop_signal_handlers(stop_event: threading.Event) -> None:
    """Turn terminal/container stop signals into a safe PPO shutdown."""

    def request_stop(signal_number: int, _frame: Any) -> None:
        signal_name = signal.Signals(signal_number).name
        print(
            f"{signal_name} received: stopping safely after the current step...",
            flush=True,
        )
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)


def _make_env(args: argparse.Namespace, region: Region | None, worker_id: int) -> BrowserSlopeEnv:
    return BrowserSlopeEnv(
        region=region,
        fps=args.fps,
        layout=args.keys,
        restart=args.restart_key,
        restart_click=args.restart_click,
        restart_wait=args.restart_wait,
        episode_seconds=args.episode_seconds,
        browser_url=args.url,
        browser_channel=args.browser_channel,
        browser_headless=worker_id >= visible_instance_count(args),
        browser_load_timeout=args.browser_load_timeout,
        browser_realtime=(
            args.command == "play" and worker_id < visible_instance_count(args)
        ),
        worker_id=worker_id,
    )


def visible_instance_count(args: argparse.Namespace) -> int:
    """Return how many leading URL-mode workers should show a window."""

    return args.instances if args.headed else args.visible_instances


def make_vec_env(args: argparse.Namespace) -> VecFrameStack:
    region = None if args.url else load_config()
    factories = [
        partial(_make_env, args=args, region=region, worker_id=worker_id)
        for worker_id in range(args.instances)
    ]
    if args.instances == 1:
        vector_env = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(factories, start_method="spawn")

    # VecMonitor adds completed-episode reward and length to PPO's rollout log.
    # Four frames allow the policy to infer speed and direction from still images.
    monitored_env = VecMonitor(vector_env)
    return VecFrameStack(monitored_env, n_stack=4, channels_order="last")


def rollout_steps(instances: int, target_samples: int = 512, batch_size: int = 64) -> int:
    """Keep each PPO rollout near the original 512 aggregate samples."""

    return max(batch_size, math.ceil(target_samples / (instances * batch_size)) * batch_size)


def checkpoint_frequency(checkpoint_every: int, instances: int) -> int:
    """Convert aggregate transitions to Stable-Baselines3 vector callback calls."""

    return max(1, checkpoint_every // instances)


def start_emergency_listener(stop_event: threading.Event) -> keyboard.Listener:
    def on_press(key: Any) -> bool | None:
        if key == keyboard.Key.f9:
            print("F9 pressed: stopping safely after the current step...")
            stop_event.set()
            return False
        return None

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def focus_countdown(seconds: int, restart_click: bool = False) -> None:
    if restart_click:
        print("Leave the game on the death screen and hover the mouse over AGAIN.")
        print("Keep the pointer there for the entire run; it will be clicked automatically.")
    else:
        print("Click the browser game now and leave it focused.")
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}...", flush=True)
        time.sleep(1)


def train(args: argparse.Namespace) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    listener: keyboard.Listener | None = None
    env: VecFrameStack | None = None
    model: PPO | None = None
    install_stop_signal_handlers(stop_event)
    try:
        if args.url:
            visible = visible_instance_count(args)
            print(
                f"Launching {args.instances} isolated browser instance"
                f"{'s' if args.instances != 1 else ''} "
                f"({visible} visible, {args.instances - visible} headless)..."
            )
        else:
            focus_countdown(args.countdown, args.restart_click)
        env = make_vec_env(args)
        listener = start_emergency_listener(stop_event)
        load_path = model_path if model_path.exists() else model_path.with_suffix(".zip")
        n_steps = rollout_steps(args.instances)
        if args.resume and load_path.exists():
            print(f"Resuming {load_path}")
            model = PPO.load(
                load_path,
                env=env,
                device=args.device,
                custom_objects={"n_steps": n_steps},
            )
        else:
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=2.5e-4,
                n_steps=n_steps,
                batch_size=64,
                n_epochs=4,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=0.01,
                verbose=1,
                device=args.device,
            )
        checkpoint = CheckpointCallback(
            save_freq=checkpoint_frequency(args.checkpoint_every, args.instances),
            save_path=str(MODEL_DIR),
            name_prefix="slope_checkpoint",
        )
        callbacks = CallbackList([checkpoint, StopOnEmergencyKey(stop_event)])
        print("Training started. Press F9 at any time to stop and save.")
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            reset_num_timesteps=not args.resume,
        )
    finally:
        pending_error = sys.exc_info()[1]
        cleanup_errors: list[tuple[str, Exception]] = []
        if model is not None:
            try:
                model.save(model_path)
                print(f"Saved model to {model_path.with_suffix('.zip')}")
            except Exception as exc:
                cleanup_errors.append(("save the model", exc))
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                cleanup_errors.append(("close the browser environment", exc))
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:
                cleanup_errors.append(("stop the F9 listener", exc))
        for operation, exc in cleanup_errors:
            print(f"Cleanup warning: could not {operation}: {exc}", file=sys.stderr)
        if pending_error is None and cleanup_errors:
            raise cleanup_errors[0][1]


def play(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    load_path = model_path if model_path.exists() else model_path.with_suffix(".zip")
    if not load_path.exists():
        raise SystemExit(f"Model not found: {load_path}. Train it first with the `train` command.")
    stop_event = threading.Event()
    listener: keyboard.Listener | None = None
    env: VecFrameStack | None = None
    install_stop_signal_handlers(stop_event)
    try:
        if args.url:
            visible = visible_instance_count(args)
            print(
                f"Launching {args.instances} isolated browser instance"
                f"{'s' if args.instances != 1 else ''} "
                f"({visible} visible, {args.instances - visible} headless)..."
            )
        else:
            focus_countdown(args.countdown, args.restart_click)
        env = make_vec_env(args)
        listener = start_emergency_listener(stop_event)
        model = PPO.load(load_path, env=env, device=args.device)
        observation = env.reset()
        print("Trained model is playing. Press F9 to stop.")
        while not stop_event.is_set():
            action, _ = model.predict(observation, deterministic=True)
            observation, _, _, _ = env.step(action)
    finally:
        pending_error = sys.exc_info()[1]
        cleanup_errors: list[tuple[str, Exception]] = []
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                cleanup_errors.append(("close the browser environment", exc))
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:
                cleanup_errors.append(("stop the F9 listener", exc))
        for operation, exc in cleanup_errors:
            print(f"Cleanup warning: could not {operation}: {exc}", file=sys.stderr)
        if pending_error is None and cleanup_errors:
            raise cleanup_errors[0][1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run a PPO agent on browser Slope.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "play"):
        child = subparsers.add_parser(command)
        child.add_argument("--model", default=str(MODEL_DIR / "slope_ppo"))
        child.add_argument("--fps", type=int, default=12)
        child.add_argument("--keys", choices=("arrows", "ad"), default="arrows")
        child.add_argument("--restart-key", choices=("space", "enter", "r"), default="space")
        child.add_argument(
            "--restart-click",
            action="store_true",
            help="click the current mouse position to restart (keep the pointer over AGAIN)",
        )
        child.add_argument("--restart-wait", type=float, default=1.5)
        child.add_argument("--episode-seconds", type=int, default=180)
        child.add_argument("--countdown", type=int, default=5)
        child.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
        child.add_argument(
            "--url",
            help="launch an isolated browser for this game URL instead of capturing the desktop",
        )
        child.add_argument(
            "--instances",
            type=int,
            default=1,
            help="number of isolated URL-mode games to run in parallel (default 1)",
        )
        child.add_argument(
            "--browser-channel",
            choices=("chrome", "msedge", "bundled"),
            default="chrome",
            help="Chromium browser to launch for URL mode (default chrome)",
        )
        display = child.add_mutually_exclusive_group()
        display.add_argument(
            "--headed",
            action="store_true",
            help="show automated browser windows instead of running them headlessly",
        )
        display.add_argument(
            "--visible-instances",
            type=int,
            default=0,
            metavar="COUNT",
            help="show only the first COUNT URL-mode browser windows (default 0)",
        )
        child.add_argument(
            "--browser-load-timeout",
            type=float,
            default=120.0,
            help="seconds allowed for each URL-mode game to load (default 120)",
        )
    training = subparsers.choices["train"]
    training.add_argument("--steps", type=int, default=100_000)
    training.add_argument("--checkpoint-every", type=int, default=10_000)
    training.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 5 <= args.fps <= 60:
        parser.error("--fps must be between 5 and 60")
    if args.countdown < 0 or args.restart_wait < 0:
        parser.error("countdown and restart wait cannot be negative")
    if args.instances < 1:
        parser.error("--instances must be at least 1")
    if args.instances > 1 and not args.url:
        parser.error("--instances greater than 1 requires --url")
    if args.visible_instances < 0:
        parser.error("--visible-instances cannot be negative")
    if args.visible_instances > args.instances:
        parser.error("--visible-instances cannot exceed --instances")
    if (args.headed or args.visible_instances > 0) and not args.url:
        parser.error("browser visibility options require --url")
    if args.browser_load_timeout <= 0:
        parser.error("--browser-load-timeout must be positive")
    if args.episode_seconds < 1:
        parser.error("--episode-seconds must be at least 1")
    if args.command == "train" and (args.steps < 1 or args.checkpoint_every < 1):
        parser.error("steps and checkpoint interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        play(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
