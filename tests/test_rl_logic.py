import signal
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from slope_browser import PlaywrightSlopeSession, normalize_game_url, playwright_key_names
from slope_rl import (
    BrowserSlopeEnv,
    SaveBestTrainingReward,
    _make_env,
    checkpoint_frequency,
    heuristic_actions,
    install_stop_signal_handlers,
    parse_args,
    restart_key,
    rollout_steps,
    visible_instance_count,
)


class RlUtilityTests(unittest.TestCase):
    def test_stop_signals_request_a_safe_shutdown(self) -> None:
        stop_event = threading.Event()
        with patch("slope_rl.signal.signal") as register, patch("builtins.print"):
            install_stop_signal_handlers(stop_event)
            handlers = {call.args[0]: call.args[1] for call in register.call_args_list}
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        self.assertTrue(stop_event.is_set())

    def test_best_training_reward_is_saved_separately(self) -> None:
        with TemporaryDirectory() as directory:
            save_path = Path(directory) / "test_best"
            callback = SaveBestTrainingReward(
                save_path=save_path, window=3, minimum_episodes=2
            )
            callback.model = Mock()
            callback.locals = {"infos": [{"episode": {"r": 1.0}}]}
            self.assertTrue(callback._on_step())
            callback.model.save.assert_not_called()
            callback.locals = {"infos": [{"episode": {"r": 3.0}}]}
            with patch("builtins.print"):
                self.assertTrue(callback._on_step())
            callback.model.save.assert_called_once_with(save_path)

            resumed = SaveBestTrainingReward(save_path, window=3, minimum_episodes=2)
            self.assertEqual(resumed.best_mean_reward, 2.0)
            self.assertEqual(list(resumed.rewards), [1.0, 3.0])

    def test_character_restart_key_is_preserved(self) -> None:
        self.assertEqual(restart_key("r"), "r")

    def test_heuristic_warm_start_labels_steering_directions(self) -> None:
        observations = np.zeros((3, 15), dtype=np.float32)
        observations[:, 4] = (-0.5, 0.0, 0.5)
        observations[:, 8] = 0.5
        np.testing.assert_array_equal(heuristic_actions(observations), (0, 1, 2))

    def test_ppo_accepts_the_stacked_camera_observation(self) -> None:
        class FakeCameraEnvironment(gym.Env):
            action_space = spaces.Discrete(3)
            observation_space = spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros((84, 84, 3), dtype=np.uint8), {}

            def step(self, action):
                return np.zeros((84, 84, 3), dtype=np.uint8), 0.0, False, False, {}

        environment = VecFrameStack(
            DummyVecEnv([FakeCameraEnvironment]), n_stack=4, channels_order="last"
        )
        model = PPO(
            "CnnPolicy", environment, n_steps=8, batch_size=4, n_epochs=1, device="cpu"
        )
        observation = environment.reset()
        action, _ = model.predict(observation)
        self.assertIn(int(action[0]), (0, 1, 2))
        environment.close()

    def test_ppo_accepts_two_parallel_stacked_observations(self) -> None:
        class FakeCameraEnvironment(gym.Env):
            action_space = spaces.Discrete(3)
            observation_space = spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros((84, 84, 3), dtype=np.uint8), {}

            def step(self, action):
                return np.zeros((84, 84, 3), dtype=np.uint8), 0.0, False, False, {}

        environment = VecFrameStack(
            DummyVecEnv([FakeCameraEnvironment, FakeCameraEnvironment]),
            n_stack=4,
            channels_order="last",
        )
        observation = environment.reset()
        self.assertEqual(observation.shape, (2, 84, 84, 12))
        model = PPO(
            "CnnPolicy", environment, n_steps=4, batch_size=4, n_epochs=1, device="cpu"
        )
        action, _ = model.predict(observation)
        self.assertEqual(action.shape, (2,))
        environment.close()

    def test_parallel_training_helpers_use_aggregate_steps(self) -> None:
        self.assertEqual(rollout_steps(1), 512)
        self.assertEqual(rollout_steps(3), 192)
        self.assertEqual(rollout_steps(4), 128)
        self.assertEqual(checkpoint_frequency(10_000, 4), 2_500)
        self.assertEqual(checkpoint_frequency(2, 4), 1)

    def test_url_mode_arguments(self) -> None:
        args = parse_args(
            [
                "train",
                "--url",
                "https://da.y8.com/games/slope",
                "--instances",
                "4",
            ]
        )
        self.assertEqual(args.instances, 4)
        self.assertEqual(args.browser_channel, "chrome")
        self.assertEqual(args.observation, "features")
        self.assertEqual(args.browser_capture_width, 640)
        self.assertEqual(args.browser_screenshot_format, "jpeg")
        self.assertEqual(args.torch_threads, 1)
        self.assertFalse(args.headed)
        self.assertEqual(args.visible_instances, 0)

    def test_only_requested_browser_workers_are_visible(self) -> None:
        args = parse_args(
            [
                "train",
                "--url",
                "https://da.y8.com/games/slope",
                "--instances",
                "4",
                "--visible-instances",
                "1",
            ]
        )
        self.assertEqual(visible_instance_count(args), 1)

        visible_worker = _make_env(args, region=None, worker_id=0)
        hidden_worker = _make_env(args, region=None, worker_id=1)
        try:
            self.assertFalse(visible_worker.browser_session.headless)
            self.assertTrue(hidden_worker.browser_session.headless)
        finally:
            visible_worker.close()
            hidden_worker.close()

        all_visible = parse_args(
            [
                "train",
                "--url",
                "https://da.y8.com/games/slope",
                "--instances",
                "4",
                "--headed",
            ]
        )
        self.assertEqual(visible_instance_count(all_visible), 4)

    def test_only_visible_playback_uses_the_realtime_browser_clock(self) -> None:
        args = parse_args(
            [
                "play",
                "--url",
                "https://da.y8.com/games/slope",
                "--instances",
                "2",
                "--visible-instances",
                "1",
            ]
        )
        visible_worker = _make_env(args, region=None, worker_id=0)
        hidden_worker = _make_env(args, region=None, worker_id=1)
        try:
            self.assertTrue(visible_worker.browser_session.realtime)
            self.assertFalse(hidden_worker.browser_session.realtime)
        finally:
            visible_worker.close()
            hidden_worker.close()

    def test_realtime_browser_advance_keeps_the_clock_running(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game",
            worker_id=0,
            layout="arrows",
            headless=False,
            realtime=True,
        )
        session.page = Mock()
        session.frozen = True
        session._raw_frame = Mock(return_value=np.zeros((600, 960, 3), dtype=np.uint8))
        session.advance(direction=1, frame_period=0.001)
        session.page.clock.resume.assert_called_once_with()
        session.page.keyboard.down.assert_called_once_with("ArrowRight")
        self.assertFalse(session.frozen)

    def test_visible_instances_cannot_exceed_worker_count(self) -> None:
        invalid_arguments = (
            [
                "train",
                "--url",
                "https://da.y8.com/games/slope",
                "--instances",
                "2",
                "--visible-instances",
                "3",
            ],
            [
                "train",
                "--url",
                "https://da.y8.com/games/slope",
                "--visible-instances",
                "-1",
            ],
            ["train", "--visible-instances", "1"],
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    parse_args(arguments)

    def test_parallel_desktop_mode_is_rejected(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["train", "--instances", "2"])

    def test_y8_page_is_normalized_to_the_official_embed(self) -> None:
        self.assertEqual(
            normalize_game_url("https://da.y8.com/games/slope"),
            "https://www.y8.com/embed/slope",
        )
        self.assertEqual(
            normalize_game_url("https://example.com/games/slope"),
            "https://example.com/games/slope",
        )

    def test_playwright_key_layouts(self) -> None:
        self.assertEqual(playwright_key_names("arrows"), ("ArrowLeft", "ArrowRight"))
        self.assertEqual(playwright_key_names("ad"), ("a", "d"))

    def test_visible_browser_clock_is_sliced_and_headless_clock_is_fast(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=0, layout="arrows", headless=False
        )
        session.page = Mock()
        with patch("slope_browser.time.sleep") as sleep:
            session._advance_clock(83)
        calls = [call.args[0] for call in session.page.clock.run_for.call_args_list]
        self.assertEqual(calls, [16, 16, 16, 16, 16, 3])
        self.assertTrue(sleep.called)

        session.headless = True
        session.page.clock.run_for.reset_mock()
        session._advance_clock(83)
        session.page.clock.run_for.assert_called_once_with(83)

    def test_unexpected_popup_is_closed(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=0, layout="arrows"
        )
        session.page = Mock()
        popup = Mock()
        popup.url = "https://ads.example/popup"
        with patch("builtins.print"):
            session._close_unexpected_page(popup)
        popup.close.assert_called_once_with()
        session._close_unexpected_page(session.page)
        session.page.close.assert_not_called()

    def test_covered_ad_close_uses_direct_event_fallback(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=0, layout="arrows"
        )
        close = Mock()
        close.click.side_effect = TimeoutError("#card intercepts pointer events")
        self.assertTrue(session._click_ad_close(close))
        close.click.assert_called_once_with(timeout=1_000)
        close.dispatch_event.assert_called_once_with(
            "click", timeout=session.operation_timeout_ms
        )

    def test_ad_close_dom_race_does_not_escape(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=0, layout="arrows"
        )
        close = Mock()
        close.click.side_effect = TimeoutError("covered")
        close.dispatch_event.side_effect = RuntimeError("detached")
        session._ad_is_present = Mock(return_value=False)
        self.assertTrue(session._click_ad_close(close))

    def test_visible_close_control_keeps_ad_marked_present(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=0, layout="arrows"
        )
        session.page = Mock()
        session._ad_is_present = Mock(return_value=False)
        session._visible_ad_close = Mock(return_value=Mock())
        self.assertFalse(session._wait_for_ad_to_close(timeout=0))
        session._visible_ad_close.return_value = None
        self.assertTrue(session._wait_for_ad_to_close(timeout=0))

    def test_reset_reopens_browser_after_transient_failure(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=2, layout="arrows"
        )
        frame = np.zeros((600, 960, 3), dtype=np.uint8)
        session._launch = Mock()
        session._reset_once = Mock(side_effect=[RuntimeError("ad race"), frame])
        session.close = Mock()
        with patch("builtins.print"):
            result = session.reset("terminated", 0.0)
        self.assertIs(result, frame)
        self.assertEqual(session._launch.call_count, 2)
        self.assertEqual(session._reset_once.call_count, 2)
        session.close.assert_called_once_with()

    def test_close_clears_browser_session_state(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game", worker_id=0, layout="arrows"
        )
        session.started = True
        session.frozen = True
        session.direction = 1
        session.close()
        self.assertFalse(session.started)
        self.assertFalse(session.frozen)
        self.assertEqual(session.direction, 0)

    def test_privacy_panel_detection_is_locale_independent(self) -> None:
        frame = np.zeros((600, 960, 3), dtype=np.uint8)
        frame[100:500, 140:820] = 255
        self.assertEqual(
            PlaywrightSlopeSession._privacy_panel(frame),
            (140, 100, 680, 400),
        )

    def test_url_environment_uses_page_scoped_session(self) -> None:
        class FakeSession:
            def __init__(self) -> None:
                self.directions: list[int] = []
                self.closed = False

            def reset(self, reason, restart_wait):
                self.reset_reason = reason
                return np.zeros((600, 960, 3), dtype=np.uint8)

            def advance(self, direction, frame_period):
                self.directions.append(direction)
                return np.zeros((600, 960, 3), dtype=np.uint8)

            def close(self):
                self.closed = True

        environment = BrowserSlopeEnv(
            region=None,
            browser_url="https://example.com/game",
            restart_wait=0,
            observation_mode="camera",
        )
        session = FakeSession()
        environment.browser_session = session
        observation, _ = environment.reset()
        self.assertEqual(observation.shape, (84, 84, 3))
        environment.step(2)
        self.assertEqual(session.directions, [1])
        environment.close()
        self.assertTrue(session.closed)

    def test_feature_observation_is_compact_normalized_and_temporal(self) -> None:
        environment = BrowserSlopeEnv(
            region=None,
            browser_url="https://example.com/game",
            restart_wait=0,
            observation_mode="features",
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[220:360, 100:540] = (0, 255, 0)

        class FakeSession:
            def reset(self, reason, restart_wait):
                return frame

            def advance(self, direction, frame_period):
                return frame

            def close(self):
                pass

        environment.browser_session = FakeSession()
        try:
            observation, _ = environment.reset()
            self.assertEqual(observation.shape, (15,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(np.all(observation >= -1.0))
            self.assertTrue(np.all(observation <= 1.0))
            next_observation, _, _, _, info = environment.step(2)
            self.assertEqual(next_observation[-1], 1.0)
            self.assertGreaterEqual(info["browser_step_ms"], 0.0)
            self.assertGreaterEqual(info["vision_ms"], 0.0)
        finally:
            environment.close()

    def test_static_screen_requires_one_second_of_interval_checks(self) -> None:
        environment = BrowserSlopeEnv(
            region=None,
            browser_url="https://example.com/game",
            fps=20,
        )
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        try:
            _, detected = environment._static_screen_detected(frame)
            self.assertFalse(detected)
            for _ in range(environment.static_check_interval * 4 - 1):
                _, detected = environment._static_screen_detected(frame)
                self.assertFalse(detected)
            motion, detected = environment._static_screen_detected(frame)
            self.assertTrue(detected)
            self.assertEqual(motion, 0.0)
            self.assertEqual(environment.static_checks, 4)
        finally:
            environment.close()

    def test_colour_changes_are_not_treated_as_a_static_screen(self) -> None:
        environment = BrowserSlopeEnv(
            region=None,
            browser_url="https://example.com/game",
            fps=20,
        )
        red = np.zeros((84, 84, 3), dtype=np.uint8)
        red[:] = (0, 0, 255)
        equal_brightness_green = np.zeros_like(red)
        equal_brightness_green[:] = (0, 130, 0)
        detected = False
        try:
            environment._static_screen_detected(red)
            for index in range(environment.static_check_interval * 8):
                frame = equal_brightness_green if index % 2 == 0 else red
                motion, detected = environment._static_screen_detected(frame)
                self.assertFalse(detected)
            self.assertGreater(motion, environment.static_motion_threshold)
            self.assertEqual(environment.static_checks, 0)
        finally:
            environment.close()

    def test_jpeg_capture_uses_reduced_viewport_and_quality(self) -> None:
        session = PlaywrightSlopeSession(
            "https://example.com/game",
            worker_id=0,
            layout="arrows",
            capture_width=640,
            screenshot_format="jpeg",
        )
        session.page = Mock()
        session.canvas = Mock()
        session.canvas.bounding_box.return_value = {
            "x": 0,
            "y": 0,
            "width": 640,
            "height": 360,
        }
        encoded = cv2.imencode(".jpg", np.zeros((360, 640, 3), dtype=np.uint8))[1]
        session.page.screenshot.return_value = encoded.tobytes()
        frame = session._raw_frame()
        self.assertEqual(session.capture_height, 427)
        self.assertEqual(frame.shape, (360, 640, 3))
        session.page.screenshot.assert_called_once_with(
            clip=session.canvas.bounding_box.return_value,
            type="jpeg",
            timeout=session.operation_timeout_ms,
            quality=70,
        )


if __name__ == "__main__":
    unittest.main()
