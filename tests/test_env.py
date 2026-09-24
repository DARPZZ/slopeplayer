import unittest
from types import SimpleNamespace

import numpy as np

from slope_core.browser import BrowserError
from slope_core.env import EnvConfig, SlopeEnv


class FakeEncoder:
    feature_dim = 2

    def reset(self) -> None:
        pass

    def encode(self, frame):
        dead = bool(frame[0, 0, 0] == 255)
        value = float(frame[0, 0, 1]) / 255
        return SimpleNamespace(
            vector=np.asarray((value, 0.0), dtype=np.float32),
            game_over=dead,
            game_over_confidence=1.0 if dead else 0.0,
        )


class FakeBrowser:
    def __init__(self) -> None:
        self.reset_reasons = []
        self.steps = []
        self.dead_next = False
        self.fail_next = False
        self.moving = True
        self.closed = False

    @staticmethod
    def frame(value=64, dead=False, texture=0):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        # Pixels outside [0, 0] are ignored by FakeEncoder; they only give the
        # frame the change a moving camera produces.
        frame[1:, :, :] = texture
        frame[0, 0, 1] = value
        frame[0, 0, 0] = 255 if dead else 0
        return frame

    def reset(self, reason="initial"):
        self.reset_reasons.append(reason)
        return self.frame()

    def step(self, direction, dt):
        self.steps.append((direction, dt))
        if self.fail_next:
            self.fail_next = False
            raise BrowserError("renderer stopped")
        texture = (len(self.steps) % 2) * 100 if self.moving else 0
        frame = self.frame(value=128, dead=self.dead_next, texture=texture)
        self.dead_next = False
        return frame

    def close(self):
        self.closed = True


class SlopeEnvironmentTests(unittest.TestCase):
    def make_env(self, **overrides):
        browser = FakeBrowser()
        settings = {
            "fps": 20,
            "history": 4,
            "max_episode_seconds": 2,
        }
        settings.update(overrides)
        env = SlopeEnv(
            config=EnvConfig(**settings), browser=browser, encoder=FakeEncoder()
        )
        return env, browser

    def test_reset_repeats_first_frame_to_fill_history(self):
        env, browser = self.make_env()
        observation, info = env.reset()
        self.assertEqual(observation.shape, (20,))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertEqual(env.observation_space.dtype, np.uint8)
        self.assertEqual(int(observation.min()), 0)
        self.assertEqual(int(observation.max()), 255)
        np.testing.assert_array_equal(observation[:5], (64, 0, 0, 255, 0))
        np.testing.assert_array_equal(observation[:5], observation[5:10])
        self.assertEqual(browser.reset_reasons, ["initial"])
        self.assertEqual(info["survival_steps"], 0)
        env.close()

    def test_action_advances_exact_fixed_time_and_changes_reward(self):
        env, browser = self.make_env()
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(browser.steps, [(-1, 0.05)])
        self.assertAlmostEqual(reward, 0.009)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["survival_steps"], 1)
        env.close()

    def test_death_is_terminal_on_the_first_detected_frame(self):
        env, browser = self.make_env()
        env.reset()
        browser.dead_next = True
        _, reward, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)
        self.assertTrue(info["game_over"])
        self.assertEqual(info["death_signal"], "retry_screen")
        with self.assertRaises(RuntimeError):
            env.step(1)
        env.reset()
        self.assertEqual(browser.reset_reasons[-1], "terminated")
        env.close()

    def test_time_limit_has_normal_alive_reward_and_is_truncated(self):
        env, _ = self.make_env(fps=10, max_episode_seconds=1)
        env.reset()
        result = None
        for _ in range(10):
            result = env.step(1)
        assert result is not None
        _, reward, terminated, truncated, info = result
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, 0.02)
        self.assertTrue(info["target_survival_reached"])
        env.close()

    def test_frozen_screen_is_a_death_before_retry_ui_appears(self):
        env, browser = self.make_env(fps=12, death_freeze_seconds=0.25)
        browser.moving = False
        env.reset()
        # The first step differs from the reset frame; three still steps follow.
        results = [env.step(1) for _ in range(4)]
        for _, _, terminated, truncated, _ in results[:-1]:
            self.assertFalse(terminated or truncated)
        _, reward, terminated, truncated, info = results[-1]
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)
        self.assertFalse(info["game_over"])
        self.assertEqual(info["death_signal"], "frozen_screen")
        env.reset()
        self.assertEqual(browser.reset_reasons[-1], "terminated")
        env.close()

    def test_moving_screen_never_counts_as_frozen(self):
        env, _ = self.make_env(fps=12, max_episode_seconds=60)
        env.reset()
        for _ in range(40):
            _, _, terminated, truncated, _ = env.step(1)
            self.assertFalse(terminated or truncated)
        env.close()

    def test_browser_failure_is_not_reported_as_a_death(self):
        env, browser = self.make_env()
        env.reset()
        browser.fail_next = True
        _, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIn("browser_error", info)
        env.close()


if __name__ == "__main__":
    unittest.main()
