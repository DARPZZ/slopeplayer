import unittest
from types import SimpleNamespace

import numpy as np

from slope_core.browser import BrowserError
from slope_core.env import EnvConfig, SlopeEnv


class FakeEncoder:
    feature_dim = 2

    def reset(self) -> None:
        pass

    def encode(self, frame, previous_frame=None):
        dead = bool(frame[0, 0, 0] == 255)
        value = float(frame[0, 0, 1]) / 255
        motion = 0.0 if previous_frame is None else 0.5
        return SimpleNamespace(
            vector=np.asarray((value, motion), dtype=np.float32),
            game_over=dead,
            game_over_confidence=1.0 if dead else 0.0,
            green_fraction=value,
            red_fraction=0.0,
        )


class FakeBrowser:
    def __init__(self) -> None:
        self.reset_reasons = []
        self.steps = []
        self.dead_next = False
        self.fail_next = False
        self.closed = False

    @staticmethod
    def frame(value=64, dead=False):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
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
        frame = self.frame(value=128, dead=self.dead_next)
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
        self.assertAlmostEqual(reward, 0.009)  # 0.20/20 minus switch cost
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
