import unittest

import cv2
import numpy as np

from slope_ai import Perception


def synthetic_frame(obstacle_x: int | None = None) -> np.ndarray:
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    green = (40, 255, 80)
    # A perspective track that bends right, plus a round player ball.
    cv2.line(frame, (160, 590), (480, 210), green, 10)
    cv2.line(frame, (850, 590), (520, 210), green, 10)
    for y in range(270, 590, 50):
        fraction = (y - 210) / 380
        left = int(480 + (160 - 480) * fraction)
        right = int(520 + (850 - 520) * fraction)
        cv2.line(frame, (left, y), (right, y), green, 4)
    cv2.circle(frame, (450, 500), 25, green, -1)
    if obstacle_x is not None:
        cv2.rectangle(frame, (obstacle_x - 30, 390), (obstacle_x + 30, 455), (0, 0, 255), -1)
    return frame


class PerceptionTests(unittest.TestCase):
    def test_detects_ball_and_track(self) -> None:
        result = Perception().analyse(synthetic_frame(), now=1.0)
        self.assertLess(abs(result.ball[0] - 450), 35)
        self.assertLess(abs(result.ball[1] - 500), 35)
        self.assertGreater(result.confidence, 0)
        self.assertGreater(result.target_x, 350)
        self.assertLess(result.target_x, 600)

    def test_steers_away_from_obstacle(self) -> None:
        normal = Perception().analyse(synthetic_frame(), now=1.0)
        blocked = Perception().analyse(synthetic_frame(obstacle_x=450), now=1.0)
        self.assertIsNotNone(blocked.obstacle)
        self.assertGreater(abs(blocked.raw_target_x - normal.raw_target_x), 50)


if __name__ == "__main__":
    unittest.main()
