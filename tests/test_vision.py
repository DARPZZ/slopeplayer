from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import cv2
import numpy as np

from slope_core.vision import (
    DEFAULT_FEATURE_DIM,
    FRAME_SHAPE,
    VisionEncoder,
    detect_game_over,
    extract_frame_features,
    frame_pixels,
    game_over_confidence,
)


HEIGHT = 240
WIDTH = 320
GREEN = (40, 255, 80)
RED = (0, 0, 255)
FIXTURES = Path(__file__).parent / "fixtures"


def live_frame(*, obstacle: bool = True) -> np.ndarray:
    """Create a perspective road with ordinary horizontal grid segments."""

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    left_top = (145, 55)
    right_top = (175, 55)
    left_bottom = (25, 235)
    right_bottom = (295, 235)
    cv2.line(frame, left_top, left_bottom, GREEN, 5)
    cv2.line(frame, right_top, right_bottom, GREEN, 5)
    for y in (92, 132, 174):
        progress = (y - left_top[1]) / (left_bottom[1] - left_top[1])
        left = round(left_top[0] + progress * (left_bottom[0] - left_top[0]))
        right = round(right_top[0] + progress * (right_bottom[0] - right_top[0]))
        cv2.line(frame, (left, y), (right, y), GREEN, 3)
    cv2.circle(frame, (160, 184), 9, GREEN, -1)
    if obstacle:
        cv2.rectangle(frame, (205, 142), (225, 166), RED, -1)
    return frame


def horizontal_groups_frame(*, line_count: int = 3) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in (round(HEIGHT * 0.78), round(HEIGHT * 0.86), round(HEIGHT * 0.955))[
        :line_count
    ]:
        cv2.line(frame, (48, y), (272, y), GREEN, 5)
    return frame


def fixture(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIXTURES / name), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(FIXTURES / name)
    return frame


class VisionTests(unittest.TestCase):
    def test_pixels_keep_red_and_green_in_separate_channels(self) -> None:
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (319, 399), (0, 255, 0), -1)
        cv2.rectangle(frame, (320, 0), (639, 399), (0, 0, 255), -1)

        pixels = frame_pixels(frame)

        self.assertEqual(pixels.shape, FRAME_SHAPE)
        self.assertEqual(pixels.dtype, np.uint8)
        red, green = pixels
        self.assertEqual(int(red[20, 10]), 0)
        self.assertEqual(int(green[20, 10]), 255)
        self.assertEqual(int(red[20, 50]), 255)
        self.assertEqual(int(green[20, 50]), 0)

    def test_thin_distant_details_survive_as_dimmer_pixels(self) -> None:
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        cv2.line(frame, (0, 200), (639, 200), GREEN, 1)
        cv2.rectangle(frame, (400, 100), (403, 103), RED, -1)

        red, green = frame_pixels(frame)

        self.assertGreater(int(green[20].max()), 0)
        self.assertGreater(int(red[10, 40]), 0)

    def test_every_capture_width_gives_the_same_image_size(self) -> None:
        for name in ("live_start_480.jpg", "live_chevrons_640.jpg"):
            with self.subTest(capture=name):
                self.assertEqual(frame_pixels(fixture(name)).shape, FRAME_SHAPE)

    def test_default_vector_has_documented_dimension_dtype_and_bounds(self) -> None:
        features = extract_frame_features(live_frame())
        vector = features.vector

        self.assertEqual(FRAME_SHAPE, (2, 40, 64))
        self.assertEqual(DEFAULT_FEATURE_DIM, 5120)
        self.assertEqual(vector.shape, (DEFAULT_FEATURE_DIM,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(vector.flags.c_contiguous)
        self.assertGreaterEqual(float(vector.min()), 0.0)
        self.assertLessEqual(float(vector.max()), 1.0)
        np.testing.assert_allclose(
            vector, features.pixels.reshape(-1) / 255.0, atol=1e-7
        )

    def test_encoder_is_stateless_and_action_neutral(self) -> None:
        encoder = VisionEncoder()
        current = live_frame()

        first = encoder.encode(current)
        encoder.reset()
        second = encoder.encode(current)

        self.assertEqual(encoder.feature_dim, DEFAULT_FEATURE_DIM)
        self.assertNotIn("action", inspect.signature(encoder.encode).parameters)
        np.testing.assert_array_equal(first.vector, second.vector)

    def test_real_death_screens_are_game_over_at_both_capture_widths(self) -> None:
        for name in ("death_480.jpg", "death_640.jpg"):
            with self.subTest(capture=name):
                features = extract_frame_features(fixture(name))
                self.assertTrue(features.game_over)
                self.assertGreaterEqual(features.game_over_confidence, 0.65)
                self.assertLessEqual(features.game_over_confidence, 1.0)

    def test_real_live_frames_and_menu_are_not_game_over(self) -> None:
        # Road chevrons fooled the former outlined-box heuristic, the exploding
        # ball precedes the death UI, and the menu shares its Leaderboard control.
        for name in (
            "live_start_480.jpg",
            "live_chevrons_640.jpg",
            "exploding_480.jpg",
            "menu_640.jpg",
        ):
            with self.subTest(capture=name):
                frame = fixture(name)
                self.assertFalse(detect_game_over(frame))
                self.assertLessEqual(game_over_confidence(frame), 0.4)

    def test_death_detection_survives_heavier_jpeg_and_smaller_capture(self) -> None:
        death = fixture("death_640.jpg")
        _, encoded = cv2.imencode(".jpg", death, [cv2.IMWRITE_JPEG_QUALITY, 40])
        self.assertTrue(detect_game_over(cv2.imdecode(encoded, cv2.IMREAD_COLOR)))
        self.assertTrue(
            detect_game_over(cv2.resize(death, (400, 250), interpolation=cv2.INTER_AREA))
        )

    def test_red_hovered_again_control_is_still_game_over(self) -> None:
        frame = fixture("death_480.jpg")
        height, width = frame.shape[:2]
        button = frame[
            round(height * 0.85) : round(height * 0.96),
            round(width * 0.296) : round(width * 0.702),
        ]
        button[:] = button[:, :, ::-1]
        self.assertTrue(detect_game_over(frame))

    def test_synthetic_scenery_is_not_game_over(self) -> None:
        for frame in (live_frame(), horizontal_groups_frame(), np.zeros((240, 320, 3), np.uint8)):
            self.assertFalse(detect_game_over(frame))

    def test_static_live_frame_is_not_death(self) -> None:
        features = extract_frame_features(live_frame())

        self.assertFalse(features.game_over)
        self.assertLess(features.game_over_confidence, 0.4)


if __name__ == "__main__":
    unittest.main()
