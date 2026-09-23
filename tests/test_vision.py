from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import cv2
import numpy as np

from slope_core.vision import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_GRID_SHAPE,
    ROAD_FEATURE_COUNT,
    ROAD_ROWS,
    VisionEncoder,
    detect_game_over,
    extract_frame_features,
    hsv_colour_masks,
)


HEIGHT = 240
WIDTH = 320
GREEN = (40, 255, 80)  # BGR neon green
RED = (0, 0, 255)


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
    # A filled green circle exercises the colour encoding without relying on a
    # ball detector (the new encoder intentionally has none).
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


def game_over_frame(*, accent: tuple[int, int, int] = GREEN) -> np.ndarray:
    """Synthetic approximation of the low, outlined AGAIN control."""

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(
        frame,
        (80, round(HEIGHT * 0.84)),
        (240, round(HEIGHT * 0.96)),
        accent,
        3,
    )
    cv2.putText(
        frame,
        "AGAIN",
        (113, round(HEIGHT * 0.925)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        accent,
        2,
        cv2.LINE_AA,
    )
    return frame


def ad_overlay_frame() -> np.ndarray:
    """Place an observed-style pale interstitial over button-like scenery."""

    frame = game_over_frame()
    cv2.rectangle(frame, (38, 25), (282, 155), (245, 245, 245), -1)
    cv2.rectangle(frame, (48, 35), (272, 82), (75, 90, 115), -1)
    cv2.rectangle(frame, (72, 102), (248, 140), (235, 145, 55), -1)
    return frame


class VisionTests(unittest.TestCase):
    def test_hsv_masks_encode_green_and_red_independently(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 20), (60, 80), GREEN, -1)
        cv2.rectangle(frame, (100, 25), (145, 85), RED, -1)

        green, red = hsv_colour_masks(frame)

        self.assertEqual(green.dtype, np.uint8)
        self.assertEqual(red.dtype, np.uint8)
        self.assertEqual(int(green[50, 30]), 255)
        self.assertEqual(int(red[50, 30]), 0)
        self.assertEqual(int(red[50, 120]), 255)
        self.assertEqual(int(green[50, 120]), 0)

    def test_default_vector_has_documented_dimension_dtype_and_bounds(self) -> None:
        current = live_frame()
        previous = np.zeros_like(current)
        features = extract_frame_features(current, previous)
        vector = features.vector

        self.assertEqual(DEFAULT_GRID_SHAPE, (12, 16))
        self.assertEqual(ROAD_ROWS, 12)
        self.assertEqual(ROAD_FEATURE_COUNT, 6)
        self.assertEqual(DEFAULT_FEATURE_DIM, 649)
        self.assertEqual(vector.shape, (DEFAULT_FEATURE_DIM,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(vector.flags.c_contiguous)
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertGreaterEqual(float(vector.min()), 0.0)
        self.assertLessEqual(float(vector.max()), 1.0)
        self.assertEqual(features.green_occupancy.shape, DEFAULT_GRID_SHAPE)
        self.assertEqual(features.red_occupancy.shape, DEFAULT_GRID_SHAPE)
        self.assertEqual(features.motion.shape, DEFAULT_GRID_SHAPE)
        self.assertEqual(features.road_geometry.shape, (ROAD_ROWS, ROAD_FEATURE_COUNT))
        self.assertGreater(float(features.green_occupancy.sum()), 0.0)
        self.assertGreater(float(features.red_occupancy.sum()), 0.0)
        self.assertGreater(float(features.motion.sum()), 0.0)
        self.assertGreater(features.green_fraction, 0.0)
        self.assertGreater(features.red_fraction, 0.0)

    def test_road_rows_are_geometry_not_a_steering_target(self) -> None:
        features = extract_frame_features(live_frame())
        valid = features.road_geometry[:, 5] == 1.0
        self.assertGreaterEqual(int(np.count_nonzero(valid)), 6)
        valid_rows = features.road_geometry[valid]
        self.assertTrue(np.all(valid_rows[:, 0] < valid_rows[:, 2]))
        self.assertTrue(np.all(valid_rows[:, 2] < valid_rows[:, 1]))
        np.testing.assert_allclose(
            valid_rows[:, 3], valid_rows[:, 1] - valid_rows[:, 0], atol=1e-6
        )

    def test_encoder_is_stateless_and_action_neutral(self) -> None:
        encoder = VisionEncoder()
        current = live_frame()
        previous = np.zeros_like(current)

        first = encoder.encode(current, previous)
        encoder.reset()
        second = encoder.encode(current, previous)

        self.assertEqual(encoder.feature_dim, DEFAULT_FEATURE_DIM)
        self.assertNotIn("action", inspect.signature(encoder.encode).parameters)
        np.testing.assert_array_equal(first.vector, second.vector)
        np.testing.assert_array_equal(first.green_mask, second.green_mask)
        np.testing.assert_array_equal(first.red_mask, second.red_mask)

    def test_labelled_retry_button_is_conservative_game_over_evidence(self) -> None:
        frame = game_over_frame()
        features = extract_frame_features(frame)

        self.assertTrue(features.game_over)
        self.assertGreaterEqual(features.game_over_confidence, 0.5)
        self.assertLessEqual(features.game_over_confidence, 1.0)

    def test_red_hovered_retry_button_is_also_game_over(self) -> None:
        frame = game_over_frame(accent=(0, 0, 90))
        features = extract_frame_features(frame)

        self.assertTrue(features.game_over)
        self.assertGreaterEqual(features.game_over_confidence, 0.5)

    def test_unconnected_horizontal_groups_are_not_death_evidence(self) -> None:
        # These normalized rows reproduce the pattern that the former detector
        # mistook for UI.  In the captured doctor frames they belong to the
        # perspective road behind an interstitial.
        frame = horizontal_groups_frame()
        mask, _ = hsv_colour_masks(frame)

        self.assertFalse(detect_game_over(mask, frame))
        self.assertFalse(extract_frame_features(frame).game_over)

    def test_large_bright_ad_card_vetoes_button_like_background(self) -> None:
        frame = ad_overlay_frame()
        mask, _ = hsv_colour_masks(frame)
        retry_mask, _ = hsv_colour_masks(game_over_frame())

        # The underlying lower control would otherwise satisfy the conservative
        # retry geometry; the foreground ad must win.
        self.assertTrue(detect_game_over(retry_mask))
        self.assertFalse(detect_game_over(mask, frame))
        self.assertFalse(extract_frame_features(frame).game_over)

    def test_captured_doctor_resets_are_live_and_finals_are_death(self) -> None:
        artifact_directory = Path(__file__).parents[1] / "artifacts" / "doctor"
        resets = sorted(artifact_directory.glob("*_reset.png"))
        finals = sorted(artifact_directory.glob("*_final.png"))
        if not resets and not finals:
            self.skipTest("local doctor artifacts are not present")
        self.assertEqual(len(resets), 3)
        self.assertEqual(len(finals), 3)
        for capture in resets:
            with self.subTest(capture=capture.name):
                frame = cv2.imread(str(capture), cv2.IMREAD_COLOR)
                self.assertIsNotNone(frame)
                features = extract_frame_features(frame)
                self.assertFalse(features.game_over)
                self.assertEqual(features.game_over_confidence, 0.0)
        for capture in finals:
            with self.subTest(capture=capture.name):
                frame = cv2.imread(str(capture), cv2.IMREAD_COLOR)
                self.assertIsNotNone(frame)
                features = extract_frame_features(frame)
                self.assertTrue(features.game_over)
                self.assertGreaterEqual(features.game_over_confidence, 0.5)

    def test_static_live_frame_is_not_death(self) -> None:
        frame = live_frame()
        features = extract_frame_features(frame, frame.copy())

        self.assertFalse(features.game_over)
        self.assertEqual(features.game_over_confidence, 0.0)
        np.testing.assert_array_equal(
            features.motion, np.zeros(DEFAULT_GRID_SHAPE, dtype=np.float32)
        )

    def test_missing_previous_frame_has_zero_motion(self) -> None:
        features = VisionEncoder().encode(live_frame())
        np.testing.assert_array_equal(
            features.motion, np.zeros(DEFAULT_GRID_SHAPE, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
