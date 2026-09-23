"""Deterministic, action-neutral visual features for Slope.

The extractor deliberately describes what is visible instead of producing a
steering target.  It has no mutable state: temporal motion is computed only
from the explicitly supplied previous frame.

The default flat-vector layout is, in order::

    green occupancy       12 * 16 = 192
    red occupancy         12 * 16 = 192
    grayscale motion      12 * 16 = 192
    road geometry         12 *  6 =  72
    game-over flag                  =   1
                                      -----
                                        649 float32 values

Each road row contains ``left, right, center, width, density, valid``.  All
coordinates and densities are normalized to ``[0, 1]``.  Invalid rows contain
zeros.  The occupancy and motion grids also contain values in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


DEFAULT_GRID_SHAPE: Final[tuple[int, int]] = (12, 16)
ROAD_ROWS: Final[int] = 12
ROAD_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "left",
    "right",
    "center",
    "width",
    "density",
    "valid",
)
ROAD_FEATURE_COUNT: Final[int] = len(ROAD_FEATURE_NAMES)


def feature_dimension(
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    road_rows: int = ROAD_ROWS,
) -> int:
    """Return the deterministic flat-vector length for an extractor layout."""

    grid_rows, grid_columns = _validate_grid_shape(grid_shape)
    if road_rows < 1:
        raise ValueError("road_rows must be positive")
    pooled_values = grid_rows * grid_columns
    return pooled_values * 3 + road_rows * ROAD_FEATURE_COUNT + 1


DEFAULT_FEATURE_DIM: Final[int] = (
    DEFAULT_GRID_SHAPE[0] * DEFAULT_GRID_SHAPE[1] * 3
    + ROAD_ROWS * ROAD_FEATURE_COUNT
    + 1
)


@dataclass(frozen=True)
class FrameFeatures:
    """Visual measurements from one frame and an optional preceding frame.

    ``green_mask`` and ``red_mask`` are full-resolution uint8 masks containing
    0 or 255.  The three pooled grids and ``road_geometry`` are float32.  They
    are retained separately for diagnostics and are flattened by
    :meth:`as_vector` in the module's documented order.
    """

    green_mask: np.ndarray
    red_mask: np.ndarray
    green_occupancy: np.ndarray
    red_occupancy: np.ndarray
    motion: np.ndarray
    road_geometry: np.ndarray
    game_over: bool
    game_over_confidence: float
    green_fraction: float
    red_fraction: float

    def as_vector(self) -> np.ndarray:
        """Return a new, C-contiguous, normalized float32 feature vector."""

        parts = (
            self.green_occupancy.reshape(-1),
            self.red_occupancy.reshape(-1),
            self.motion.reshape(-1),
            self.road_geometry.reshape(-1),
            np.asarray((float(self.game_over),), dtype=np.float32),
        )
        vector = np.concatenate(parts).astype(np.float32, copy=False)
        # Numerical interpolation can very slightly exceed its mathematical
        # bounds on some OpenCV builds.  Keeping this guarantee here makes the
        # observation-space contract exact.
        return np.ascontiguousarray(np.clip(vector, 0.0, 1.0), dtype=np.float32)

    @property
    def vector(self) -> np.ndarray:
        """Alias for :meth:`as_vector`, convenient for Gym observations."""

        return self.as_vector()


def hsv_colour_masks(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return action-neutral green and red masks for a BGR game frame.

    The green range intentionally spans yellow-green through cyan-green.  Red
    wraps around zero in HSV, so it is represented by two ranges.  Small
    isolated compression speckles are removed without joining distinct scene
    objects.
    """

    frame = _validate_frame(frame_bgr, "frame_bgr")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, (32, 70, 40), (100, 255, 255))
    red_low = cv2.inRange(hsv, (0, 100, 55), (13, 255, 255))
    red_high = cv2.inRange(hsv, (167, 100, 55), (179, 255, 255))
    red = cv2.bitwise_or(red_low, red_high)

    # A square morphological opening destroys the real retry UI's one-pixel
    # outline.  Area filtering removes isolated JPEG speckles while preserving
    # long thin road/UI lines and compact text glyphs.
    return _remove_tiny_components(green), _remove_tiny_components(red)


def pooled_occupancy(
    mask: np.ndarray,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> np.ndarray:
    """Pool a binary mask into a fixed grid of occupied-pixel fractions."""

    grid_rows, grid_columns = _validate_grid_shape(grid_shape)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    binary = (mask > 0).astype(np.float32)
    pooled = cv2.resize(
        binary,
        (grid_columns, grid_rows),
        interpolation=cv2.INTER_AREA,
    )
    return np.clip(pooled, 0.0, 1.0).astype(np.float32, copy=False)


def pooled_frame_difference(
    frame_bgr: np.ndarray,
    previous_frame_bgr: np.ndarray | None,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> np.ndarray:
    """Return pooled absolute grayscale change, normalized to ``[0, 1]``.

    A missing previous frame represents an episode boundary and yields a zero
    motion grid.  Shape mismatches are rejected instead of silently resizing a
    frame, which would manufacture motion.
    """

    frame = _validate_frame(frame_bgr, "frame_bgr")
    grid_rows, grid_columns = _validate_grid_shape(grid_shape)
    if previous_frame_bgr is None:
        return np.zeros((grid_rows, grid_columns), dtype=np.float32)

    previous = _validate_frame(previous_frame_bgr, "previous_frame_bgr")
    if previous.shape != frame.shape:
        raise ValueError("current and previous frames must have identical shapes")
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(current_gray, previous_gray).astype(np.float32) / 255.0
    pooled = cv2.resize(
        difference,
        (grid_columns, grid_rows),
        interpolation=cv2.INTER_AREA,
    )
    return np.clip(pooled, 0.0, 1.0).astype(np.float32, copy=False)


def road_geometry(green_mask: np.ndarray, rows: int = ROAD_ROWS) -> np.ndarray:
    """Measure raw green-road extent at fixed look-ahead rows.

    The sampled vertical range runs from 28% to 96% of the frame height.  For
    each band, columns containing a meaningful amount of green define the left
    and right observed extents.  This is deliberately geometry, not a chosen
    route or steering target.
    """

    if not isinstance(green_mask, np.ndarray) or green_mask.ndim != 2:
        raise ValueError("green_mask must be a two-dimensional array")
    if green_mask.size == 0:
        raise ValueError("green_mask cannot be empty")
    if rows < 1:
        raise ValueError("rows must be positive")

    height, width = green_mask.shape
    geometry = np.zeros((rows, ROAD_FEATURE_COUNT), dtype=np.float32)
    # Partitioning bands, rather than sampling individual scan lines, makes the
    # measurement stable under JPEG noise and one-pixel grid-line movement.
    boundaries = np.linspace(0.28 * height, 0.96 * height, rows + 1)
    denominator = float(max(1, width - 1))

    for index in range(rows):
        top = max(0, min(height - 1, int(np.floor(boundaries[index]))))
        bottom = max(top + 1, min(height, int(np.ceil(boundaries[index + 1]))))
        band = green_mask[top:bottom] > 0
        band_height = band.shape[0]
        column_counts = np.count_nonzero(band, axis=0)
        # Require either two pixels in a band or 12.5% vertical occupancy.  The
        # latter scales sensibly for both tiny tests and full game captures.
        threshold = max(1, min(2, int(np.ceil(band_height * 0.125))))
        columns = np.flatnonzero(column_counts >= threshold)
        if columns.size < 2:
            continue

        left = int(columns[0])
        right = int(columns[-1])
        span = right - left
        if span < max(2, round(width * 0.02)):
            continue
        left_normalized = left / denominator
        right_normalized = right / denominator
        geometry[index] = (
            left_normalized,
            right_normalized,
            (left_normalized + right_normalized) * 0.5,
            span / denominator,
            float(np.count_nonzero(band)) / float(band.size),
            1.0,
        )

    return np.clip(geometry, 0.0, 1.0).astype(np.float32, copy=False)


def detect_game_over(
    green_mask: np.ndarray,
    frame_bgr: np.ndarray | None = None,
) -> bool:
    """Detect a visible Slope retry button, while rejecting ad cards.

    Motion is intentionally irrelevant.  Two generic horizontal green lines
    are not enough: active road crossbars regularly produce that pattern.  A
    positive match requires the axis-aligned border and interior glyphs of the
    low, centered retry button used by Slope's death UI.
    """

    return game_over_confidence(green_mask, frame_bgr) >= 0.5


def game_over_confidence(
    green_mask: np.ndarray,
    frame_bgr: np.ndarray | None = None,
) -> float:
    """Return conservative visual evidence for Slope's death/retry UI.

    When the original BGR frame is supplied, a large bright central card vetoes
    the result.  Y8 advertisements leave the green road visible behind a white
    or pale card, so classifying only the green layer is unsafe.
    """

    _validate_mask(green_mask, "green_mask")
    if frame_bgr is not None:
        frame = _validate_frame(frame_bgr, "frame_bgr")
        if frame.shape[:2] != green_mask.shape:
            raise ValueError("frame_bgr and green_mask must have identical dimensions")
        if _has_large_bright_overlay(frame):
            return 0.0
        # Hovering Slope's AGAIN control can switch its outline/text from green
        # to red.  Use a deliberately low value floor for this UI-only mask so
        # dim one-pixel borders survive; the normal observation masks retain
        # their stricter thresholds.
        return _retry_button_confidence(_ui_accent_mask(frame))
    return _retry_button_confidence(green_mask)


def _retry_button_confidence(green_mask: np.ndarray) -> float:
    """Find an outlined, labelled retry control in the bottom screen band."""

    height, width = green_mask.shape
    y0 = int(height * 0.70)
    y1 = min(height, int(np.ceil(height * 0.99)))
    binary = green_mask > 0
    minimum_run = max(9, round(width * 0.18))
    lines: list[tuple[float, int, int]] = []
    for y in range(y0, y1):
        padded = np.pad(binary[y].astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        for left, right in zip(starts, ends, strict=True):
            if right - left + 1 >= minimum_run:
                lines.append((float(y), int(left), int(right)))
    if len(lines) < 2:
        return 0.0

    minimum_gap = max(6.0, height * 0.045)
    maximum_gap = max(minimum_gap, height * 0.16)
    side_radius = max(2, round(width * 0.008))
    best_confidence = 0.0

    for upper_index, (upper_y, upper_left, upper_right) in enumerate(lines):
        for lower_y, lower_left, lower_right in lines[upper_index + 1 :]:
            gap = lower_y - upper_y
            if not minimum_gap <= gap <= maximum_gap:
                continue
            upper_width = upper_right - upper_left + 1
            lower_width = lower_right - lower_left + 1
            overlap = min(upper_right, lower_right) - max(upper_left, lower_left) + 1
            if overlap <= 0 or overlap / min(upper_width, lower_width) < 0.78:
                continue

            # Scenery can touch and extend one border (as it does in all three
            # captured death frames).  The narrower run still gives the exact
            # button endpoints, so use it as the rectangle reference.
            if upper_width <= lower_width:
                left, right, reference_width = upper_left, upper_right, upper_width
            else:
                left, right, reference_width = lower_left, lower_right, lower_width
            center_x = (left + right) * 0.5
            if (
                not width * 0.24 <= reference_width <= width * 0.62
                or not width * 0.35 <= center_x <= width * 0.65
            ):
                continue
            top = max(0, round(upper_y))
            bottom = min(height - 1, round(lower_y))
            if bottom < height * 0.86:
                continue

            left_strip = binary[
                top : bottom + 1,
                max(0, left - side_radius) : min(width, left + side_radius + 1),
            ]
            right_strip = binary[
                top : bottom + 1,
                max(0, right - side_radius) : min(width, right + side_radius + 1),
            ]
            left_support = float(np.mean(np.any(left_strip, axis=1)))
            right_support = float(np.mean(np.any(right_strip, axis=1)))
            side_support = min(left_support, right_support)
            if side_support < 0.75:
                continue

            inset_x = max(3, side_radius * 2)
            inset_y = max(3, round(height * 0.008))
            interior = binary[
                min(bottom, top + inset_y) : max(top + inset_y + 1, bottom - inset_y),
                min(right, left + inset_x) : max(left + inset_x + 1, right - inset_x),
            ]
            if interior.size == 0:
                continue
            glyph_fraction = float(np.mean(interior))
            # A blank frame is too generic; the retry button must contain its
            # green label.  Conversely, a mostly filled region is scenery.
            if not 0.004 <= glyph_fraction <= 0.42:
                continue

            width_strength = np.clip(
                min(upper_width, lower_width) / max(1.0, width * 0.36), 0.0, 1.0
            )
            glyph_strength = np.clip(glyph_fraction / 0.06, 0.0, 1.0)
            confidence = (
                0.5
                + 0.22 * side_support
                + 0.14 * float(width_strength)
                + 0.14 * float(glyph_strength)
            )
            best_confidence = max(best_confidence, float(confidence))

    return float(np.clip(best_confidence, 0.0, 1.0))


def _remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    """Remove isolated colour noise without erasing one-pixel-long UI lines."""

    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, component_count):
        if statistics[label, cv2.CC_STAT_AREA] >= 3:
            cleaned[labels == label] = 255
    return cleaned


def _has_large_bright_overlay(frame_bgr: np.ndarray) -> bool:
    """Recognize the large pale cards used by the observed Y8 interstitials."""

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bright_neutral = (hsv[:, :, 2] >= 150) & (hsv[:, :, 1] <= 75)
    height, width = bright_neutral.shape
    y0, y1 = int(height * 0.08), int(np.ceil(height * 0.90))
    x0, x1 = int(width * 0.08), int(np.ceil(width * 0.92))
    central = bright_neutral[y0:y1, x0:x1]
    if central.size == 0:
        return False

    fraction = float(np.mean(central))
    longest_row = float(np.max(np.mean(central, axis=1)))
    longest_column = float(np.max(np.mean(central, axis=0)))
    return fraction >= 0.07 and longest_row >= 0.45 and longest_column >= 0.20


def _ui_accent_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return dim green-or-red pixels used by outlined Slope UI controls."""

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (32, 70, 10), (100, 255, 255))
    red_low = cv2.inRange(hsv, (0, 100, 10), (13, 255, 255))
    red_high = cv2.inRange(hsv, (167, 100, 10), (179, 255, 255))
    accent = cv2.bitwise_or(green, cv2.bitwise_or(red_low, red_high))
    return _remove_tiny_components(accent)


def extract_frame_features(
    frame_bgr: np.ndarray,
    previous_frame_bgr: np.ndarray | None = None,
    *,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    geometry_rows: int = ROAD_ROWS,
) -> FrameFeatures:
    """Extract all visual features without policy state or action input."""

    frame = _validate_frame(frame_bgr, "frame_bgr")
    green_mask, red_mask = hsv_colour_masks(frame)
    death_confidence = game_over_confidence(green_mask, frame)
    return FrameFeatures(
        green_mask=green_mask,
        red_mask=red_mask,
        green_occupancy=pooled_occupancy(green_mask, grid_shape),
        red_occupancy=pooled_occupancy(red_mask, grid_shape),
        motion=pooled_frame_difference(frame, previous_frame_bgr, grid_shape),
        road_geometry=road_geometry(green_mask, geometry_rows),
        game_over=death_confidence >= 0.5,
        game_over_confidence=death_confidence,
        green_fraction=float(np.count_nonzero(green_mask)) / float(green_mask.size),
        red_fraction=float(np.count_nonzero(red_mask)) / float(red_mask.size),
    )


class VisionEncoder:
    """Configured façade for the stateless frame-feature functions.

    The object stores layout dimensions only.  It never stores a prior frame or
    an action, so calling :meth:`reset` is intentionally a no-op and identical
    inputs always produce identical outputs.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
        road_rows: int = ROAD_ROWS,
    ) -> None:
        self.grid_shape = _validate_grid_shape(grid_shape)
        if road_rows < 1:
            raise ValueError("road_rows must be positive")
        self.road_rows = int(road_rows)

    @property
    def feature_dim(self) -> int:
        """Length of vectors produced by this encoder."""

        return feature_dimension(self.grid_shape, self.road_rows)

    def reset(self) -> None:
        """Reset temporal state (there is none; provided for environment APIs)."""

    def encode(
        self,
        frame: np.ndarray,
        previous_frame: np.ndarray | None = None,
    ) -> FrameFeatures:
        """Encode a BGR frame and an explicitly supplied previous frame."""

        return extract_frame_features(
            frame,
            previous_frame,
            grid_shape=self.grid_shape,
            geometry_rows=self.road_rows,
        )


def _validate_frame(frame: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 BGR pixels")
    return frame


def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    return mask


def _validate_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain rows and columns")
    rows, columns = int(grid_shape[0]), int(grid_shape[1])
    if rows < 1 or columns < 1:
        raise ValueError("grid dimensions must be positive")
    return rows, columns


__all__ = [
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_GRID_SHAPE",
    "FrameFeatures",
    "ROAD_FEATURE_COUNT",
    "ROAD_FEATURE_NAMES",
    "ROAD_ROWS",
    "VisionEncoder",
    "detect_game_over",
    "extract_frame_features",
    "feature_dimension",
    "game_over_confidence",
    "hsv_colour_masks",
    "pooled_frame_difference",
    "pooled_occupancy",
    "road_geometry",
]
