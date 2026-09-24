"""Deterministic, action-neutral visual observations for Slope.

Slope draws everything in two colours on black: the road, ball, buildings, and
text are green, and deadly walls and blocks are red.  Each frame is therefore
shrunk to a small two-channel image holding its red and green intensities, and
the network learns the road, ball, and obstacles from those pixels itself.
Hand-measured road geometry could not tell the road from green tunnels,
arches, and buildings.

The flat vector for one frame is the ``FRAME_SHAPE`` image in channel, row,
column order, scaled to ``[0, 1]``::

    red   40 * 64 = 2560
    green 40 * 64 = 2560
                   -----
                    5120 values

The extractor has no state; motion is visible to the model through the
environment's frame history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np


# (channels, height, width).  The game canvas is 16:10, so 64x40 keeps its
# aspect; the ball is still about six pixels wide at this size.
FRAME_SHAPE: Final[tuple[int, int, int]] = (2, 40, 64)
DEFAULT_FEATURE_DIM: Final[int] = FRAME_SHAPE[0] * FRAME_SHAPE[1] * FRAME_SHAPE[2]


@dataclass(frozen=True)
class FrameFeatures:
    """The model image and death-screen evidence for one frame.

    ``pixels`` is a uint8 ``FRAME_SHAPE`` array of red and green intensity.
    """

    pixels: np.ndarray
    game_over: bool
    game_over_confidence: float

    def as_vector(self) -> np.ndarray:
        """Return a new, C-contiguous, normalized float32 feature vector."""

        vector = self.pixels.reshape(-1).astype(np.float32) / 255.0
        return np.ascontiguousarray(vector, dtype=np.float32)

    @property
    def vector(self) -> np.ndarray:
        """Alias for :meth:`as_vector`, convenient for Gym observations."""

        return self.as_vector()


def frame_pixels(frame_bgr: np.ndarray) -> np.ndarray:
    """Shrink a BGR frame to the model's red/green ``FRAME_SHAPE`` image.

    Area averaging keeps thin road lines and distant red blocks as dimmer
    pixels instead of dropping them.  Raw channel intensity also keeps the
    difference between bright road tiles and the darker green buildings.
    """

    frame = _validate_frame(frame_bgr, "frame_bgr")
    _, height, width = FRAME_SHAPE
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small[:, :, (2, 1)].transpose(2, 0, 1))


GAME_OVER_THRESHOLD: Final[float] = 0.50

# Slope's death screen always shows three outlined controls at fixed canvas
# fractions: AGAIN, Menu, and Leaderboard.  Each entry is (template name,
# exact control box, search window) as (x0, y0, x1, y1) frame fractions.  The
# window leaves a few pixels of slack for canvas rounding.
_DEATH_UI_CONTROLS: Final[tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]] = (
    ("again", (0.296, 0.850, 0.702, 0.960), (0.25, 0.80, 0.75, 1.0)),
    ("menu", (0.016, 0.850, 0.212, 0.960), (0.0, 0.80, 0.26, 1.0)),
    ("leader", (0.340, 0.675, 0.657, 0.785), (0.29, 0.63, 0.71, 0.83)),
)
# Crops are resampled as if the canvas were this size, so one set of templates
# serves every capture width.
_TEMPLATE_CANVAS: Final[tuple[int, int]] = (320, 200)
_TEMPLATE_PATH: Final[Path] = Path(__file__).with_name("assets") / "death_ui.npz"
_death_ui_templates: dict[str, np.ndarray] | None = None


def _load_death_ui_templates() -> dict[str, np.ndarray]:
    global _death_ui_templates
    if _death_ui_templates is None:
        with np.load(_TEMPLATE_PATH) as stored:
            _death_ui_templates = {
                name: stored[name].astype(np.float32) / 255.0
                for name, _, _ in _DEATH_UI_CONTROLS
            }
    return _death_ui_templates


def _ui_intensity(frame_bgr: np.ndarray, box: tuple[float, ...]) -> np.ndarray:
    """Brightness of saturated pixels in ``box``, at template scale.

    Using saturated brightness rather than hue keeps the match valid when a
    hovered control switches its outline from green to red.
    """

    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box
    crop = frame_bgr[
        round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)
    ]
    size = (
        round((x1 - x0) * _TEMPLATE_CANVAS[0]),
        round((y1 - y0) * _TEMPLATE_CANVAS[1]),
    )
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32) / 255.0
    return value * (hsv[:, :, 1] >= 60)


def detect_game_over(frame_bgr: np.ndarray) -> bool:
    """Return whether Slope's death screen is visible in a BGR frame."""

    return game_over_confidence(frame_bgr) >= GAME_OVER_THRESHOLD


def game_over_confidence(frame_bgr: np.ndarray) -> float:
    """Template-correlation evidence for Slope's death screen, in ``[0, 1]``.

    The templates are medians of real captured death screens, so scenery
    behind the opaque controls does not appear in them.  AGAIN and Menu exist
    only on the death screen, while Leaderboard also appears on the main menu;
    requiring both a death-only control and Leaderboard rejects road chevrons
    that resemble a single outlined box.  On 4,586 real 480- and 640-pixel
    captures, live frames scored at most 0.29, the main menu 0.33, and death
    screens at least 0.69.
    """

    frame = _validate_frame(frame_bgr, "frame_bgr")
    templates = _load_death_ui_templates()
    scores: dict[str, float] = {}
    for name, _, window in _DEATH_UI_CONTROLS:
        search = _ui_intensity(frame, window)
        template = templates[name]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            return 0.0
        correlation = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        scores[name] = float(np.nan_to_num(correlation).max())
    death_only = (scores["again"] + scores["menu"]) / 2.0
    return float(np.clip(min(death_only, scores["leader"]), 0.0, 1.0))




def extract_frame_features(frame_bgr: np.ndarray) -> FrameFeatures:
    """Extract the model image and death-screen evidence for one frame."""

    frame = _validate_frame(frame_bgr, "frame_bgr")
    death_confidence = game_over_confidence(frame)
    return FrameFeatures(
        pixels=frame_pixels(frame),
        game_over=death_confidence >= GAME_OVER_THRESHOLD,
        game_over_confidence=death_confidence,
    )


class VisionEncoder:
    """Configured façade for the stateless frame-feature functions.

    It never stores a prior frame or an action, so calling :meth:`reset` is
    intentionally a no-op and identical inputs always produce identical
    outputs.
    """

    @property
    def feature_dim(self) -> int:
        """Length of vectors produced by this encoder."""

        return DEFAULT_FEATURE_DIM

    def reset(self) -> None:
        """Reset temporal state (there is none; provided for environment APIs)."""

    def encode(self, frame: np.ndarray) -> FrameFeatures:
        """Encode one BGR frame."""

        return extract_frame_features(frame)


def _validate_frame(frame: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 BGR pixels")
    return frame



__all__ = [
    "DEFAULT_FEATURE_DIM",
    "FRAME_SHAPE",
    "GAME_OVER_THRESHOLD",
    "FrameFeatures",
    "VisionEncoder",
    "detect_game_over",
    "extract_frame_features",
    "frame_pixels",
    "game_over_confidence",
]
