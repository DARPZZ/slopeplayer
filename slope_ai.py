"""A training-free, screen-reading controller for browser versions of Slope.

The program deliberately does not inject JavaScript into the game.  It captures
the selected part of the screen, estimates the track direction and nearby red
obstacles, and sends ordinary left/right key presses to the focused browser.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
    import mss
    import numpy as np
    from pynput import keyboard
except ImportError as exc:  # Give a more useful error than a raw traceback.
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing {missing}. Install the dependencies with:\n"
        f"  {sys.executable} -m pip install -r requirements.txt"
    ) from exc


CONFIG_PATH = Path(__file__).with_name("slope_ai_config.json")


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class VisionResult:
    ball: tuple[int, int]
    target_x: float
    raw_target_x: float
    obstacle: tuple[int, int, int, int] | None
    confidence: float
    green_mask: Any
    red_mask: Any


class Perception:
    """Extract a steering target from the game's neon green/red graphics."""

    def __init__(self) -> None:
        self._smoothed_target: float | None = None
        self._last_ball: tuple[int, int] | None = None
        self._avoid_direction = 0
        self._avoid_until = 0.0

    @staticmethod
    def _colour_masks(frame: Any) -> tuple[Any, Any]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Slope clones vary from yellow-green to cyan-green. Saturation removes
        # most white UI text and the value floor removes the black background.
        green = cv2.inRange(hsv, (32, 75, 45), (100, 255, 255))
        red_a = cv2.inRange(hsv, (0, 105, 65), (12, 255, 255))
        red_b = cv2.inRange(hsv, (168, 105, 65), (179, 255, 255))
        red = cv2.bitwise_or(red_a, red_b)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return green, red

    def _find_ball(self, green: Any) -> tuple[int, int]:
        h, w = green.shape
        joined = cv2.morphologyEx(
            green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: tuple[float, tuple[int, int]] | None = None

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            if area < max(35.0, w * h * 0.000035) or area > w * h * 0.025:
                continue
            if y + ch / 2 < h * 0.48 or cw == 0 or ch == 0:
                continue
            aspect = cw / ch
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-6)
            fill = area / (cw * ch)
            if not 0.62 <= aspect <= 1.55 or circularity < 0.40 or fill < 0.42:
                continue
            cx, cy = x + cw // 2, y + ch // 2
            continuity = 0.0
            if self._last_ball is not None:
                distance = math.dist((cx, cy), self._last_ball) / max(w, h)
                continuity = max(0.0, 1.0 - distance * 7.0)
            score = circularity + fill * 0.5 + cy / h * 0.25 + continuity
            if best is None or score > best[0]:
                best = (score, (cx, cy))

        # The camera normally keeps the ball near this location. Falling back
        # here lets steering continue across frames where a grid line hides it.
        ball = best[1] if best else (w // 2, int(h * 0.79))
        if best or self._last_ball is None:
            self._last_ball = ball
        elif self._last_ball is not None:
            ball = self._last_ball
        return ball

    @staticmethod
    def _band_centre(
        mask: Any, y0: int, y1: int, around: float, radius: float
    ) -> tuple[float | None, float]:
        h, w = mask.shape
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, int(around - radius)), min(w, int(around + radius))
        if y1 <= y0 or x1 <= x0:
            return None, 0.0
        band = mask[y0:y1, x0:x1]
        counts = np.count_nonzero(band, axis=0).astype(np.float32)
        # Ignore isolated sparkles while preserving thin perspective grid lines.
        threshold = max(1.0, (y1 - y0) * 0.018)
        counts[counts < threshold] = 0
        total = float(counts.sum())
        if total <= 0:
            return None, 0.0
        xs = np.arange(x0, x1, dtype=np.float32)
        centre = float(np.dot(xs, counts) / total)
        confidence = min(1.0, total / max(1.0, (y1 - y0) * w * 0.10))
        return centre, confidence

    def _track_target(self, green: Any, ball: tuple[int, int]) -> tuple[float, float]:
        h, w = green.shape
        # Follow the road from near the ball toward the horizon. Closer bands
        # matter more, while the far band helps anticipate turns.
        bands = ((0.64, 0.74, 0.48), (0.52, 0.64, 0.32), (0.40, 0.52, 0.20))
        centre = float(ball[0])
        weighted_sum = 0.0
        weight_total = 0.0
        confidence_total = 0.0
        for top, bottom, weight in bands:
            found, confidence = self._band_centre(
                green, int(h * top), int(h * bottom), centre, w * 0.40
            )
            if found is not None:
                centre = centre * 0.35 + found * 0.65
                effective_weight = weight * (0.30 + confidence)
                weighted_sum += centre * effective_weight
                weight_total += effective_weight
                confidence_total += confidence * weight
        if weight_total == 0:
            return float(ball[0]), 0.0
        return weighted_sum / weight_total, min(1.0, confidence_total)

    def _avoid_obstacle(
        self, red: Any, ball: tuple[int, int], target: float, now: float
    ) -> tuple[float, tuple[int, int, int, int] | None]:
        h, w = red.shape
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        threats: list[tuple[float, tuple[int, int, int, int], float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            cx, cy = x + cw / 2, y + ch / 2
            if area < w * h * 0.00008 or cy < h * 0.34 or cy > ball[1] + h * 0.03:
                continue
            path_x = target + (ball[0] - target) * ((cy - h * 0.34) / max(1, ball[1] - h * 0.34))
            clearance = max(w * 0.075, cw * 0.75)
            distance = abs(cx - path_x)
            if distance < clearance:
                proximity = cy / h + (1.0 - distance / clearance)
                threats.append((proximity, (x, y, cw, ch), cx))

        obstacle = max(threats, default=None, key=lambda item: item[0])
        if obstacle is not None:
            _, box, obstacle_x = obstacle
            if now >= self._avoid_until:
                if abs(obstacle_x - target) < w * 0.025:
                    self._avoid_direction = -1 if target >= w / 2 else 1
                else:
                    self._avoid_direction = -1 if obstacle_x > target else 1
            self._avoid_until = now + 0.30
            return target + self._avoid_direction * w * 0.16, box
        if now < self._avoid_until:
            return target + self._avoid_direction * w * 0.11, None
        self._avoid_direction = 0
        return target, None

    def analyse(self, frame: Any, now: float | None = None) -> VisionResult:
        now = time.monotonic() if now is None else now
        green, red = self._colour_masks(frame)
        ball = self._find_ball(green)
        raw_target, confidence = self._track_target(green, ball)
        raw_target, obstacle = self._avoid_obstacle(red, ball, raw_target, now)
        raw_target = float(np.clip(raw_target, frame.shape[1] * 0.08, frame.shape[1] * 0.92))
        if self._smoothed_target is None:
            self._smoothed_target = raw_target
        else:
            self._smoothed_target = self._smoothed_target * 0.70 + raw_target * 0.30
        return VisionResult(
            ball=ball,
            target_x=self._smoothed_target,
            raw_target_x=raw_target,
            obstacle=obstacle,
            confidence=confidence,
            green_mask=green,
            red_mask=red,
        )


class KeyController:
    def __init__(self, layout: str) -> None:
        self.keyboard = keyboard.Controller()
        self.left = keyboard.Key.left if layout == "arrows" else "a"
        self.right = keyboard.Key.right if layout == "arrows" else "d"
        self.direction = 0

    def set_direction(self, direction: int) -> None:
        direction = max(-1, min(1, direction))
        if direction == self.direction:
            return
        self.release()
        if direction < 0:
            self.keyboard.press(self.left)
        elif direction > 0:
            self.keyboard.press(self.right)
        self.direction = direction

    def release(self) -> None:
        # Releasing keys that are already up is harmless and makes shutdown safe.
        self.keyboard.release(self.left)
        self.keyboard.release(self.right)
        self.direction = 0


class SlopeBot:
    def __init__(self, region: Region, fps: int, layout: str, preview: bool) -> None:
        self.region = region
        self.frame_period = 1.0 / fps
        self.preview = preview
        self.perception = Perception()
        self.keys = KeyController(layout)
        self.running = threading.Event()
        self.stopping = threading.Event()
        self.previous_error = 0.0
        self.filtered_turn = 0.0

    def toggle(self) -> None:
        if self.running.is_set():
            self.running.clear()
            self.keys.release()
            print("Paused — keys released. Press F8 to resume.")
        else:
            self.running.set()
            print("Playing. Keep the browser focused; press F8 to pause.")

    def stop(self) -> None:
        self.stopping.set()
        self.running.clear()
        self.keys.release()

    def _hotkey(self, key: Any) -> bool | None:
        if key == keyboard.Key.f8:
            self.toggle()
        elif key == keyboard.Key.f9:
            self.stop()
            return False
        return None

    def _steer(self, result: VisionResult, width: int) -> int:
        error = (result.target_x - result.ball[0]) / max(1.0, width * 0.5)
        derivative = error - self.previous_error
        self.previous_error = error
        desired = 1.55 * error + 0.42 * derivative
        self.filtered_turn = self.filtered_turn * 0.55 + desired * 0.45

        # Low-confidence frames use a wider dead zone to avoid random twitching.
        dead_zone = 0.050 if result.confidence > 0.08 else 0.085
        if self.filtered_turn < -dead_zone:
            return -1
        if self.filtered_turn > dead_zone:
            return 1
        return 0

    @staticmethod
    def _draw_preview(frame: Any, result: VisionResult, direction: int, active: bool) -> Any:
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        bx, by = result.ball
        target = int(result.target_x)
        cv2.circle(canvas, (bx, by), max(7, w // 90), (255, 255, 255), 2)
        cv2.line(canvas, (bx, by), (target, int(h * 0.43)), (0, 255, 255), 3)
        if result.obstacle:
            x, y, cw, ch = result.obstacle
            cv2.rectangle(canvas, (x, y), (x + cw, y + ch), (255, 0, 255), 3)
        label = "LEFT" if direction < 0 else "RIGHT" if direction > 0 else "STRAIGHT"
        state = "RUNNING" if active else "PAUSED"
        cv2.putText(
            canvas,
            f"{state} | {label} | confidence {result.confidence:.2f}",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def run(self) -> None:
        listener = keyboard.Listener(on_press=self._hotkey)
        listener.start()
        print("Ready. Focus the browser, then press F8 to play/pause. Press F9 to quit.")
        try:
            with mss.mss() as capture:
                while not self.stopping.is_set():
                    started = time.perf_counter()
                    shot = capture.grab(self.region.as_mss())
                    frame = np.asarray(shot)[:, :, :3]
                    result = self.perception.analyse(frame)
                    active = self.running.is_set()
                    direction = self._steer(result, frame.shape[1]) if active else 0
                    self.keys.set_direction(direction if active else 0)

                    if self.preview:
                        cv2.imshow(
                            "Slope AI preview (do not focus this window while playing)",
                            self._draw_preview(frame, result, direction, active),
                        )
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            self.stop()
                    elapsed = time.perf_counter() - started
                    if elapsed < self.frame_period:
                        time.sleep(self.frame_period - elapsed)
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.keys.release()
            listener.stop()
            cv2.destroyAllWindows()
            print("Stopped — keys released.")


def select_region() -> Region:
    print("Taking a screenshot. Draw a box around only the game canvas, then press Enter.")
    with mss.mss() as capture:
        virtual = capture.monitors[0]
        screenshot = np.asarray(capture.grab(virtual))[:, :, :3]
    x, y, width, height = cv2.selectROI(
        "Select the Slope game area", screenshot, showCrosshair=True, fromCenter=False
    )
    cv2.destroyAllWindows()
    if width < 200 or height < 200:
        raise SystemExit("Selection cancelled or too small; no configuration was saved.")
    return Region(
        left=int(virtual["left"] + x),
        top=int(virtual["top"] + y),
        width=int(width),
        height=int(height),
    )


def save_config(region: Region, path: Path = CONFIG_PATH) -> None:
    path.write_text(json.dumps(region.__dict__, indent=2) + "\n", encoding="utf-8")
    print(f"Saved game area to {path}")


def load_config(path: Path = CONFIG_PATH) -> Region:
    if not path.exists():
        raise SystemExit(
            "No game area is configured yet. Run `python slope_ai.py --setup` first."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        region = Region(**{key: int(data[key]) for key in ("left", "top", "width", "height")})
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if region.width < 200 or region.height < 200:
        raise SystemExit("The saved game area is too small. Run setup again.")
    return region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a browser-based Slope game using screen vision (no training needed)."
    )
    parser.add_argument(
        "--setup", action="store_true", help="interactively select and save the game area"
    )
    parser.add_argument(
        "--preview", action="store_true", help="show what the controller sees"
    )
    parser.add_argument(
        "--keys", choices=("arrows", "ad"), default="arrows", help="game steering keys"
    )
    parser.add_argument("--fps", type=int, default=30, help="capture rate (10-60; default 30)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.setup:
        save_config(select_region())
        return
    if not 10 <= args.fps <= 60:
        raise SystemExit("--fps must be between 10 and 60")
    SlopeBot(load_config(), args.fps, args.keys, args.preview).run()


if __name__ == "__main__":
    main()
