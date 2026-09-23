"""Deterministic Playwright transport for the browser version of Slope.

The browser owns exactly one page and exposes only the operations needed by a
reinforcement-learning environment: reset, fixed-duration steering, capture,
and close.  Game time remains paused between calls.  Advancing the game is
always done in display-sized slices so training and visible playback use the
same physics schedule.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np


Y8_EMBED_URL = "https://www.y8.com/embed/{slug}"
_PHYSICS_SLICE_MS = 16
_RESET_PHYSICS_FRAMES = 3
_RESET_POLL_MS = _RESET_PHYSICS_FRAMES * _PHYSICS_SLICE_MS
_RESET_MAX_TRANSITION_MS = 2_000
_MENU_POLL_MS = 64
_AD_POLL_MS = 250
_AD_AFTER_CLICK_MS = _RESET_POLL_MS
_AD_MAX_WAIT_MS = 30_000
_RESET_REASONS = frozenset({"initial", "terminated", "truncated", "manual"})

# Y8 occasionally attaches pop-under handlers to the game page or one of its
# advertising frames.  Playwright's page event lets us close the resulting tab,
# but preventing it in the first place avoids a visible flash in headed mode and
# avoids a short-lived extra renderer in headless training.  Context init scripts
# run in the owner page and every child frame before site JavaScript executes.
_POPUP_GUARD_SCRIPT = """
(() => {
    window.open = () => null;
    document.addEventListener("click", (event) => {
        const target = event.target;
        const anchor = target instanceof Element ? target.closest("a") : null;
        if (anchor && anchor.target.toLowerCase() === "_blank") {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, true);
})();
"""

_AD_CLOSE_SELECTORS = (
    "button:has-text('Close')",
    "[role='button']:has-text('Close')",
    "text=/^\\s*Close\\s*$/i",
    "button[aria-label*='close' i]",
    "[role='button'][aria-label*='close' i]",
    "[title*='close' i]",
)

_AD_GENERIC_CLOSE_SELECTORS = (
    "[id*='close' i]",
    "[class*='close' i]",
)


class BrowserError(RuntimeError):
    """Raised when the game browser cannot provide a valid transition."""


# Playwright's sync API permits only one started instance per thread; a second
# ``sync_playwright().start()`` fails while the first is running.  Training and
# its periodic evaluation each own a browser, so both launch from one shared,
# reference-counted driver that stops only after the last browser closes.
_shared_playwright: Any | None = None
_shared_playwright_users = 0


def _acquire_playwright() -> Any:
    global _shared_playwright, _shared_playwright_users
    if _shared_playwright is None:
        from playwright.sync_api import sync_playwright

        _shared_playwright = sync_playwright().start()
        _shared_playwright_users = 0
    _shared_playwright_users += 1
    return _shared_playwright


def _release_playwright(instance: Any) -> None:
    global _shared_playwright, _shared_playwright_users
    if instance is not _shared_playwright:
        # Not the shared driver (already stopped, or supplied directly).
        instance.stop()
        return
    _shared_playwright_users -= 1
    if _shared_playwright_users <= 0:
        _shared_playwright = None
        _shared_playwright_users = 0
        instance.stop()


def normalize_game_url(url: str) -> str:
    """Convert a public/localized Y8 game page to its stable embed URL."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    is_y8 = host == "y8.com" or host.endswith(".y8.com")
    if is_y8 and len(parts) == 2 and parts[0] in {"games", "embed"}:
        slug = parts[1]
        if slug and all(character.isalnum() or character in "_-" for character in slug):
            return Y8_EMBED_URL.format(slug=slug)
    return url


@dataclass(frozen=True, slots=True)
class BrowserConfig:
    """Configuration for one isolated Slope browser."""

    url: str
    worker_id: int = 0
    channel: str = "bundled"
    headless: bool = True
    key_layout: str = "arrows"
    viewport_width: int = 640
    viewport_height: int = 427
    load_timeout_s: float = 120.0
    operation_timeout_ms: int = 10_000
    screenshot_format: str = "jpeg"
    jpeg_quality: int = 70
    reset_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("url cannot be empty")
        if self.worker_id < 0:
            raise ValueError("worker_id cannot be negative")
        if self.channel not in {"bundled", "chrome", "msedge"}:
            raise ValueError("channel must be bundled, chrome, or msedge")
        if self.key_layout not in {"arrows", "ad"}:
            raise ValueError("key_layout must be arrows or ad")
        if self.viewport_width < 320 or self.viewport_height < 240:
            raise ValueError("viewport dimensions are too small for the game canvas")
        if not math.isfinite(self.load_timeout_s) or self.load_timeout_s <= 0:
            raise ValueError("load_timeout_s must be positive")
        if self.operation_timeout_ms < 1:
            raise ValueError("operation_timeout_ms must be positive")
        if self.screenshot_format not in {"jpeg", "png"}:
            raise ValueError("screenshot_format must be jpeg or png")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 1 <= self.reset_attempts <= 10:
            raise ValueError("reset_attempts must be between 1 and 10")


class SlopeBrowser:
    """One synchronously controlled, fixed-step Slope browser session."""

    def __init__(self, config: BrowserConfig) -> None:
        self.config = config
        self.url = normalize_game_url(config.url)
        if config.key_layout == "arrows":
            self.left_key, self.right_key = "ArrowLeft", "ArrowRight"
        else:
            self.left_key, self.right_key = "a", "d"

        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.canvas: Any | None = None
        self.canvas_frame: Any | None = None
        self.direction = 0
        self.started = False
        self.clock_frozen = False
        self.resume_after_popup = False

    @property
    def playwright_channel(self) -> str | None:
        """Return Playwright's channel value for the configured browser."""

        return None if self.config.channel == "bundled" else self.config.channel

    @property
    def worker_name(self) -> str:
        return f"Worker {self.config.worker_id + 1}"

    def _launch(self) -> None:
        if self.page is not None:
            return
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError as exc:
            raise BrowserError(
                "Playwright is required. Install the project dependencies and run "
                "`python -m playwright install chromium`."
            ) from exc

        try:
            self.playwright = _acquire_playwright()
            self.browser = self.playwright.chromium.launch(
                channel=self.playwright_channel,
                headless=self.config.headless,
                ignore_default_args=["--disable-popup-blocking"],
                args=[
                    "--enable-webgl",
                    "--ignore-gpu-blocklist",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
            self.context = self.browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                device_scale_factor=1,
                locale="en-US",
            )
            self.context.add_init_script(_POPUP_GUARD_SCRIPT)
            self.page = self.context.new_page()
            self._install_popup_handlers()
            self.page.clock.install(time=time.time())
            self.page.set_default_timeout(self.config.operation_timeout_ms)

            response = self.page.goto(
                self.url,
                wait_until="domcontentloaded",
                timeout=round(self.config.load_timeout_s * 1000),
            )
            self._check_response(response, self.url)
            self._promote_y8_embed()
            self._discover_canvas()
            self._wait_for_unity_loader()
            self._close_extra_pages()
            self._dismiss_consent()
            self._wait_for_menu_pixels()
            self._freeze_clock()
        except Exception as exc:
            self.close()
            if isinstance(exc, BrowserError):
                raise
            hint = (
                " Install Chrome or Edge, or install bundled Chromium with "
                "`python -m playwright install chromium`."
            )
            raise BrowserError(f"{self.worker_name}: browser launch failed: {exc}.{hint}") from exc

    @staticmethod
    def _check_response(response: Any | None, url: str) -> None:
        if response is not None and response.status >= 400:
            raise BrowserError(f"Game URL returned HTTP {response.status}: {url}")

    def _close_popup(self, popup: Any) -> None:
        if popup is self.page:
            return
        game_was_started = self.started
        try:
            popup.close()
        except Exception:
            # Popups frequently close themselves before the event is handled.
            pass
        if not self.config.headless and self.page is not None:
            try:
                self.page.bring_to_front()
            except Exception:
                pass
        if game_was_started:
            # Unity pauses Slope when an advertising page steals visibility.
            # Remember that the centered Resume control may need one click once
            # execution returns to the owner page.
            self.resume_after_popup = True

    @staticmethod
    def _dismiss_dialog(dialog: Any) -> None:
        """Dismiss alert/confirm prompts without stalling browser automation."""

        try:
            dialog.dismiss()
        except Exception:
            # A page navigation can dispose a dialog before its event runs.
            pass

    def _install_popup_handlers(self) -> None:
        """Keep the game owner page and reject every auxiliary UI surface."""

        if self.context is None or self.page is None:
            raise BrowserError("Cannot install popup handlers before creating a page")
        # Register after creating the owner page so it cannot be mistaken for
        # an advertising popup by a synchronous context event callback.
        self.context.on("page", self._close_popup)
        self.page.on("popup", self._close_popup)
        self.page.on("dialog", self._dismiss_dialog)

    def _close_extra_pages(self) -> None:
        if self.context is None:
            return
        for candidate in tuple(self.context.pages):
            if candidate is not self.page:
                self._close_popup(candidate)

    def _resume_after_closed_popup(self) -> bool:
        """Click Unity's Resume overlay after a pop-under stole page focus."""

        if not self.resume_after_popup:
            return False
        self.resume_after_popup = False
        if self.page is None or not self.started or not self.clock_frozen:
            return False
        self._click_resume_control()
        return True

    def _click_resume_control(self) -> None:
        """Refocus the canvas and clear Unity's centered Resume overlay."""

        self._click_canvas(0.50, 0.50)
        self._advance_physics_ms(_RESET_POLL_MS)
        self._park_mouse()

    def _promote_y8_embed(self) -> None:
        """Navigate from Y8's wrapper into the actual Unity host page."""

        assert self.page is not None
        parsed = urlparse(self.page.url)
        host = (parsed.hostname or "").lower()
        if not (host == "y8.com" or host.endswith(".y8.com")):
            return
        if not parsed.path.startswith("/embed/"):
            return

        embed = self.page.locator("iframe.embed-content")
        try:
            embed.wait_for(
                state="attached",
                timeout=round(self.config.load_timeout_s * 1000),
            )
        except Exception:
            # Some Y8 revisions put the Unity canvas directly on the embed page.
            return
        source = embed.get_attribute("src")
        if not source:
            return
        target = urljoin(self.page.url, source)
        response = self.page.goto(
            target,
            wait_until="domcontentloaded",
            timeout=round(self.config.load_timeout_s * 1000),
        )
        self._check_response(response, target)

    def _discover_canvas(self) -> None:
        assert self.page is not None
        deadline = time.monotonic() + self.config.load_timeout_s
        while time.monotonic() < deadline:
            candidates: list[tuple[float, Any, Any]] = []
            for frame in tuple(self.page.frames):
                try:
                    preferred = frame.locator("#unity-canvas")
                    preferred_count = preferred.count()
                    locators = preferred if preferred_count else frame.locator("canvas")
                    count = preferred_count or min(locators.count(), 10)
                    for index in range(count):
                        locator = locators.nth(index)
                        if not locator.is_visible():
                            continue
                        box = locator.bounding_box()
                        if box is None or box["width"] < 200 or box["height"] < 200:
                            continue
                        score = float(box["width"] * box["height"])
                        if preferred_count:
                            score *= 2.0
                        candidates.append((score, frame, locator))
                except Exception:
                    # Ads and consent helpers create short-lived frames.
                    continue
            if candidates:
                _, self.canvas_frame, self.canvas = max(
                    candidates, key=lambda candidate: candidate[0]
                )
                return
            self.page.wait_for_timeout(100)
        raise BrowserError(f"{self.worker_name}: no usable game canvas appeared")

    def _wait_for_unity_loader(self) -> None:
        if self.canvas_frame is None:
            raise BrowserError("Game canvas frame is unavailable")
        loader = self.canvas_frame.locator("#unity-loading-bar")
        if loader.count():
            loader.wait_for(
                state="hidden",
                timeout=round(self.config.load_timeout_s * 1000),
            )

    def _canvas_box(self) -> dict[str, float]:
        if self.canvas is None:
            raise BrowserError("Game canvas is unavailable")
        box = self.canvas.bounding_box(timeout=self.config.operation_timeout_ms)
        if box is None:
            raise BrowserError("Game canvas is not visible")
        return box

    def _click_canvas(self, x_ratio: float, y_ratio: float) -> None:
        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        box = self._canvas_box()
        self.page.mouse.click(
            box["x"] + box["width"] * x_ratio,
            box["y"] + box["height"] * y_ratio,
        )

    def _park_mouse(self) -> None:
        """Move the pointer away from START/AGAIN and other lower controls."""

        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        box = self._canvas_box()
        margin = max(2.0, min(12.0, box["width"] * 0.02, box["height"] * 0.02))
        self.page.mouse.move(box["x"] + margin, box["y"] + margin)

    def _capture(self) -> np.ndarray:
        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        options: dict[str, Any] = {
            "clip": self._canvas_box(),
            "type": self.config.screenshot_format,
            "timeout": self.config.operation_timeout_ms,
        }
        if self.config.screenshot_format == "jpeg":
            options["quality"] = self.config.jpeg_quality
        encoded = self.page.screenshot(**options)
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise BrowserError("Chromium returned an empty canvas screenshot")
        return frame

    def _visible_locator(self, selectors: tuple[str, ...]) -> Any | None:
        if self.page is None:
            return None
        for frame in tuple(self.page.frames):
            for selector in selectors:
                try:
                    matches = frame.locator(selector)
                    for index in range(min(matches.count(), 10)):
                        candidate = matches.nth(index)
                        if candidate.is_visible():
                            return candidate
                except Exception:
                    continue
        return None

    @staticmethod
    def _privacy_panel(frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Locate a large bright consent card covering the dark game canvas."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.inRange(gray, 225, 255)
        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_CLOSE,
            np.ones((15, 15), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = gray.shape
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area_ratio = box_width * box_height / float(width * height)
            if area_ratio >= 0.18 and box_width >= width * 0.45 and box_height >= height * 0.25:
                candidates.append((area_ratio, (x, y, box_width, box_height)))
        return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    @staticmethod
    def _ad_panel(frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Locate either the wide or compact light card used by Y8 ads."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.inRange(gray, 210, 255)
        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = gray.shape
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area_ratio = box_width * box_height / float(width * height)
            covers_center = x <= width * 0.50 <= x + box_width
            if (
                area_ratio >= 0.08
                and box_width >= width * 0.25
                and box_height >= height * 0.20
                and covers_center
            ):
                candidates.append((area_ratio, (x, y, box_width, box_height)))
        return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    def _dismiss_consent(self) -> None:
        """Reject Y8's optional privacy prompt before waiting for Unity pixels."""

        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        selectors = (
            "button:has-text('Reject All')",
            "button:has-text('Reject')",
            "button:has-text('Decline')",
            "button[aria-label*='Reject' i]",
            "#notice-reject-all",
            ".sp_choice_type_13",
        )
        deadline = time.monotonic() + min(5.0, self.config.load_timeout_s)
        while time.monotonic() < deadline:
            button = self._visible_locator(selectors)
            if button is not None:
                try:
                    button.click(timeout=min(1_500, self.config.operation_timeout_ms))
                except Exception:
                    button.dispatch_event("click", timeout=self.config.operation_timeout_ms)
                self.page.wait_for_timeout(250)
                return

            frame = self._capture()
            panel = self._privacy_panel(frame)
            if panel is not None:
                x, y, panel_width, panel_height = panel
                box = self._canvas_box()
                frame_height, frame_width = frame.shape[:2]
                # Sourcepoint's three choices place Reject in the lower middle.
                self.page.mouse.click(
                    box["x"] + (x + panel_width * 0.50) * box["width"] / frame_width,
                    box["y"] + (y + panel_height * 0.82) * box["height"] / frame_height,
                )
                self.page.wait_for_timeout(250)
                return
            self.page.wait_for_timeout(100)

    @staticmethod
    def _interstitial_close_point(frame: np.ndarray) -> tuple[float, float] | None:
        """Locate the delayed Close label above Y8's in-game ad card.

        The creative itself is cross-origin and changes frequently, but Y8's
        host draws a stable wide light card with the Close label immediately
        above its right edge.  Detecting the card edge first avoids mistaking
        arbitrary white text inside an ad for the dismiss control.
        """

        panel = SlopeBrowser._ad_panel(frame)
        if panel is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = gray >= 220
        height, width = gray.shape
        panel_x, panel_top, panel_width, _ = panel
        panel_right = panel_x + panel_width

        # Search only immediately above the panel's right edge.  In the real
        # 640x400 captures this contains the six small components in "Close"
        # and is empty while the ad is still in its mandatory viewing period.
        x0 = max(0, panel_right - round(width * 0.12))
        x1 = min(width, panel_right + round(width * 0.03))
        y0 = max(0, panel_top - round(height * 0.12))
        y1 = max(y0 + 1, panel_top - 2)
        close_region = np.asarray(bright[y0:y1, x0:x1], dtype=np.uint8) * 255
        count, _, stats, _ = cv2.connectedComponentsWithStats(close_region)
        glyphs: list[tuple[int, int, int, int, int]] = []
        for index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[index])
            if area >= 4 and box_height >= 3:
                glyphs.append((x, y, box_width, box_height, area))
        if len(glyphs) < 4 or sum(glyph[4] for glyph in glyphs) < 60:
            return None

        left = min(glyph[0] for glyph in glyphs)
        top = min(glyph[1] for glyph in glyphs)
        right = max(glyph[0] + glyph[2] for glyph in glyphs)
        bottom = max(glyph[1] + glyph[3] for glyph in glyphs)
        return x0 + (left + right) / 2.0, y0 + (top + bottom) / 2.0

    @classmethod
    def _looks_like_interstitial(cls, frame: np.ndarray) -> bool:
        """Return whether a wide or compact light ad is covering the game."""

        return cls._ad_panel(frame) is not None

    def _click_frame_point(
        self, frame: np.ndarray, point: tuple[float, float]
    ) -> None:
        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        frame_height, frame_width = frame.shape[:2]
        box = self._canvas_box()
        x, y = point
        self.page.mouse.click(
            box["x"] + x * box["width"] / frame_width,
            box["y"] + y * box["height"] / frame_height,
        )

    def _click_ad_close(self, frame: np.ndarray) -> bool:
        """Click an available ad close control through DOM or pixel geometry."""

        close = self._visible_locator(_AD_CLOSE_SELECTORS)
        if close is not None:
            try:
                close.click(timeout=min(750, self.config.operation_timeout_ms))
                return True
            except Exception:
                try:
                    close.dispatch_event(
                        "click", timeout=self.config.operation_timeout_ms
                    )
                    return True
                except Exception:
                    pass

        point = self._interstitial_close_point(frame)
        if point is not None:
            self._click_frame_point(frame, point)
            return True

        # SDK-specific class/id names are useful before the label renders, but
        # are intentionally last because creatives often contain unrelated
        # elements whose class also includes the word "close".
        close = self._visible_locator(_AD_GENERIC_CLOSE_SELECTORS)
        if close is None:
            return False
        try:
            close.click(timeout=min(750, self.config.operation_timeout_ms))
        except Exception:
            try:
                close.dispatch_event("click", timeout=self.config.operation_timeout_ms)
            except Exception:
                return False
        return True

    def _dismiss_interstitial(self, frame: np.ndarray) -> np.ndarray:
        """Block until an in-game interstitial is closable and fully removed.

        Unity pauses gameplay beneath this modal.  Keeping the transition
        inside the browser boundary prevents the environment from recording
        dozens of identical ad frames as gameplay or false deaths.
        """

        if not self._looks_like_interstitial(frame):
            return frame
        if self.page is None or not self.clock_frozen:
            raise BrowserError("Cannot dismiss an ad without a paused game clock")

        self._release_keys()
        waited_ms = 0
        while self._looks_like_interstitial(frame):
            clicked = self._click_ad_close(frame)
            advance_ms = _AD_AFTER_CLICK_MS if clicked else _AD_POLL_MS
            if waited_ms + advance_ms > _AD_MAX_WAIT_MS:
                raise BrowserError(
                    f"{self.worker_name}: interstitial ad did not become dismissible"
                )
            self._advance_physics_ms(advance_ms)
            waited_ms += advance_ms
            self._close_extra_pages()
            frame = self._capture()

        # Y8/Unity commonly pauses after an ad or its pop-under takes focus.
        # Clicking the canvas center activates Resume when present and is inert
        # during ordinary keyboard-controlled Slope gameplay.
        self._click_resume_control()
        frame = self._capture()
        print(f"{self.worker_name}: dismissed an interstitial ad", flush=True)
        return frame

    @staticmethod
    def _looks_like_menu(frame: np.ndarray) -> bool:
        """Recognize the rendered main menu, not merely Unity's DOM loader."""

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (32, 70, 40), (100, 255, 255))
        # Before START is clicked, Slope's only green-rich canvas state is its
        # rendered menu. This also survives theme/ad variants whose PLAY border
        # is not consistently red. The Unity splash and consent card both fail
        # this saturated-neon threshold.
        del height, width
        return bool(np.count_nonzero(green) / green.size >= 0.006)

    @staticmethod
    def _looks_like_running_game(frame: np.ndarray) -> bool:
        """Recognize the first controllable scene from its green road geometry."""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (32, 70, 40), (100, 255, 255))
        return bool(
            not SlopeBrowser._looks_like_interstitial(frame)
            and np.count_nonzero(green) / green.size >= 0.12
        )

    def _wait_for_menu_pixels(self) -> None:
        """Advance only the pre-game menu until Unity has actually rendered it."""

        deadline = time.monotonic() + self.config.load_timeout_s
        last_frame: np.ndarray | None = None
        while time.monotonic() < deadline:
            frame = self._capture()
            last_frame = frame
            if self._looks_like_menu(frame):
                return
            if self.clock_frozen:
                self._advance_physics_ms(_MENU_POLL_MS)
            else:
                assert self.page is not None
                self.page.wait_for_timeout(_MENU_POLL_MS)
        details = ""
        if last_frame is not None:
            hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, (32, 70, 40), (100, 255, 255))
            red = cv2.bitwise_or(
                cv2.inRange(hsv, (0, 100, 55), (13, 255, 255)),
                cv2.inRange(hsv, (167, 100, 55), (179, 255, 255)),
            )
            details = (
                f" (green={np.count_nonzero(green) / green.size:.3f}, "
                f"red={np.count_nonzero(red) / red.size:.3f}, "
                f"brightness={last_frame.mean():.1f})"
            )
        raise BrowserError(f"{self.worker_name}: Unity menu did not become ready{details}")

    def _freeze_clock(self) -> None:
        if self.page is None:
            raise BrowserError("Browser clock is unavailable")
        if self.clock_frozen:
            return

        # Reading the emulated clock avoids asking Clock to travel backwards if
        # navigation took longer than expected.  Small margins handle call
        # latency without consuming meaningful game time (the game is at menu).
        last_error: Exception | None = None
        for margin in (0.05, 0.10, 0.25):
            try:
                current = float(self.page.evaluate("Date.now()")) / 1000.0
                self.page.clock.pause_at(current + margin)
                self.clock_frozen = True
                return
            except Exception as exc:
                if "past" not in str(exc).lower():
                    raise
                last_error = exc
        raise BrowserError("Could not pause the game clock safely") from last_error

    def _resume_clock(self) -> None:
        if self.page is None:
            raise BrowserError("Browser clock is unavailable")
        if self.clock_frozen:
            self.page.clock.resume()
            self.clock_frozen = False

    def _advance_physics_ms(self, milliseconds: int) -> None:
        """Advance paused game time using only display-sized physics slices."""

        if self.page is None:
            raise BrowserError("Browser clock is unavailable")
        if not self.clock_frozen:
            raise BrowserError("Refusing to advance an unpaused game clock")
        if milliseconds < 1:
            raise ValueError("milliseconds must be positive")

        wall_started = time.perf_counter()
        advanced = 0
        while advanced < milliseconds:
            physics_slice = min(_PHYSICS_SLICE_MS, milliseconds - advanced)
            self.page.clock.run_for(physics_slice)
            advanced += physics_slice
            if not self.config.headless:
                delay = wall_started + advanced / 1000.0 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

    def _release_keys(self) -> None:
        if self.page is not None:
            # Release both keys even if local direction state became stale after
            # a navigation or interrupted transition.
            self.page.keyboard.up(self.left_key)
            self.page.keyboard.up(self.right_key)
        self.direction = 0

    def _set_direction(self, direction: int) -> None:
        if direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        if direction == self.direction:
            return
        if self.direction < 0:
            self.page.keyboard.up(self.left_key)
        elif self.direction > 0:
            self.page.keyboard.up(self.right_key)
        if direction < 0:
            self.page.keyboard.down(self.left_key)
        elif direction > 0:
            self.page.keyboard.down(self.right_key)
        self.direction = direction

    def _wait_for_running_frame(self) -> np.ndarray:
        """Return the earliest rendered game frame after START/AGAIN.

        Unity briefly renders black while changing scenes. Polling in three
        display-frame chunks does not confuse that transition with an episode,
        and exposes the first controllable frame with at most 48 ms latency.
        """

        advanced = 0
        last_frame: np.ndarray | None = None
        while advanced < _RESET_MAX_TRANSITION_MS:
            self._advance_physics_ms(_RESET_POLL_MS)
            advanced += _RESET_POLL_MS
            self._close_extra_pages()
            self._resume_after_closed_popup()
            last_frame = self._capture()
            if self._looks_like_interstitial(last_frame):
                last_frame = self._dismiss_interstitial(last_frame)
                if self._resume_after_closed_popup():
                    last_frame = self._capture()
            if self._looks_like_running_game(last_frame):
                return last_frame
        details = ""
        if last_frame is not None:
            hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, (32, 70, 40), (100, 255, 255))
            red = cv2.bitwise_or(
                cv2.inRange(hsv, (0, 100, 55), (13, 255, 255)),
                cv2.inRange(hsv, (167, 100, 55), (179, 255, 255)),
            )
            details = (
                f" (green={np.count_nonzero(green) / green.size:.3f}, "
                f"red={np.count_nonzero(red) / red.size:.3f}, "
                f"brightness={last_frame.mean():.1f})"
            )
        raise BrowserError(
            f"{self.worker_name}: game did not render a running scene after reset{details}"
        )

    def _start_from_menu(self) -> np.ndarray:
        self._release_keys()
        self._click_canvas(0.50, 0.462)
        frame = self._wait_for_running_frame()
        self._park_mouse()
        self.started = True
        return frame

    def _restart_after_death(self) -> np.ndarray:
        self._release_keys()
        self._click_canvas(0.50, 0.908)
        frame = self._wait_for_running_frame()
        self._park_mouse()
        self.started = True
        return frame

    def _reload_to_menu(self) -> None:
        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        self._release_keys()
        self._resume_clock()
        response = self.page.reload(
            wait_until="domcontentloaded",
            timeout=round(self.config.load_timeout_s * 1000),
        )
        self._check_response(response, self.page.url)
        self.canvas = None
        self.canvas_frame = None
        self._discover_canvas()
        self._wait_for_unity_loader()
        self._close_extra_pages()
        self._dismiss_consent()
        self._wait_for_menu_pixels()
        self._freeze_clock()
        self.started = False

    def _reset_once(self, reason: str) -> np.ndarray:
        if self.page is None:
            raise BrowserError("Browser page is unavailable")
        if not self.clock_frozen:
            self._freeze_clock()

        if not self.started:
            return self._start_from_menu()
        elif reason == "terminated":
            return self._restart_after_death()
        else:
            # A truncation/manual reset can occur while the ball is alive, so
            # the death-screen AGAIN button is not guaranteed to exist.
            self._reload_to_menu()
            return self._start_from_menu()

    def reset(self, reason: str = "initial") -> np.ndarray:
        """Start a new run and return its first BGR frame.

        A reset retries with a completely new browser when navigation, ads, or
        transient canvas races make the current page unusable.
        """

        if reason not in _RESET_REASONS:
            raise ValueError(f"unknown reset reason: {reason}")
        last_error: Exception | None = None
        for attempt in range(1, self.config.reset_attempts + 1):
            try:
                self._launch()
                return self._reset_once(reason)
            except Exception as exc:
                last_error = exc
                self.close()
                if attempt < self.config.reset_attempts:
                    summary = str(exc).splitlines()[0] or type(exc).__name__
                    print(
                        f"{self.worker_name}: reset attempt {attempt}/"
                        f"{self.config.reset_attempts} failed ({summary}); relaunching...",
                        flush=True,
                    )
        raise BrowserError(
            f"{self.worker_name}: reset failed after {self.config.reset_attempts} attempts"
        ) from last_error

    def step(self, direction: int, dt: float) -> np.ndarray:
        """Apply steering for exactly ``dt`` simulated seconds and capture BGR."""

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")
        if self.page is None or not self.started:
            raise BrowserError("reset() must succeed before step()")
        try:
            self._set_direction(direction)
            self._advance_physics_ms(max(1, round(dt * 1000)))
            self._close_extra_pages()
            self._resume_after_closed_popup()
            frame = self._capture()
            if self._looks_like_interstitial(frame):
                frame = self._dismiss_interstitial(frame)
                if self._resume_after_closed_popup():
                    frame = self._capture()
                # Dismissal releases both keys. Restore the requested control
                # without advancing extra gameplay time after the ad unpauses.
                self._set_direction(direction)
            return frame
        except (BrowserError, ValueError):
            raise
        except Exception as exc:
            raise BrowserError(f"{self.worker_name}: browser step failed: {exc}") from exc

    def close(self) -> None:
        """Release controls and idempotently dispose every Playwright resource."""

        try:
            self._release_keys()
        except Exception:
            self.direction = 0
        try:
            if self.context is not None:
                self.context.close()
        except Exception:
            pass
        try:
            if self.browser is not None:
                self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright is not None:
                _release_playwright(self.playwright)
        except Exception:
            pass
        finally:
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None
            self.canvas = None
            self.canvas_frame = None
            self.direction = 0
            self.started = False
            self.clock_frozen = False
            self.resume_after_popup = False


__all__ = ["BrowserConfig", "BrowserError", "SlopeBrowser", "normalize_game_url"]
