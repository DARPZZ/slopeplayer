"""Isolated Playwright control for browser-hosted Slope games.

Each instance owns a browser process, so screenshots and keyboard events never
depend on which desktop window has focus.  Chromium pages are frozen between
Gym steps; this prevents the real-time game from continuing while PPO updates
the policy or another vector environment restarts.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np


Y8_EMBED_URL = "https://www.y8.com/embed/{slug}"


class BrowserSessionError(RuntimeError):
    """Raised when an isolated game page cannot be controlled safely."""


def normalize_game_url(url: str) -> str:
    """Turn a public Y8 game page into its stable, lightweight embed URL."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if (host == "y8.com" or host.endswith(".y8.com")) and len(parts) == 2:
        if parts[0] == "games" and all(
            character.isalnum() or character in "_-" for character in parts[1]
        ):
            return Y8_EMBED_URL.format(slug=parts[1])
    return url


def playwright_key_names(layout: str) -> tuple[str, str]:
    if layout == "arrows":
        return "ArrowLeft", "ArrowRight"
    return "a", "d"


class PlaywrightSlopeSession:
    """One independently addressable Y8/Unity game page."""

    def __init__(
        self,
        url: str,
        worker_id: int,
        layout: str,
        browser_channel: str = "chrome",
        headless: bool = True,
        load_timeout: float = 120.0,
        realtime: bool = False,
    ) -> None:
        self.url = normalize_game_url(url)
        self.worker_id = worker_id
        self.left_key, self.right_key = playwright_key_names(layout)
        self.browser_channel = browser_channel
        self.headless = headless
        self.realtime = realtime
        self.load_timeout_ms = max(1, round(load_timeout * 1000))
        self.operation_timeout_ms = 10_000

        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.canvas: Any | None = None
        self.canvas_frame: Any | None = None
        self.direction = 0
        self.frozen = False
        self.clock_time = time.time()
        self.last_step_at = time.perf_counter()
        self.started = False

    def _launch(self) -> None:
        if self.page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserSessionError(
                "Playwright is required for --url mode. Install the RL dependencies "
                "with `python -m pip install -r requirements-rl.txt`."
            ) from exc

        try:
            self.playwright = sync_playwright().start()
            channel = None if self.browser_channel == "bundled" else self.browser_channel
            self.browser = self.playwright.chromium.launch(
                channel=channel,
                headless=self.headless,
                # Playwright normally disables Chromium's popup blocker for
                # automation. Keep the browser default so an ad cannot open a
                # second visible window from a canvas click.
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
                viewport={"width": 960, "height": 641},
                device_scale_factor=1,
                locale="en-US",
            )
            self.page = self.context.new_page()
            self.context.on("page", self._close_unexpected_page)
            # Playwright's clock controls requestAnimationFrame for the whole
            # context, including Unity and its helper frames. It is installed
            # before navigation so every game timer can be paused later.
            self.page.clock.install(time=self.clock_time)
            self.page.set_default_timeout(self.operation_timeout_ms)
            response = self.page.goto(
                self.url,
                wait_until="domcontentloaded",
                timeout=self.load_timeout_ms,
            )
            if response is not None and response.status >= 400:
                raise BrowserSessionError(
                    f"Worker {self.worker_id + 1}: game URL returned HTTP {response.status}"
                )
            self._promote_y8_embed()
            self._find_canvas()
            self._wait_for_unity()
            self._dismiss_consent()
            self._wait_for_menu()
        except Exception as exc:
            self.close()
            if isinstance(exc, BrowserSessionError):
                raise
            hint = (
                " Install Chrome, choose `--browser-channel msedge`, or run "
                "`python -m playwright install chromium` and use "
                "`--browser-channel bundled`."
            )
            raise BrowserSessionError(
                f"Worker {self.worker_id + 1}: could not start the game browser: {exc}.{hint}"
            ) from exc

    def _close_unexpected_page(self, page: Any) -> None:
        """Close advertising popups while preserving this worker's game page."""

        if page is self.page:
            return
        try:
            popup_url = page.url
            page.close()
            print(
                f"Worker {self.worker_id + 1}: blocked an unexpected popup"
                f"{f' ({popup_url})' if popup_url else ''}.",
                flush=True,
            )
        except Exception:
            # A popup may close itself before the event handler runs.
            pass

    def _promote_y8_embed(self) -> None:
        """Navigate into Y8's current game frame so the clock controls Unity itself."""

        assert self.page is not None
        parsed = urlparse(self.page.url)
        host = (parsed.hostname or "").lower()
        if not (host == "y8.com" or host.endswith(".y8.com")):
            return
        if not parsed.path.startswith("/embed/"):
            return
        embed = self.page.locator("iframe.embed-content")
        embed.wait_for(state="attached", timeout=self.load_timeout_ms)
        source = embed.get_attribute("src")
        if not source:
            raise BrowserSessionError("Y8's game embed did not provide a source URL")
        response = self.page.goto(
            urljoin(self.page.url, source),
            wait_until="domcontentloaded",
            timeout=self.load_timeout_ms,
        )
        if response is not None and response.status >= 400:
            raise BrowserSessionError(
                f"Worker {self.worker_id + 1}: Y8 game frame returned HTTP {response.status}"
            )

    def _find_canvas(self) -> None:
        assert self.page is not None
        deadline = time.monotonic() + self.load_timeout_ms / 1000
        while time.monotonic() < deadline:
            candidates: list[tuple[float, Any, Any]] = []
            for frame in self.page.frames:
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
                            score *= 2
                        candidates.append((score, frame, locator))
                except Exception:
                    # Ads and consent helpers create short-lived frames while
                    # the Unity page starts. A detached candidate is harmless.
                    continue
            if candidates:
                _, self.canvas_frame, self.canvas = max(
                    candidates, key=lambda candidate: candidate[0]
                )
                return
            self.page.wait_for_timeout(250)
        raise BrowserSessionError(
            f"Worker {self.worker_id + 1}: no usable game canvas appeared at {self.url}"
        )

    def _wait_for_unity(self) -> None:
        assert self.canvas_frame is not None
        loader = self.canvas_frame.locator("#unity-loading-bar")
        if loader.count():
            loader.wait_for(state="hidden", timeout=self.load_timeout_ms)

    def _canvas_box(self) -> dict[str, float]:
        if self.canvas is None:
            raise BrowserSessionError("The game canvas is not available")
        box = self.canvas.bounding_box(timeout=self.operation_timeout_ms)
        if box is None:
            raise BrowserSessionError("The game canvas is not visible")
        return box

    def _click_canvas(self, x_ratio: float, y_ratio: float) -> None:
        assert self.page is not None
        box = self._canvas_box()
        self.page.mouse.click(
            box["x"] + box["width"] * x_ratio,
            box["y"] + box["height"] * y_ratio,
        )

    def _raw_frame(self) -> np.ndarray:
        assert self.page is not None
        box = self._canvas_box()
        encoded = self.page.screenshot(
            clip=box,
            type="png",
            timeout=self.operation_timeout_ms,
        )
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise BrowserSessionError("Chromium returned an empty canvas screenshot")
        return frame

    @staticmethod
    def _green_ratio(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (32, 75, 45), (100, 255, 255))
        return float(np.count_nonzero(mask)) / mask.size

    @staticmethod
    def _privacy_panel(frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Locate Sourcepoint's large white consent card in a canvas capture."""

        white = cv2.inRange(frame, (235, 235, 235), (255, 255, 255))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        frame_height, frame_width = frame.shape[:2]
        if (
            width < frame_width * 0.50
            or height < frame_height * 0.25
            or cv2.contourArea(contour) < frame_width * frame_height * 0.20
        ):
            return None
        return x, y, width, height

    def _find_visible_locator(self, selector: str) -> Any | None:
        assert self.page is not None
        for frame in self.page.frames:
            try:
                matches = frame.locator(selector)
                for index in range(matches.count()):
                    candidate = matches.nth(index)
                    if candidate.is_visible():
                        return candidate
            except Exception:
                continue
        return None

    def _dismiss_consent(self) -> None:
        """Reject Y8's optional consent dialog without touching ad content."""

        assert self.page is not None
        deadline = time.monotonic() + 4.0
        consent = None
        while time.monotonic() < deadline:
            consent = self._find_visible_locator("iframe[title*='Consent']")
            if consent is not None:
                break
            self.page.wait_for_timeout(100)
        if consent is None:
            return
        frame = self._raw_frame()
        panel = self._privacy_panel(frame)
        if panel is None:
            return
        x, y, width, height = panel
        canvas_box = self._canvas_box()
        frame_height, frame_width = frame.shape[:2]
        # Sourcepoint lays out Preferences / Reject / Accept in three columns.
        # The middle button sits near the bottom centre of the white card. Its
        # vertical position varies with translated text, so derive it from the
        # detected card rather than relying on one locale's fixed coordinate.
        self.page.mouse.click(
            canvas_box["x"] + (x + width * 0.50) * canvas_box["width"] / frame_width,
            canvas_box["y"] + (y + height * 0.82) * canvas_box["height"] / frame_height,
        )
        hidden_deadline = time.monotonic() + 5.0
        while time.monotonic() < hidden_deadline:
            self.page.wait_for_timeout(100)
            if self._privacy_panel(self._raw_frame()) is None:
                return
        raise BrowserSessionError("Y8's privacy dialog could not be dismissed")

    def _wait_for_menu(self) -> None:
        assert self.page is not None
        deadline = time.monotonic() + self.load_timeout_ms / 1000
        while time.monotonic() < deadline:
            frame = self._raw_frame()
            if self._privacy_panel(frame) is None and self._green_ratio(frame) >= 0.006:
                return
            self.page.wait_for_timeout(250)
        raise BrowserSessionError("The Slope menu did not finish loading")

    def _ad_is_present(self) -> bool:
        assert self.page is not None
        hints = ("googlesyndication", "doubleclick", "safeframe")
        for frame in self.page.frames:
            try:
                if any(hint in frame.url.lower() for hint in hints):
                    owner = frame.frame_element()
                    box = owner.bounding_box() if owner.is_visible() else None
                    if box is not None and box["width"] * box["height"] >= 10_000:
                        return True
                matches = frame.locator("iframe[id*='google_ads_iframe']")
                for index in range(matches.count()):
                    candidate = matches.nth(index)
                    box = candidate.bounding_box() if candidate.is_visible() else None
                    if box is not None and box["width"] * box["height"] >= 10_000:
                        return True
            except Exception:
                continue
        return False

    def _wait_for_ad_to_close(self, timeout: float = 5.0) -> bool:
        assert self.page is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._ad_is_present() and self._visible_ad_close() is None:
                return True
            self.page.wait_for_timeout(100)
        return not self._ad_is_present() and self._visible_ad_close() is None

    def _visible_ad_close(self) -> Any | None:
        for selector in (
            "#close-button",
            "#dismiss-button",
            "#dismiss-button-element",
        ):
            found = self._find_visible_locator(selector)
            if found is not None:
                return found
        return None

    def _click_ad_close(self, close: Any) -> bool:
        """Dismiss a vetted ad close control even when its card overlaps it."""

        try:
            # Do not spend the full operation timeout on Playwright's pointer
            # actionability retries. Some Google creatives briefly place their
            # own #card above an otherwise-visible dismiss button.
            close.click(timeout=min(self.operation_timeout_ms, 1_000))
            return True
        except Exception:
            try:
                # This locator came only from the explicit close selectors
                # above. Dispatching directly bypasses hit testing without
                # guessing coordinates or clicking arbitrary ad content.
                close.dispatch_event("click", timeout=self.operation_timeout_ms)
                return True
            except Exception:
                # The creative may have replaced its DOM or closed naturally.
                # Let the polling loop rediscover current state instead of
                # terminating the entire subprocess.
                return not self._ad_is_present()

    def _handle_optional_ad(self, detection_seconds: float) -> bool:
        """Wait for and click only the ad provider's explicit Close control."""

        assert self.page is not None
        detection_deadline = time.monotonic() + detection_seconds
        close_deadline: float | None = None
        absent_since: float | None = None
        saw_ad = False
        while True:
            ad_present = self._ad_is_present()
            close = self._visible_ad_close()
            ad_present = ad_present or close is not None
            saw_ad = saw_ad or ad_present
            if close is not None:
                saw_ad = True
                close_deadline = close_deadline or time.monotonic() + 45.0
                if self._click_ad_close(close) and self._wait_for_ad_to_close():
                    return True
            now = time.monotonic()
            if saw_ad and not ad_present:
                absent_since = now if absent_since is None else absent_since
                if now - absent_since >= 0.5:
                    return True
            else:
                absent_since = None
            if saw_ad and close_deadline is None:
                close_deadline = now + 45.0
            if saw_ad and close_deadline is not None and now >= close_deadline:
                raise BrowserSessionError(
                    "A Y8 pre-roll ad appeared but its Close control did not become available"
                )
            if not saw_ad and now >= detection_deadline:
                return False
            self.page.wait_for_timeout(200)

    def _set_lifecycle(self, state: str) -> None:
        if self.page is None:
            raise BrowserSessionError("Browser clock control is unavailable")
        if state == "active":
            self.page.clock.resume()
            self.frozen = False
            return
        # Read the emulated clock itself. Deriving it from wall time can be a
        # few milliseconds behind after navigation, which Clock correctly
        # rejects as an attempt to travel backwards.
        last_error: Exception | None = None
        for margin in (0.25, 0.50, 1.0):
            try:
                self.clock_time = float(self.page.evaluate("Date.now()")) / 1000 + margin
                self.page.clock.pause_at(self.clock_time)
                self.frozen = True
                return
            except Exception as exc:
                if "past" not in str(exc).lower():
                    raise
                last_error = exc
        raise BrowserSessionError("Could not pause the game clock safely") from last_error

    def _activate(self) -> None:
        if self.frozen:
            self._set_lifecycle("active")

    def _freeze(self) -> None:
        if not self.frozen:
            self._set_lifecycle("frozen")

    def _release(self) -> None:
        if self.page is None:
            self.direction = 0
            return
        self.page.keyboard.up(self.left_key)
        self.page.keyboard.up(self.right_key)
        self.direction = 0

    def _set_direction(self, direction: int) -> None:
        assert self.page is not None
        direction = max(-1, min(1, int(direction)))
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

    def _advance_clock(self, milliseconds: int) -> None:
        """Advance exactly this much game time, pacing visible animation frames."""

        if self.page is None:
            raise BrowserSessionError("Browser clock control is unavailable")
        milliseconds = max(1, int(milliseconds))
        if self.headless:
            self.page.clock.run_for(milliseconds)
            return

        # run_for executes all callbacks as quickly as possible. That is ideal
        # for hidden workers, but a headed window appears to teleport and then
        # freeze. Advance one display frame at a time and pace those slices
        # against wall time without ever unpausing the deterministic game clock.
        started_at = time.perf_counter()
        advanced = 0
        while advanced < milliseconds:
            step = min(16, milliseconds - advanced)
            self.page.clock.run_for(step)
            advanced += step
            delay = started_at + advanced / 1000 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def _start_from_menu(self, restart_wait: float) -> None:
        assert self.page is not None
        self._click_canvas(0.50, 0.462)
        if self._handle_optional_ad(detection_seconds=1.5):
            # Y8 pauses Unity after an ad and renders Resume in the canvas.
            self._click_canvas(0.50, 0.50)
        self._freeze()
        if restart_wait > 0:
            self._advance_clock(round(restart_wait * 1000))

    def _reload_to_menu(self) -> None:
        assert self.page is not None
        self._activate()
        self._release()
        self.page.reload(wait_until="domcontentloaded", timeout=self.load_timeout_ms)
        self.canvas = None
        self.canvas_frame = None
        self._find_canvas()
        self._wait_for_unity()
        self._dismiss_consent()
        self._wait_for_menu()

    def _reset_once(self, reason: str, restart_wait: float) -> np.ndarray:
        assert self.page is not None
        try:
            self._activate()
            self._release()
            if not self.started:
                self._start_from_menu(restart_wait)
                self.started = True
            elif reason == "terminated":
                # Y8 renders AGAIN near the bottom of the Unity canvas.
                self._click_canvas(0.50, 0.908)
                if self._handle_optional_ad(detection_seconds=0.50):
                    self._click_canvas(0.50, 0.50)
                self._freeze()
                if restart_wait > 0:
                    self._advance_clock(round(restart_wait * 1000))
            else:
                # A time limit/manual reset can happen while the ball is alive,
                # so the death-screen AGAIN button is not available.
                self._reload_to_menu()
                self._start_from_menu(restart_wait)
        except Exception:
            try:
                self._freeze()
            except Exception:
                pass
            raise
        else:
            self._freeze()
        return self._raw_frame()

    def reset(self, reason: str, restart_wait: float) -> np.ndarray:
        attempts = 3
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._launch()
                frame = self._reset_once(reason, restart_wait)
                self.last_step_at = time.perf_counter()
                return frame
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    summary = str(exc).splitlines()[0] or type(exc).__name__
                    print(
                        f"Worker {self.worker_id + 1}: reset attempt {attempt}/{attempts} "
                        f"failed ({summary}); reopening the browser...",
                        flush=True,
                    )
                try:
                    self.close()
                except Exception:
                    pass
        raise BrowserSessionError(
            f"Worker {self.worker_id + 1}: could not reset the game after "
            f"{attempts} browser attempts"
        ) from last_error

    def advance(self, direction: int, frame_period: float) -> np.ndarray:
        self._launch()
        assert self.page is not None
        if self.realtime:
            # Watched playback should remain fluid. Keep the browser clock live
            # and pace observations against wall time, accounting for the time
            # spent capturing the previous frame and running policy inference.
            self._activate()
            self._set_direction(direction)
            deadline = self.last_step_at + frame_period
            delay = deadline - time.perf_counter()
            if delay > 0:
                self.page.wait_for_timeout(max(1, round(delay * 1000)))
            self.last_step_at = time.perf_counter()
        elif self.frozen:
            self._set_direction(direction)
            self._advance_clock(round(frame_period * 1000))
        else:
            self._set_direction(direction)
            self.page.wait_for_timeout(max(1, round(frame_period * 1000)))
            self._freeze()
        return self._raw_frame()

    def close(self) -> None:
        try:
            if self.page is not None:
                try:
                    self._activate()
                    self._release()
                except Exception:
                    pass
            if self.browser is not None:
                self.browser.close()
        finally:
            self.page = None
            self.context = None
            self.browser = None
            self.canvas = None
            self.canvas_frame = None
            self.direction = 0
            self.frozen = False
            self.started = False
            self.clock_time = time.time()
            if self.playwright is not None:
                self.playwright.stop()
                self.playwright = None
