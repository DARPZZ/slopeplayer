from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from slope_core.browser import BrowserConfig, BrowserError, SlopeBrowser, normalize_game_url


def make_browser(*, headless: bool = True, reset_attempts: int = 3) -> SlopeBrowser:
    return SlopeBrowser(
        BrowserConfig(
            url="https://da.y8.com/games/slope",
            headless=headless,
            reset_attempts=reset_attempts,
        )
    )


def running_frame() -> np.ndarray:
    frame = np.zeros((400, 640, 3), dtype=np.uint8)
    frame[100:300, 100:300] = (0, 255, 0)
    frame[120:300, 400:500] = (0, 0, 255)
    return frame


def interstitial_frame(*, close_ready: bool) -> np.ndarray:
    frame = running_frame()
    cv2.rectangle(frame, (106, 124), (533, 279), (255, 255, 255), -1)
    if close_ready:
        cv2.putText(
            frame,
            "Close",
            (497, 111),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_8,
        )
    return frame


def compact_interstitial_frame() -> np.ndarray:
    frame = running_frame()
    cv2.rectangle(frame, (214, 92), (426, 312), (245, 245, 245), -1)
    cv2.putText(
        frame,
        "Close",
        (388, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_8,
    )
    return frame


class BrowserTests(unittest.TestCase):
    def test_normalizes_localized_y8_game_and_embed_urls(self) -> None:
        expected = "https://www.y8.com/embed/slope"
        self.assertEqual(normalize_game_url("https://da.y8.com/games/slope"), expected)
        self.assertEqual(normalize_game_url("https://y8.com/embed/slope"), expected)
        self.assertEqual(
            normalize_game_url("https://example.com/games/slope"),
            "https://example.com/games/slope",
        )
        self.assertEqual(
            normalize_game_url("https://da.y8.com/games/not/a/slug"),
            "https://da.y8.com/games/not/a/slug",
        )

    def test_channel_and_key_layout_selection(self) -> None:
        bundled = make_browser()
        self.assertIsNone(bundled.playwright_channel)
        chrome = SlopeBrowser(
            BrowserConfig(
                url="https://example.com/slope",
                channel="chrome",
                key_layout="ad",
            )
        )
        self.assertEqual(chrome.playwright_channel, "chrome")
        self.assertEqual((chrome.left_key, chrome.right_key), ("a", "d"))

    def test_direction_changes_release_the_previous_key(self) -> None:
        browser = make_browser()
        browser.page = Mock()

        browser._set_direction(-1)
        browser._set_direction(-1)
        browser._set_direction(1)
        browser._set_direction(0)

        self.assertEqual(browser.page.keyboard.down.call_args_list, [call("ArrowLeft"), call("ArrowRight")])
        self.assertEqual(browser.page.keyboard.up.call_args_list, [call("ArrowLeft"), call("ArrowRight")])
        self.assertEqual(browser.direction, 0)

    def test_release_keys_releases_both_even_when_state_is_stale(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.direction = 1
        browser._release_keys()
        self.assertEqual(
            browser.page.keyboard.up.call_args_list,
            [call("ArrowLeft"), call("ArrowRight")],
        )
        self.assertEqual(browser.direction, 0)

    def test_every_headless_physics_advance_is_split_at_16_ms(self) -> None:
        browser = make_browser(headless=True)
        browser.page = Mock()
        browser.clock_frozen = True

        browser._advance_physics_ms(49)

        self.assertEqual(
            browser.page.clock.run_for.call_args_list,
            [call(16), call(16), call(16), call(1)],
        )

    def test_visible_physics_uses_identical_slices_and_only_wall_paces(self) -> None:
        browser = make_browser(headless=False)
        browser.page = Mock()
        browser.clock_frozen = True

        with patch("slope_core.browser.time.sleep") as sleep:
            browser._advance_physics_ms(33)

        self.assertEqual(
            browser.page.clock.run_for.call_args_list,
            [call(16), call(16), call(1)],
        )
        self.assertTrue(sleep.called)

    def test_initial_reset_uses_start_click_and_only_three_physics_frames(self) -> None:
        browser = make_browser(reset_attempts=1)
        browser.page = Mock()
        browser.clock_frozen = True
        browser._launch = Mock()
        events: list[str] = []
        browser._click_canvas = Mock(side_effect=lambda *_: events.append("click"))
        browser._park_mouse = Mock(side_effect=lambda: events.append("park"))
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        browser._wait_for_running_frame = Mock(
            side_effect=lambda: events.append("running") or frame
        )

        result = browser.reset("initial")

        self.assertIs(result, frame)
        browser._click_canvas.assert_called_once_with(0.50, 0.462)
        browser._park_mouse.assert_called_once_with()
        browser._wait_for_running_frame.assert_called_once_with()
        self.assertEqual(events, ["click", "running", "park"])
        self.assertTrue(browser.started)

    def test_terminated_reset_uses_again_click_and_same_small_budget(self) -> None:
        browser = make_browser(reset_attempts=1)
        browser.page = Mock()
        browser.clock_frozen = True
        browser.started = True
        browser._launch = Mock()
        events: list[str] = []
        browser._click_canvas = Mock(side_effect=lambda *_: events.append("click"))
        browser._park_mouse = Mock(side_effect=lambda: events.append("park"))
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        browser._wait_for_running_frame = Mock(
            side_effect=lambda: events.append("running") or frame
        )

        browser.reset("terminated")

        browser._click_canvas.assert_called_once_with(0.50, 0.908)
        browser._park_mouse.assert_called_once_with()
        browser._wait_for_running_frame.assert_called_once_with()
        self.assertEqual(events, ["click", "running", "park"])

    def test_mouse_is_parked_in_safe_canvas_corner(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.canvas = Mock()
        browser.canvas.bounding_box.return_value = {
            "x": 100,
            "y": 50,
            "width": 640,
            "height": 400,
        }

        browser._park_mouse()

        browser.page.mouse.move.assert_called_once_with(108.0, 58.0)

    def test_running_scene_uses_green_track_without_requiring_red_pixels(self) -> None:
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        frame[100:300, 100:300] = (0, 255, 0)
        self.assertTrue(SlopeBrowser._looks_like_running_game(frame))

    def test_running_scene_rejects_game_pixels_covered_by_interstitial(self) -> None:
        frame = interstitial_frame(close_ready=False)
        self.assertTrue(SlopeBrowser._looks_like_interstitial(frame))
        self.assertFalse(SlopeBrowser._looks_like_running_game(frame))

    def test_visual_ad_close_appears_only_after_close_label_is_ready(self) -> None:
        self.assertIsNone(
            SlopeBrowser._interstitial_close_point(
                interstitial_frame(close_ready=False)
            )
        )
        point = SlopeBrowser._interstitial_close_point(
            interstitial_frame(close_ready=True)
        )
        self.assertIsNotNone(point)
        assert point is not None
        self.assertLess(abs(point[0] - 515), 20)
        self.assertLess(abs(point[1] - 105), 15)

    def test_compact_interstitial_is_detected_and_has_close_point(self) -> None:
        frame = compact_interstitial_frame()
        self.assertTrue(SlopeBrowser._looks_like_interstitial(frame))
        point = SlopeBrowser._interstitial_close_point(frame)
        self.assertIsNotNone(point)
        assert point is not None
        self.assertLess(abs(point[0] - 407), 20)
        self.assertLess(abs(point[1] - 75), 15)

    def test_interstitial_waits_for_close_then_uses_visual_fallback(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.canvas = Mock()
        browser.canvas.bounding_box.return_value = {
            "x": 0,
            "y": 0,
            "width": 640,
            "height": 400,
        }
        browser.clock_frozen = True
        waiting = interstitial_frame(close_ready=False)
        ready = interstitial_frame(close_ready=True)
        game = running_frame()
        browser._visible_locator = Mock(return_value=None)
        browser._advance_physics_ms = Mock()
        browser._click_resume_control = Mock()
        browser._capture = Mock(side_effect=[ready, game, game])

        with patch("builtins.print"):
            result = browser._dismiss_interstitial(waiting)

        self.assertIs(result, game)
        self.assertEqual(
            browser._advance_physics_ms.call_args_list,
            [call(250), call(48)],
        )
        browser.page.mouse.click.assert_called_once()
        browser._click_resume_control.assert_called_once_with()
        self.assertEqual(browser.direction, 0)

    def test_interstitial_prefers_explicit_dom_close_control(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.clock_frozen = True
        close = Mock()
        browser._visible_locator = Mock(return_value=close)
        browser._advance_physics_ms = Mock()
        browser._click_resume_control = Mock()
        game = running_frame()
        browser._capture = Mock(return_value=game)

        with patch("builtins.print"):
            result = browser._dismiss_interstitial(
                interstitial_frame(close_ready=False)
            )

        self.assertIs(result, game)
        close.click.assert_called_once_with(timeout=750)
        browser._advance_physics_ms.assert_called_once_with(48)
        browser._click_resume_control.assert_called_once_with()

    def test_reset_transition_dismisses_ad_before_accepting_gameplay(self) -> None:
        browser = make_browser()
        ad = interstitial_frame(close_ready=False)
        game = running_frame()
        browser._advance_physics_ms = Mock()
        browser._capture = Mock(return_value=ad)
        browser._dismiss_interstitial = Mock(return_value=game)

        result = browser._wait_for_running_frame()

        self.assertIs(result, game)
        browser._dismiss_interstitial.assert_called_once_with(ad)

    def test_step_dismisses_ad_and_restores_requested_direction(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.started = True
        browser.clock_frozen = True
        ad = interstitial_frame(close_ready=False)
        game = running_frame()
        browser._capture = Mock(return_value=ad)

        def dismiss(_frame: np.ndarray) -> np.ndarray:
            browser._release_keys()
            return game

        browser._dismiss_interstitial = Mock(side_effect=dismiss)

        result = browser.step(1, 0.05)

        self.assertIs(result, game)
        browser._dismiss_interstitial.assert_called_once_with(ad)
        self.assertEqual(
            browser.page.keyboard.down.call_args_list,
            [call("ArrowRight"), call("ArrowRight")],
        )
        self.assertEqual(browser.direction, 1)

    def test_reset_transition_polls_in_three_frame_chunks(self) -> None:
        browser = make_browser()
        transition = np.zeros((400, 640, 3), dtype=np.uint8)
        running = transition.copy()
        running[100:300, 100:300] = (0, 255, 0)
        running[100:300, 400:500] = (0, 0, 255)
        browser._advance_physics_ms = Mock()
        browser._capture = Mock(side_effect=[transition, running])
        result = browser._wait_for_running_frame()
        self.assertIs(result, running)
        self.assertEqual(
            browser._advance_physics_ms.call_args_list,
            [call(48), call(48)],
        )

    def test_reset_relaunches_after_a_transient_failure(self) -> None:
        browser = make_browser(reset_attempts=2)
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        browser._launch = Mock()
        browser._reset_once = Mock(side_effect=[RuntimeError("canvas race"), frame])
        browser.close = Mock()

        with patch("builtins.print"):
            result = browser.reset("initial")

        self.assertIs(result, frame)
        self.assertEqual(browser._launch.call_count, 2)
        self.assertEqual(browser._reset_once.call_count, 2)
        browser.close.assert_called_once_with()

    def test_reset_reports_final_failure_with_original_cause(self) -> None:
        browser = make_browser(reset_attempts=2)
        browser._launch = Mock(side_effect=RuntimeError("cannot launch"))
        browser.close = Mock()

        with patch("builtins.print"), self.assertRaises(BrowserError) as caught:
            browser.reset("initial")

        self.assertIn("reset failed after 2 attempts", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_popup_handler_preserves_owner_and_closes_every_other_page(self) -> None:
        browser = make_browser(headless=False)
        browser.page = Mock()
        popup = Mock()

        browser._close_popup(browser.page)
        browser._close_popup(popup)

        browser.page.close.assert_not_called()
        popup.close.assert_called_once_with()
        browser.page.bring_to_front.assert_called_once_with()

    def test_popup_during_game_requests_one_resume_click(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.canvas = Mock()
        browser.started = True
        browser.clock_frozen = True
        popup = Mock()
        browser._click_canvas = Mock()
        browser._advance_physics_ms = Mock()
        browser._park_mouse = Mock()

        browser._close_popup(popup)
        first = browser._resume_after_closed_popup()
        second = browser._resume_after_closed_popup()

        self.assertTrue(first)
        self.assertFalse(second)
        browser._click_canvas.assert_called_once_with(0.50, 0.50)
        browser._advance_physics_ms.assert_called_once_with(48)
        browser._park_mouse.assert_called_once_with()

    def test_popup_handlers_cover_tabs_page_popups_and_dialogs(self) -> None:
        browser = make_browser()
        browser.context = Mock()
        browser.page = Mock()

        browser._install_popup_handlers()

        browser.context.on.assert_called_once_with("page", browser._close_popup)
        self.assertEqual(
            browser.page.on.call_args_list,
            [
                call("popup", browser._close_popup),
                call("dialog", browser._dismiss_dialog),
            ],
        )

    def test_dialog_handler_dismisses_javascript_prompt(self) -> None:
        dialog = Mock()
        SlopeBrowser._dismiss_dialog(dialog)
        dialog.dismiss.assert_called_once_with()

    def test_capture_decodes_playwright_bytes_as_bgr(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.canvas = Mock()
        box = {"x": 2, "y": 3, "width": 16, "height": 12}
        browser.canvas.bounding_box.return_value = box
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        source[:, :] = (7, 80, 240)
        browser.page.screenshot.return_value = cv2.imencode(".png", source)[1].tobytes()

        result = browser._capture()

        np.testing.assert_array_equal(result, source)
        browser.page.screenshot.assert_called_once_with(
            clip=box,
            type="jpeg",
            timeout=browser.config.operation_timeout_ms,
            quality=browser.config.jpeg_quality,
        )

    def test_menu_readiness_rejects_unity_splash_and_accepts_play_screen(self) -> None:
        splash = np.zeros((400, 640, 3), dtype=np.uint8)
        menu = splash.copy()
        cv2.putText(menu, "SLOPE", (220, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.rectangle(menu, (220, 160), (420, 210), (0, 0, 255), 3)
        cv2.putText(menu, "Play", (270, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        self.assertFalse(SlopeBrowser._looks_like_menu(splash))
        self.assertTrue(SlopeBrowser._looks_like_menu(menu))

    def test_privacy_panel_finds_large_card_not_small_white_ui(self) -> None:
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        frame[10:30, 10:80] = 255
        self.assertIsNone(SlopeBrowser._privacy_panel(frame))
        frame[80:340, 100:540] = 255
        panel = SlopeBrowser._privacy_panel(frame)
        self.assertIsNotNone(panel)
        assert panel is not None
        x, y, width, height = panel
        self.assertLessEqual(abs(x - 100), 2)
        self.assertLessEqual(abs(y - 80), 2)
        self.assertGreaterEqual(width, 438)
        self.assertGreaterEqual(height, 258)

    def test_consent_prefers_explicit_reject_button(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        reject = Mock()
        browser._visible_locator = Mock(return_value=reject)
        browser._capture = Mock()
        browser._dismiss_consent()
        reject.click.assert_called_once()
        browser._capture.assert_not_called()

    def test_wait_for_menu_advances_only_pre_game_clock(self) -> None:
        browser = make_browser()
        browser.clock_frozen = True
        splash = np.zeros((400, 640, 3), dtype=np.uint8)
        menu = splash.copy()
        menu[20:80, 200:440] = (0, 255, 0)
        menu[160:210, 220:420] = (0, 0, 255)
        browser._capture = Mock(side_effect=[splash, menu])
        browser._advance_physics_ms = Mock()
        browser._wait_for_menu_pixels()
        browser._advance_physics_ms.assert_called_once_with(64)

        browser.clock_frozen = False
        browser.page = Mock()
        browser._capture = Mock(side_effect=[splash, menu])
        browser._advance_physics_ms.reset_mock()
        browser._wait_for_menu_pixels()
        browser._advance_physics_ms.assert_not_called()
        browser.page.wait_for_timeout.assert_called_once_with(64)

    def test_step_requires_reset_and_valid_duration(self) -> None:
        browser = make_browser()
        with self.assertRaises(BrowserError):
            browser.step(0, 1 / 30)
        browser.page = Mock()
        browser.started = True
        browser.clock_frozen = True
        with self.assertRaises(ValueError):
            browser.step(0, 0)
        with self.assertRaises(ValueError):
            browser.step(2, 1 / 30)

    def test_step_wraps_transport_failures_as_browser_errors(self) -> None:
        browser = make_browser()
        browser.page = Mock()
        browser.started = True
        browser.clock_frozen = True
        browser.page.clock.run_for.side_effect = RuntimeError("page crashed")

        with self.assertRaises(BrowserError) as caught:
            browser.step(0, 1 / 30)

        self.assertIn("browser step failed", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_close_releases_keys_and_clears_all_resources(self) -> None:
        browser = make_browser()
        page = Mock()
        context = Mock()
        playwright_browser = Mock()
        playwright = Mock()
        browser.page = page
        browser.context = context
        browser.browser = playwright_browser
        browser.playwright = playwright
        browser.canvas = Mock()
        browser.canvas_frame = Mock()
        browser.direction = -1
        browser.started = True
        browser.clock_frozen = True

        browser.close()

        self.assertEqual(
            page.keyboard.up.call_args_list,
            [call("ArrowLeft"), call("ArrowRight")],
        )
        context.close.assert_called_once_with()
        playwright_browser.close.assert_called_once_with()
        playwright.stop.assert_called_once_with()
        self.assertIsNone(browser.page)
        self.assertIsNone(browser.context)
        self.assertIsNone(browser.browser)
        self.assertIsNone(browser.playwright)
        self.assertFalse(browser.started)
        self.assertFalse(browser.clock_frozen)


if __name__ == "__main__":
    unittest.main()
