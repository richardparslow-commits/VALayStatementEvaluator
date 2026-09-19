"""Offline tests for the VA.gov records download automation.

The Playwright driver is exercised against a fake browser: no Playwright install,
no network, no VA.gov account. The site itself cannot be tested here, so these
tests lock the parts that *can* be verified — the sign-in wait, the wizard
sequencing, filename hardening, PDF validation, and the failure messages that a
maintainer will see when VA.gov changes its markup.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from tests import hermetic  # noqa: E402,F401  (hermetic test session; see tests/hermetic.py)

import va_records_download as vrd  # noqa: E402

LOGIN_URL = "https://www.va.gov/?next=loginModal&oauth=true"
IDME_URL = "https://api.id.me/en/session/authentication_options"
MY_VA_URL = "https://www.va.gov/my-va/?postLogin=true"

# Selector ids that exist on the fake page, in the same format Candidate.describe()
# produces. The driver tries candidates in order, so an available set can hold any
# one of them per action.
DATE_RANGE = "label:all time"
CONTINUE = "role:button=continue"
SELECT_ALL = "label:select all"
PDF = "label:pdf"
DOWNLOAD = "role:button=download report"

WIZARD_SELECTORS = {DATE_RANGE, CONTINUE, SELECT_ALL, PDF, DOWNLOAD}


class _FakeDownload:
    def __init__(self, suggested: str, payload: bytes) -> None:
        self.suggested_filename = suggested
        self._payload = payload
        self.saved_to: Path | None = None

    def save_as(self, path: str) -> None:
        self.saved_to = Path(path)
        self.saved_to.write_bytes(self._payload)


class _FakeLocator:
    def __init__(self, page: "_FakePage", candidate: vrd.Candidate, matched: bool) -> None:
        self._page = page
        self._candidate = candidate
        self._matched = matched

    @property
    def first(self) -> "_FakeLocator":
        return self

    def count(self) -> int:
        return 1 if self._matched else 0

    def is_visible(self) -> bool:
        return self._matched

    def click(self, timeout: int | None = None) -> None:
        self._page.actions.append(("click", self._candidate.describe()))

    def check(self, timeout: int | None = None) -> None:
        if self._candidate.describe() in self._page.check_raises:
            # Mirrors Playwright: check() throws on anything that is not an input.
            raise RuntimeError("Element is not a checkbox or radio")
        self._page.actions.append(("check", self._candidate.describe()))


class _FakeDownloadCapture:
    def __init__(self, page: "_FakePage") -> None:
        self._page = page

    def __enter__(self) -> "_FakeDownloadCapture":
        self._page.actions.append(("expect_download", "download"))
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    @property
    def value(self) -> _FakeDownload:
        return self._page.download


class _FakePage:
    """Just enough of Playwright's Page API for the driver, fully observable."""

    def __init__(
        self,
        available: set[str] | None = None,
        *,
        signed_in: bool = True,
        url: str = "",
        url_sequence: list[str] | None = None,
        check_raises: set[str] | None = None,
        suggested_filename: str = "VA_medical_records.pdf",
        payload: bytes = b"%PDF-1.7\nrecord\n%%EOF\n",
        polls_before_render: int = 0,
    ) -> None:
        # `available=set()` must mean "no elements", so only None takes the default.
        self.available = set(WIZARD_SELECTORS if available is None else available)
        self.polls_before_render = polls_before_render
        self.signed_in = signed_in
        self._url = url or vrd.START_URL
        self.url_sequence = list(url_sequence or [])
        self.check_raises = set(check_raises or set())
        self.actions: list[tuple[str, str]] = []
        self.screenshots: list[str] = []
        self.download = _FakeDownload(suggested_filename, payload)

    @property
    def url(self) -> str:
        """Current URL: the queued timeline while signed out, else the last navigation.

        Observing a signed-in URL means the human completed ID.me + MFA, after which
        the browser keeps that session — the model VA.gov actually has.
        """
        if self.url_sequence:
            # Hold the final value so a poll loop cannot exhaust the script.
            value = (
                self.url_sequence.pop(0)
                if len(self.url_sequence) > 1
                else self.url_sequence[0]
            )
        else:
            value = self._url
        if not self.signed_in and vrd.is_signed_in_url(value):
            self.signed_in = True
        return value

    def goto(self, url: str, wait_until: str | None = None, timeout: int | None = None) -> None:
        self.actions.append(("goto", url))
        if self.signed_in:
            self.url_sequence.clear()
            self._url = url
        else:
            # Signed out: VA.gov redirects to the login page, and the queued timeline
            # supplies what the browser shows once the user finishes signing in.
            self._url = LOGIN_URL

    def _resolve(self, candidate: vrd.Candidate) -> _FakeLocator:
        if self.polls_before_render > 0:
            # The client-side app has not rendered this route yet: no elements exist,
            # which is precisely the state that a single is_visible() cannot tell
            # apart from "the markup changed".
            self.polls_before_render -= 1
            return _FakeLocator(self, candidate, False)
        present = self._elements_present()
        return _FakeLocator(self, candidate, present and candidate.describe() in self.available)

    def _elements_present(self) -> bool:
        """Wizard elements exist only on the wizard route, as in the real SPA."""
        return vrd.is_wizard_url(self.url)

    def title(self) -> str:
        return "Download your medical records | Veterans Affairs"

    def inner_text(self, selector: str) -> str:
        return "Download your medical records All time Date range Continue"

    def nth(self, index: int) -> "_FakeLocator":
        return self._resolve(vrd.Candidate("css", f"nth-{index}"))

    def get_attribute(self, name: str) -> str | None:
        return None

    def get_by_role(self, role: str, name: object = None) -> _FakeLocator:
        pattern = getattr(name, "pattern", str(name))
        return self._resolve(vrd.Candidate("role", str(pattern), role=role))

    def get_by_label(self, text: object) -> _FakeLocator:
        return self._resolve(vrd.Candidate("label", str(getattr(text, "pattern", text))))

    def get_by_text(self, text: object) -> _FakeLocator:
        return self._resolve(vrd.Candidate("text", str(getattr(text, "pattern", text))))

    def locator(self, selector: str) -> _FakeLocator:
        return self._resolve(vrd.Candidate("css", selector))

    def expect_download(self, timeout: int | None = None) -> _FakeDownloadCapture:
        return _FakeDownloadCapture(self)

    def screenshot(self, path: str, full_page: bool = False) -> None:
        self.screenshots.append(path)
        Path(path).write_bytes(b"png")

    def content(self) -> str:
        return "<html>signed-in page</html>"


def _downloader(page: _FakePage, tmp: Path, **over: object) -> vrd.VaGovDownloader:
    kwargs: dict[str, object] = {
        "out_dir": tmp / "out",
        "artifacts_dir": tmp / "artifacts",
        "wait_login_seconds": 5,
        "poll_seconds": 0,
        # No real waiting in unit tests: the behaviour under test is *what* the
        # driver does with a given page state, not how long it waits.
        "ready_timeout_ms": 0,
        "step_timeout_ms": 0,
        "log": lambda *_a, **_k: None,
    }
    kwargs.update(over)
    return vrd.VaGovDownloader(page, **kwargs)  # type: ignore[arg-type]


class TestUrlClassification(unittest.TestCase):
    def test_signed_in_pages(self) -> None:
        for url in (
            vrd.START_URL,
            "https://www.va.gov/my-health/medical-records/",
            MY_VA_URL,
            "https://va.gov/my-va/",
        ):
            self.assertTrue(vrd.is_signed_in_url(url), msg=url)

    def test_login_pages_are_not_signed_in(self) -> None:
        for url in (
            LOGIN_URL,
            IDME_URL,
            "https://www.va.gov/?oauth=true",
            "https://www.va.gov/loginModal/",
        ):
            self.assertFalse(vrd.is_signed_in_url(url), msg=url)

    def test_unrelated_pages_are_not_signed_in(self) -> None:
        for url in (
            "https://www.va.gov/",
            "https://example.com/my-health/",
            "https://www.va.gov/my-healthcare-benefits/",  # prefix, not a real match
        ):
            self.assertFalse(vrd.is_signed_in_url(url), msg=url)

    def test_wizard_url_matches_entry_page(self) -> None:
        self.assertTrue(vrd.is_wizard_url(vrd.START_URL))
        self.assertTrue(vrd.is_wizard_url(vrd.START_URL + "/"))
        self.assertFalse(vrd.is_wizard_url("https://www.va.gov/my-health/medical-records/"))


class TestFilenameHardening(unittest.TestCase):
    def test_directory_components_and_traversal_are_stripped(self) -> None:
        self.assertEqual(vrd.safe_pdf_name("../../etc/passwd"), "passwd.pdf")
        self.assertEqual(vrd.safe_pdf_name("C:\\Users\\me\\report.pdf"), "report.pdf")
        self.assertEqual(vrd.safe_pdf_name("/tmp/../report.pdf"), "report.pdf")

    def test_hostile_names_cannot_escape_the_output_directory(self) -> None:
        for hostile in ("../evil.pdf", "..\\evil.pdf", "/etc/passwd", "..", ".hidden"):
            name = vrd.safe_pdf_name(hostile)
            self.assertTrue(name.lower().endswith(".pdf"), msg=name)
            self.assertNotIn("..", name, msg=name)
            self.assertNotIn("/", name, msg=name)
            self.assertNotIn("\\", name, msg=name)
            self.assertFalse(name.startswith("."), msg=name)

    def test_extension_is_forced(self) -> None:
        self.assertEqual(vrd.safe_pdf_name("report"), "report.pdf")
        self.assertEqual(vrd.safe_pdf_name("report.pdf"), "report.pdf")
        self.assertEqual(vrd.safe_pdf_name("report.PDF"), "report.PDF")

    def test_empty_names_get_a_default(self) -> None:
        self.assertEqual(vrd.safe_pdf_name(""), "va_medical_records.pdf")
        self.assertEqual(vrd.safe_pdf_name(".."), "va_medical_records.pdf")

    def test_unusual_characters_are_collapsed(self) -> None:
        self.assertEqual(vrd.safe_pdf_name("va$records.pdf"), "va_records.pdf")

    def test_spaces_and_unicode_are_collapsed(self) -> None:
        self.assertEqual(
            vrd.safe_pdf_name("va medical records 2026.pdf"), "va_medical_records_2026.pdf"
        )


class TestPdfValidation(unittest.TestCase):
    def test_accepts_a_pdf_magic_header(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.pdf"
            path.write_bytes(b"%PDF-1.4\n...")
            self.assertTrue(vrd.looks_like_pdf(path))

    def test_rejects_error_pages_and_empty_files(self) -> None:
        with TemporaryDirectory() as tmp:
            html = Path(tmp) / "error.pdf"
            html.write_bytes(b"<!DOCTYPE html><html>Sign in</html>")
            empty = Path(tmp) / "empty.pdf"
            empty.write_bytes(b"")
            self.assertFalse(vrd.looks_like_pdf(html))
            self.assertFalse(vrd.looks_like_pdf(empty))
            self.assertFalse(vrd.looks_like_pdf(Path(tmp) / "missing.pdf"))


class TestDownloadInspection(unittest.TestCase):
    """A downloaded PDF is checked for being *the* record set, not just a PDF."""

    def _pdf(self, path: Path, pages: list[str]) -> None:
        import io

        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        for text in pages:
            y = 760
            for line in (text.split("\n") if text else []):
                pdf.drawString(72, y, line)
                y -= 14
            pdf.showPage()
        pdf.save()
        path.write_bytes(buffer.getvalue())

    def test_summary_counts_text_and_image_pages(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.pdf"
            self._pdf(path, ["Knee pain noted.", "", "Tinnitus noted."])
            summary = vrd.pdf_summary(path)
        self.assertEqual(summary["pages"], 3)
        self.assertEqual(summary["text_pages"], 2)
        self.assertEqual(summary["image_only_pages"], 1)
        self.assertAlmostEqual(summary["text_ratio"], 0.667, places=3)
        self.assertGreater(summary["bytes"], 0)

    def test_a_tiny_export_is_flagged_as_possibly_incomplete(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.pdf"
            self._pdf(path, ["One page."])
            warnings = vrd.report_download(path)
            manifest = json.loads(
                (Path(tmp) / "records.pdf.manifest.json").read_text(encoding="utf-8")
            )
        self.assertTrue(any("incomplete" in w for w in warnings))
        self.assertEqual(manifest["selections"]["date_range"], "All time")
        self.assertEqual(manifest["selections"]["record_type"], "Select all VA records")
        self.assertEqual(manifest["content"]["pages"], 1)
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_a_mostly_scanned_export_points_at_the_ocr_script(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.pdf"
            self._pdf(path, ["Text.", "", "", ""])
            warnings = vrd.report_download(path)
        self.assertTrue(any("ocr_records.py" in w for w in warnings))

    def test_a_healthy_export_produces_no_warnings(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.pdf"
            self._pdf(path, [f"Encounter note {i}." for i in range(1, 9)])
            warnings = vrd.report_download(path)
        self.assertEqual(warnings, [])

    def test_an_unreadable_download_is_reported_not_crashed(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.pdf"
            path.write_bytes(b"%PDF-1.4 not really a pdf")
            warnings = vrd.report_download(path)
        self.assertEqual(len(warnings), 1)
        self.assertIn("could not read", warnings[0])


class TestWizardFlow(unittest.TestCase):
    def test_happy_path_clicks_every_step_and_saves_the_pdf(self) -> None:
        page = _FakePage()
        with TemporaryDirectory() as tmp:
            saved = _downloader(page, Path(tmp)).run()
            self.assertTrue(saved.exists())
            self.assertTrue(vrd.looks_like_pdf(saved))
            self.assertEqual(saved.parent, Path(tmp) / "out")
            self.assertEqual(
                page.actions,
                [
                    ("goto", vrd.START_URL),
                    ("check", DATE_RANGE),
                    ("click", CONTINUE),
                    ("check", SELECT_ALL),
                    ("click", CONTINUE),
                    ("check", PDF),
                    ("expect_download", "download"),
                    ("click", DOWNLOAD),
                ],
            )

    def test_non_input_elements_fall_back_to_click(self) -> None:
        page = _FakePage(check_raises={DATE_RANGE, PDF})
        with TemporaryDirectory() as tmp:
            _downloader(page, Path(tmp)).run()
        self.assertIn(("click", DATE_RANGE), page.actions)
        self.assertIn(("click", PDF), page.actions)

    def test_dry_run_stops_before_downloading(self) -> None:
        page = _FakePage()
        with TemporaryDirectory() as tmp:
            result = _downloader(page, Path(tmp), dry_run=True).run()
        self.assertEqual(result, Path())
        self.assertNotIn(("expect_download", "download"), page.actions)
        self.assertNotIn(("click", DOWNLOAD), page.actions)
        self.assertIn(("check", PDF), page.actions)

    def test_fallback_selectors_are_tried_in_order(self) -> None:
        # VA.gov as rendered with the design-system components: only the CSS
        # variants exist, so the earlier candidates must be skipped silently.
        page = _FakePage(
            {
                "css:va-button[text='Continue']",
                "css:va-radio[value='allTime']",
                "css:va-radio[value='all']",
                "css:va-radio[value='pdf']",
                "css:va-button[text='Download report']",
            }
        )
        with TemporaryDirectory() as tmp:
            saved = _downloader(page, Path(tmp)).run()
            self.assertTrue(saved.exists())
        self.assertIn(("check", "css:va-radio[value='pdf']"), page.actions)

    def test_missing_element_names_the_step_and_dumps_artifacts(self) -> None:
        page = _FakePage({CONTINUE, SELECT_ALL, PDF, DOWNLOAD})  # date-range option gone
        with TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            with self.assertRaises(vrd.StepFailure) as ctx:
                _downloader(page, Path(tmp)).run()
            # Diagnostics: screenshot, page HTML, and the URL/title it was on.
            self.assertEqual(len(page.screenshots), 1)
            self.assertEqual(
                sorted(p.suffix for p in artifacts.iterdir()),
                [".html", ".png", ".txt"],
            )
        message = str(ctx.exception)
        self.assertIn("date-range", message)
        self.assertIn("All time", message)
        self.assertIn("label:all time", message)
        self.assertIn("--pause", message)

    def test_download_that_is_not_a_pdf_is_reported_and_kept(self) -> None:
        page = _FakePage(payload=b"<!DOCTYPE html><html>Session expired</html>")
        with TemporaryDirectory() as tmp:
            with self.assertRaises(vrd.DownloadError) as ctx:
                _downloader(page, Path(tmp)).run()
            kept = sorted(p.name for p in (Path(tmp) / "out").iterdir())
        self.assertIn("not a PDF", str(ctx.exception))
        self.assertEqual(kept, ["VA_medical_records.pdf"])
        self.assertIn("kept", str(ctx.exception))


class TestSignInWait(unittest.TestCase):
    def test_already_signed_in_skips_the_wait(self) -> None:
        page = _FakePage(signed_in=True)
        with TemporaryDirectory() as tmp:
            _downloader(page, Path(tmp)).run()
        self.assertEqual(page.actions.count(("goto", vrd.START_URL)), 1)

    def test_waits_for_sign_in_then_opens_the_wizard(self) -> None:
        # goto() lands on the login page; the poll loop then sees the post-login
        # page and re-enters the wizard.
        page = _FakePage(signed_in=False, url_sequence=[LOGIN_URL, MY_VA_URL])
        with TemporaryDirectory() as tmp:
            saved = _downloader(page, Path(tmp), wait_login_seconds=30).run()
            self.assertTrue(saved.exists())
            self.assertTrue(vrd.looks_like_pdf(saved))
        self.assertEqual(
            [action for action in page.actions if action[0] == "goto"],
            [("goto", vrd.START_URL), ("goto", vrd.START_URL)],
        )

    def test_wizard_url_with_nothing_rendered_is_not_reported_as_signed_in(self) -> None:
        """Regression: the live site loads the wizard URL then redirects to the login
        modal, so a URL-only check used to claim "Already signed in" and then fail on
        the first element. On the right route with no content, the step must name the
        missing element instead — and the driver must never claim a sign-in."""
        logs: list[str] = []
        page = _FakePage(available=set())  # right route, nothing rendered
        with TemporaryDirectory() as tmp:
            with self.assertRaises(vrd.StepFailure) as ctx:
                _downloader(page, Path(tmp), log=logs.append, wait_login_seconds=0).run()
        message = str(ctx.exception)
        self.assertIn("date-range", message)
        self.assertIn(vrd.START_URL, message)  # tells the operator where it was
        self.assertFalse(any("Signed in" in line for line in logs))
        self.assertTrue(any("has not rendered yet" in line for line in logs))

    def test_element_that_renders_a_moment_later_is_found(self) -> None:
        """A client-side route renders asynchronously; the driver must poll."""
        page = _FakePage(polls_before_render=2)
        with TemporaryDirectory() as tmp:
            saved = _downloader(page, Path(tmp), ready_timeout_ms=2_000).run()
            self.assertTrue(vrd.looks_like_pdf(saved))

    def test_login_timeout_explains_the_manual_fallback(self) -> None:
        page = _FakePage(signed_in=False, url_sequence=[IDME_URL])
        with TemporaryDirectory() as tmp:
            with self.assertRaises(vrd.LoginTimeout) as ctx:
                _downloader(page, Path(tmp), wait_login_seconds=0).run()
        message = str(ctx.exception)
        self.assertIn("--wait-login", message)
        self.assertIn("by hand", message)
        self.assertNotIn("check", [action[0] for action in page.actions])


class _FakeContext:
    def __init__(self, pages: list[object]) -> None:
        self.pages = pages
        self.new_pages = 0

    def new_page(self) -> object:
        self.new_pages += 1
        return object()


class _FakeBrowser:
    def __init__(self, contexts: list[_FakeContext] | None = None) -> None:
        self.contexts = contexts or []
        self.new_contexts = 0
        self.closed = False

    def new_context(self) -> _FakeContext:
        self.new_contexts += 1
        return _FakeContext([])

    def new_page(self) -> object:
        return object()

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, contexts: list[_FakeContext] | None = None) -> None:
        self.browser = _FakeBrowser(contexts)
        self.context = _FakeContext([object()])
        self.connected_to: str | None = None
        self.launch_kwargs: dict | None = None
        self.persistent_kwargs: dict | None = None

    def connect_over_cdp(self, url: str) -> _FakeBrowser:
        self.connected_to = url
        return self.browser

    def launch(self, **kwargs: object) -> _FakeBrowser:
        self.launch_kwargs = dict(kwargs)
        return self.browser

    def launch_persistent_context(self, profile: str, **kwargs: object) -> _FakeContext:
        self.persistent_kwargs = {"profile": profile, **kwargs}
        return self.context


class _FakePlaywright:
    def __init__(self, contexts: list[_FakeContext] | None = None) -> None:
        self.chromium = _FakeChromium(contexts)


def _args(*argv: str):
    return vrd.build_parser().parse_args(list(argv))


class TestInspectMode(unittest.TestCase):
    """--inspect is the maintenance path: it must click nothing and say which
    candidate selectors matched, so a broken step can be fixed from its report."""

    def test_reports_matches_and_click_nothing(self) -> None:
        page = _FakePage()
        with TemporaryDirectory() as tmp:
            path = _downloader(page, Path(tmp), inspect_only=True).run()
            report = path.read_text(encoding="utf-8")
            self.assertTrue(path.name.startswith("inspect-"))
        self.assertIn("date-range option: ['label:all time']", report)
        self.assertIn("continue button: ['role:button=continue']", report)
        self.assertIn("download button: ['role:button=download report']", report)
        self.assertIn(vrd.START_URL, report)
        self.assertEqual([a for a in page.actions if a[0] in ("click", "check")], [])

    def test_reports_no_match_when_the_markup_changed(self) -> None:
        page = _FakePage(available=set())
        with TemporaryDirectory() as tmp:
            path = _downloader(page, Path(tmp), inspect_only=True).run()
            report = path.read_text(encoding="utf-8")
        for label in ("date-range option", "continue button", "download button"):
            self.assertIn(f"{label}: NO MATCH", report)

    def test_inspect_does_not_need_the_known_markers_to_render(self) -> None:
        # The point of --inspect is to work when nothing matches, so it is satisfied
        # by being on the wizard route even though no marker is visible.
        logs: list[str] = []
        page = _FakePage(available=set())
        with TemporaryDirectory() as tmp:
            _downloader(page, Path(tmp), inspect_only=True, log=logs.append).run()
        self.assertTrue(any("dumping the page for inspection" in line for line in logs))
        self.assertFalse(any("Timed out" in line for line in logs))


class TestBrowserSelection(unittest.TestCase):
    """--cdp must attach to a browser the human owns, and never close it."""

    def test_cdp_attaches_to_the_existing_context(self) -> None:
        existing_page = object()
        playwright = _FakePlaywright([_FakeContext([existing_page])])
        browser, page, owned = vrd._open_browser(
            playwright, _args("--cdp", "http://127.0.0.1:9222")
        )
        self.assertEqual(playwright.chromium.connected_to, "http://127.0.0.1:9222")
        self.assertIs(page, existing_page)
        self.assertFalse(owned)
        self.assertIsNone(playwright.chromium.launch_kwargs)
        self.assertIsNone(playwright.chromium.persistent_kwargs)
        self.assertFalse(browser.closed)

    def test_cdp_without_an_open_context_creates_one(self) -> None:
        playwright = _FakePlaywright([])
        _, _, owned = vrd._open_browser(playwright, _args("--cdp", "http://127.0.0.1:9222"))
        self.assertEqual(playwright.chromium.browser.new_contexts, 1)
        self.assertFalse(owned)

    def test_default_launches_a_persistent_profile(self) -> None:
        playwright = _FakePlaywright()
        _, _, owned = vrd._open_browser(playwright, _args())
        self.assertEqual(playwright.chromium.persistent_kwargs["profile"], str(vrd.DEFAULT_PROFILE_DIR))
        self.assertFalse(playwright.chromium.persistent_kwargs["headless"])
        self.assertTrue(owned)

    def test_no_persist_launches_a_throwaway_browser(self) -> None:
        playwright = _FakePlaywright()
        browser, _, owned = vrd._open_browser(playwright, _args("--no-persist"))
        self.assertEqual(playwright.chromium.launch_kwargs, {"headless": False})
        self.assertTrue(owned)
        browser.close()
        self.assertTrue(browser.closed)


class TestPlaywrightSetup(unittest.TestCase):
    def test_missing_playwright_explains_the_install(self) -> None:
        with patch.dict(sys.modules, {"playwright": None, "playwright.sync_api": None}):
            with self.assertRaises(vrd.PlaywrightMissing) as ctx:
                vrd._load_playwright()
        message = str(ctx.exception)
        self.assertIn("requirements-local.txt", message)
        self.assertIn("playwright install chromium", message)

    def test_cli_defaults(self) -> None:
        args = vrd.build_parser().parse_args([])
        self.assertTrue(vrd.default_out_dir().is_dir())
        self.assertEqual(args.out, vrd.default_out_dir())
        self.assertFalse(args.dry_run)
        self.assertEqual(args.wait_login, vrd.DEFAULT_WAIT_LOGIN_SECONDS)
        self.assertEqual(args.start_url, vrd.START_URL)


if __name__ == "__main__":
    unittest.main()
