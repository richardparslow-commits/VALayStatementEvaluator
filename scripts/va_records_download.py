"""Download your own VA medical records from VA.gov, with you driving the login.

WHY THIS EXISTS
    Real VA.gov records sit behind ID.me sign-in plus SMS multi-factor
    authentication, and VA.gov publishes no patient-facing records API. The app's
    VA.gov source (``VA_GOV_API_BASE_URL``) therefore only ever talks to a
    sandbox or simulator. To obtain *your* complete record set you have to walk
    VA.gov's download wizard, which is what this script automates.

WHAT IT DOES — AND DOES NOT — DO
    * It opens a real, visible Chromium window and *waits* while you sign in to
      VA.gov through ID.me, including the SMS code. It never types, reads, or
      stores your email, password, or one-time code.
    * It automates only the wizard *after* you are signed in:
      date range "All time" -> "Select all VA records" -> PDF -> Download report,
      then saves the PDF into ``--out``.
    * There is no headless mode, no CAPTCHA or MFA bypass, and no credential
      file. Automating ID.me's sign-in itself would likely breach ID.me's terms;
      waiting for a human to sign in does not. You remain responsible for VA.gov's
      and ID.me's terms, and for using only your own records.
    * It writes a browser profile directory so your VA.gov session can be reused
      (fewer SMS rounds). That directory holds VA.gov session cookies on this
      machine — delete it to sign out, or pass ``--no-persist`` for a throwaway
      session. This is the one place a session is kept on disk; the Streamlit app
      never writes credentials or sessions anywhere.

SETUP
    .venv/bin/python -m pip install -r requirements-local.txt
    .venv/bin/python -m playwright install chromium

RUN
    .venv/bin/python scripts/va_records_download.py                    # ~/Desktop
    .venv/bin/python scripts/va_records_download.py --out ./records
    .venv/bin/python scripts/va_records_download.py --dry-run          # stop before download
    .venv/bin/python scripts/va_records_download.py --pause            # confirm each step

    # Or sign in with your own Chrome and let the script attach to it:
    #   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    #       --remote-debugging-port=9222 --user-data-dir="$HOME/.va_lse_debug_chrome"
    .venv/bin/python scripts/va_records_download.py --cdp http://127.0.0.1:9222

    A dedicated --user-data-dir is required: Chrome refuses remote debugging against
    its default profile, and the attached browser is never closed by this script.

IF A STEP FAILS (VA.gov changed its markup)
    The script names the failing step and writes a screenshot plus the page HTML to
    ``--artifacts-dir`` so the candidate selectors below can be updated. Treat that
    HTML as sensitive: it is a signed-in page and can contain your medical data.
    The manual route always works: sign in at https://www.va.gov/, then
    My Health -> Medical records -> Download -> date range "All time" ->
    "Select all VA records" -> PDF -> Download report.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Entry point of the wizard. VA.gov redirects to ID.me when the profile has no
# live session, which is how sign-in is detected below.
START_URL = "https://www.va.gov/my-health/medical-records/download/date-range"
VA_GOV_HOSTS = ("www.va.gov", "va.gov")
# Signed-in areas the browser lands on after ID.me + MFA (`/my-va/?postLogin=true`)
# and after navigating into the wizard (`/my-health/...`).
SIGNED_IN_PATHS = ("/my-health", "/my-va")
LOGIN_URL_MARKERS = ("loginmodal", "/oauth", "id.me")
DEFAULT_WAIT_LOGIN_SECONDS = 300
DEFAULT_STEP_TIMEOUT_MS = 20_000
# How long to give the client-side app to render a route before deciding that the
# session is not signed in (rather than that the markup changed).
DEFAULT_READY_TIMEOUT_MS = 8_000
# How often a step re-checks for its element while rendering is still in flight.
POLL_INTERVAL_SECONDS = 0.25
DEFAULT_PROFILE_DIR = Path.home() / ".va_lse_va_gov_profile"


# ---------------------------------------------------------------------------
# Selector candidates. VA.gov renders the VA Design System web components
# (`va-radio`, `va-button`, ...) inside shadow DOM, so each action lists several
# ways to find the same element — accessible role first, then label/text, then a
# CSS fallback. The first candidate that matches a visible element wins, and a
# step only fails when none of them do. Add candidates here when VA.gov changes.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One way to locate an element. ``kind`` picks the Playwright locator API."""

    kind: str  # "role" | "label" | "text" | "css"
    value: str
    role: str = ""  # for kind == "role"

    def describe(self) -> str:
        return f"{self.kind}:{self.role + '=' if self.role else ''}{self.value}"

    def locator(self, page: Any) -> Any:
        pattern = re.compile(self.value, re.IGNORECASE)
        if self.kind == "role":
            return page.get_by_role(self.role, name=pattern)
        if self.kind == "label":
            return page.get_by_label(pattern)
        if self.kind == "text":
            return page.get_by_text(pattern)
        return page.locator(self.value)


ALL_TIME_OPTIONS = (
    Candidate("label", "all time"),
    Candidate("role", "all time", role="radio"),
    Candidate("text", "all time"),
    Candidate("css", "va-radio[value='allTime']"),
    Candidate("css", "input[value='allTime']"),
)

SELECT_ALL_RECORDS_OPTIONS = (
    Candidate("label", "select all"),
    Candidate("role", "select all va records", role="radio"),
    Candidate("role", "select all", role="checkbox"),
    Candidate("text", "select all va records"),
    Candidate("css", "va-radio[value='all']"),
)

PDF_FILE_TYPE_OPTIONS = (
    Candidate("label", "pdf"),
    Candidate("role", "pdf", role="radio"),
    Candidate("text", "pdf"),
    Candidate("css", "va-radio[value='pdf']"),
    Candidate("css", "input[value='pdf']"),
)

CONTINUE_BUTTONS = (
    Candidate("role", "continue", role="button"),
    Candidate("css", "va-button[text='Continue']"),
    Candidate("css", "button:has-text('Continue')"),
)

DOWNLOAD_BUTTONS = (
    Candidate("role", "download report", role="button"),
    Candidate("css", "va-button[text='Download report']"),
    Candidate("css", "button:has-text('Download report')"),
)

# Anything that proves the date-range step actually rendered. VA.gov's wizard is a
# client-side route: the requested URL can load while the app is still hydrating,
# and an unauthenticated visit is redirected to `/?next=…&oauth=true` a moment
# later — so "the URL is right" is not evidence that the wizard is on screen.
WIZARD_MARKERS = ALL_TIME_OPTIONS + (
    Candidate("role", "download your medical records", role="heading"),
    Candidate("text", "date range"),
)


class DownloadError(RuntimeError):
    """Base class for every expected failure of the wizard run."""


class PlaywrightMissing(DownloadError):
    """Playwright is not installed in this interpreter."""


class LoginTimeout(DownloadError):
    """The user did not finish ID.me sign-in within the wait budget."""


class StepFailure(DownloadError):
    """No candidate selector matched a visible element for one wizard step."""

    def __init__(
        self,
        step: str,
        description: str,
        tried: tuple[Candidate, ...],
        url: str = "",
    ) -> None:
        tried_text = ", ".join(c.describe() for c in tried)
        where = f" The browser was at {url}." if url else ""
        super().__init__(
            f"Could not {description} (step '{step}'). None of these matched a visible "
            f"element: {tried_text}.{where} VA.gov may have changed its markup — re-run "
            "with --pause to inspect the page, or update the candidate selectors in "
            "scripts/va_records_download.py."
        )
        self.step = step
        self.description = description
        self.tried = tried
        self.url = url


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a browser).
# ---------------------------------------------------------------------------
def is_signed_in_url(url: str) -> bool:
    """True when ``url`` is a signed-in VA.gov page rather than a login page.

    VA.gov bounces unauthenticated visitors to ``/?next=loginModal&oauth=true``
    and then to ID.me; both must read as "not signed in" so the wait loop keeps
    waiting instead of clicking into a login form.  Path matching is by segment,
    so ``/my-healthcare-benefits`` is not mistaken for ``/my-health``.
    """
    lowered = url.lower()
    parsed = urlsplit(url)
    if parsed.netloc not in VA_GOV_HOSTS:
        return False
    if any(marker in lowered for marker in LOGIN_URL_MARKERS):
        return False
    return any(
        parsed.path == prefix or parsed.path.startswith(f"{prefix}/")
        for prefix in SIGNED_IN_PATHS
    )


def is_wizard_url(url: str) -> bool:
    """True when the browser is on the wizard entry page itself."""
    return urlsplit(url).path.rstrip("/") == urlsplit(START_URL).path.rstrip("/")


def safe_pdf_name(suggested: str) -> str:
    """Reduce a browser-suggested filename to a safe ``.pdf`` basename.

    The name comes from VA.gov's ``Content-Disposition`` header, which is remote
    input: strip any directory component (including Windows separators and
    traversal), collapse the rest to a conservative character set, and force a
    ``.pdf`` suffix so the file cannot be written outside ``--out``.
    """
    base = Path((suggested or "").replace("\\", "/")).name
    # Collapse anything exotic, then drop leading dots/dashes so the result cannot
    # read as a hidden file or a relative path.
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).lstrip("._-")
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf" if base else "va_medical_records.pdf"
    return base


def looks_like_pdf(path: Path) -> bool:
    """True when the file exists and starts with the ``%PDF`` magic bytes.

    A signed-out or errored wizard can hand back an HTML error page, which the
    browser happily saves with a ``.pdf`` name; this catches that before the file
    is handed to the app as a record set.
    """
    try:
        if path.stat().st_size == 0:
            return False
        with path.open("rb") as handle:
            return handle.read(5).startswith(b"%PDF")
    except OSError:
        return False


def _load_playwright() -> Any:
    """Return Playwright's ``sync_playwright`` entry point, or explain the fix."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised via test patches
        raise PlaywrightMissing(
            "Playwright is not installed for this interpreter. Install the local "
            "automation extras, then the browser binary:\n"
            "  python -m pip install -r requirements-local.txt\n"
            "  python -m playwright install chromium"
        ) from exc
    return sync_playwright


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class VaGovDownloader:
    """Walk the VA.gov download wizard on an already-authenticated page.

    The page is injected so the flow can be unit-tested against a fake browser:
    every Playwright call this class makes is one of ``page.url``, ``goto``,
    ``get_by_role``/``get_by_label``/``get_by_text``/``locator`` (returning a
    locator with ``count``/``first``/``is_visible``/``click``/``check``),
    ``expect_download``, ``screenshot``, and ``content``.
    """

    def __init__(
        self,
        page: Any,
        *,
        out_dir: Path,
        artifacts_dir: Path,
        wait_login_seconds: int = DEFAULT_WAIT_LOGIN_SECONDS,
        step_timeout_ms: int = DEFAULT_STEP_TIMEOUT_MS,
        ready_timeout_ms: int = DEFAULT_READY_TIMEOUT_MS,
        poll_seconds: float = 2.0,
        dry_run: bool = False,
        pause: bool = False,
        start_url: str = START_URL,
        log: Any = print,
    ) -> None:
        self._page = page
        self._out_dir = out_dir
        self._artifacts_dir = artifacts_dir
        self._wait_login_seconds = max(0, wait_login_seconds)
        self._step_timeout_ms = step_timeout_ms
        self._ready_timeout_ms = ready_timeout_ms
        self._poll_seconds = poll_seconds
        self._dry_run = dry_run
        self._pause = pause
        self._start_url = start_url
        self._log = log

    # -- run -----------------------------------------------------------------
    def run(self) -> Path:
        """Drive the wizard end to end and return the saved PDF path."""
        self._open_wizard()
        self._select(ALL_TIME_OPTIONS, step="date-range", description="select 'All time'")
        self._click(CONTINUE_BUTTONS, step="date-range", description="press Continue")
        self._select(
            SELECT_ALL_RECORDS_OPTIONS,
            step="record-type",
            description="select 'Select all VA records'",
        )
        self._click(CONTINUE_BUTTONS, step="record-type", description="press Continue")
        self._select(PDF_FILE_TYPE_OPTIONS, step="file-type", description="select PDF")
        if self._dry_run:
            self._log(
                "[dry-run] stopping before 'Download report' — the browser stays open so "
                "you can inspect the wizard or click it yourself."
            )
            return Path()
        return self._download()

    # -- steps ---------------------------------------------------------------
    def _open_wizard(self) -> None:
        """Wait until the download wizard is genuinely on screen and signed in.

        Two traps this avoids, both of which were observed live against the real
        site: VA.gov is a client-side app that redirects an unauthenticated visit to
        ``/?next=…&oauth=true`` a moment *after* the requested URL loads, so a URL
        check taken too early reports "signed in" while the browser is about to
        bounce to the login modal; and the wizard renders asynchronously, so an
        element check taken too early reports "markup changed". Presence of a wizard
        element — not the URL alone — is what counts as ready here.

        The page is never re-navigated while the human is signing in: that would
        discard an in-progress ID.me step. Re-navigation happens only after VA.gov
        itself reports a signed-in page (which is where the credential flow has
        already finished), because signing in can land on ``/my-va`` instead of
        returning to the wizard.
        """
        self._goto(self._start_url)
        if self._wizard_rendered(timeout_ms=self._ready_timeout_ms):
            self._log(f"Signed in — opened {self._start_url}")
            return
        if is_wizard_url(self._page.url):
            # On the right route but nothing rendered within the ready budget: this is
            # markup, not a missing sign-in, so let the steps report what is absent.
            # (Steps poll for their own element, so a still-hydrating page is fine.)
            self._log(
                "The wizard route is open but its content has not rendered yet — "
                "continuing; a missing step element will name itself."
            )
            return

        deadline = time.monotonic() + self._wait_login_seconds
        self._log(
            "Sign in to VA.gov in the browser window (email, password, then the SMS "
            "code) and stop once you reach My VA or the wizard. This script does not "
            f"type your credentials — it just waits, up to {self._wait_login_seconds}s."
        )
        while time.monotonic() < deadline:
            if is_wizard_url(self._page.url):
                # Confirm it sticks: an unauthenticated visit touches the wizard URL
                # for a moment before the SPA redirects to the login modal.
                time.sleep(min(self._poll_seconds, 1.0))
                if is_wizard_url(self._page.url):
                    self._log("Signed in — the download wizard is open.")
                    return
                continue
            if is_signed_in_url(self._page.url):
                self._goto(self._start_url)
                if self._wizard_rendered(timeout_ms=self._ready_timeout_ms):
                    self._log("Signed in — opened the download wizard.")
                    return
            time.sleep(min(self._poll_seconds, max(0.0, deadline - time.monotonic())))
        raise LoginTimeout(
            "Timed out waiting for you to finish VA.gov sign-in (ID.me email, password, "
            "SMS code). Re-run with a longer --wait-login if you need more time, or "
            "download the report by hand and upload the PDF to the app."
        )

    def _wizard_rendered(self, *, timeout_ms: int) -> bool:
        """True once any wizard marker is visible (see WIZARD_MARKERS)."""
        return self._wait_for_any(WIZARD_MARKERS, timeout_ms=timeout_ms) is not None

    def _select(self, candidates: tuple[Candidate, ...], *, step: str, description: str) -> None:
        """Check/click the first matching candidate (radios and checkboxes)."""
        locator = self._first_visible(candidates, step=step, description=description)
        self._maybe_pause(step, description)
        self._log(f"[{step}] {description}")
        try:
            locator.check(timeout=self._step_timeout_ms)
        except Exception:  # noqa: BLE001 - not an input; e.g. a label or custom element
            locator.click(timeout=self._step_timeout_ms)

    def _click(self, candidates: tuple[Candidate, ...], *, step: str, description: str) -> None:
        locator = self._first_visible(candidates, step=step, description=description)
        self._maybe_pause(step, description)
        self._log(f"[{step}] {description}")
        locator.click(timeout=self._step_timeout_ms)

    def _download(self) -> Path:
        locator = self._first_visible(
            DOWNLOAD_BUTTONS, step="download", description="press 'Download report'"
        )
        self._maybe_pause("download", "press 'Download report'")
        self._out_dir.mkdir(parents=True, exist_ok=True)
        with self._page.expect_download(timeout=self._step_timeout_ms * 6) as capture:
            self._log("[download] pressing 'Download report'")
            locator.click(timeout=self._step_timeout_ms)
        download = capture.value
        suggested = ""
        try:
            suggested = str(download.suggested_filename or "")
        except Exception:  # noqa: BLE001 - the driver only needs a best-effort name
            suggested = ""
        target = self._out_dir / safe_pdf_name(suggested)
        download.save_as(str(target))
        if not looks_like_pdf(target):
            raise DownloadError(
                f"VA.gov produced a file that is not a PDF ({target.name}). That usually "
                "means the session was signed out or the report errored — the file was "
                f"kept at {target} so you can inspect it, and you can retry the run."
            )
        return target

    # -- internals -----------------------------------------------------------
    def _first_visible(
        self, candidates: tuple[Candidate, ...], *, step: str, description: str
    ) -> Any:
        locator = self._wait_for_any(candidates, timeout_ms=self._step_timeout_ms)
        if locator is None:
            self._dump_artifacts(step)
            raise StepFailure(step, description, candidates, url=self._page.url)
        return locator

    def _wait_for_any(
        self, candidates: tuple[Candidate, ...], *, timeout_ms: int
    ) -> Any | None:
        """First visible candidate, re-checked until ``timeout_ms`` runs out.

        Playwright's locators are evaluated against the DOM *now*, so a single
        ``count()``/``is_visible()`` cannot distinguish "this page has no such
        element" from "this page has not rendered yet" — which is exactly how a
        wait-for-render bug reads as a changed-markup bug.
        """
        deadline = time.monotonic() + max(0, timeout_ms) / 1000
        while True:
            for candidate in candidates:
                try:
                    locator = candidate.locator(self._page).first
                    if locator.count() and locator.is_visible():
                        return locator
                except Exception:  # noqa: BLE001 - a failing candidate just means "keep looking"
                    continue
            if time.monotonic() >= deadline:
                return None
            time.sleep(POLL_INTERVAL_SECONDS)

    def _goto(self, url: str) -> None:
        """Navigate, preferring to wait for the client-side app to settle.

        ``networkidle`` is the useful signal for a React route, but VA.gov keeps
        long-lived connections open often enough that it can time out; that is a
        slow page, not a broken one, so fall back to ``domcontentloaded`` and let
        the render wait in ``_wait_for_any`` cover the rest.
        """
        for wait_until, timeout in (("networkidle", 15_000), ("domcontentloaded", 30_000)):
            try:
                self._page.goto(url, wait_until=wait_until, timeout=timeout)
                return
            except Exception as exc:  # noqa: BLE001 - retry with the looser condition
                self._log(f"[nav] {url} (wait_until={wait_until}): {type(exc).__name__}")

    def _maybe_pause(self, step: str, description: str) -> None:
        if not self._pause:
            return
        if not self._stdin_available():
            self._log(f"[{step}] --pause needs a terminal; continuing without confirmation")
            return
        try:
            input(f"[{step}] about to {description} — press Enter to continue: ")
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive only
            self._log(f"[{step}] continuing without confirmation")

    @staticmethod
    def _stdin_available() -> bool:
        """True when there is a terminal to prompt on (None/closed stdin is not one)."""
        try:
            return bool(sys.stdin and sys.stdin.isatty())
        except (ValueError, OSError):  # detached or closed stream
            return False

    def _dump_artifacts(self, step: str) -> Path | None:
        """Save a screenshot + page HTML for a failed step (maintenance aid)."""
        try:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            shot = self._artifacts_dir / f"{step}-{stamp}.png"
            self._page.screenshot(path=str(shot), full_page=True)
            (self._artifacts_dir / f"{step}-{stamp}.html").write_text(
                self._page.content(), encoding="utf-8"
            )
            # The URL is the first thing worth knowing: an element miss is usually a
            # client-side redirect away from the wizard, not changed markup.
            try:
                url = f"{self._page.url}\ntitle: {self._page.title()}\n"
            except Exception:  # noqa: BLE001
                url = "(unavailable)\n"
            (self._artifacts_dir / f"{step}-{stamp}.url.txt").write_text(url, encoding="utf-8")
            self._log(
                f"[{step}] at {url.splitlines()[0]}\n"
                f"[{step}] wrote {shot.name}, {step}-{stamp}.html and "
                f"{step}-{stamp}.url.txt to {self._artifacts_dir} — these are signed-in "
                "pages, so treat them as sensitive and delete them when done."
            )
            return shot
        except Exception:  # noqa: BLE001 - diagnostics must never mask the real error
            return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def default_out_dir() -> Path:
    """Where the browser would have put the file: the desktop, else the CWD."""
    desktop = Path.home() / "Desktop"
    return desktop if desktop.is_dir() else Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate the VA.gov medical-records download wizard. You sign in (ID.me + "
            "SMS code) in the browser window; the script only clicks the wizard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=default_out_dir(),
                        help="Directory to save the downloaded PDF into.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("va_gov_download_artifacts"),
                        help="Where to write the screenshot/HTML of a failed step.")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR,
                        help="Persistent Chromium profile (reuses your VA.gov session).")
    parser.add_argument("--no-persist", action="store_true",
                        help="Use a throwaway profile (no session cookies kept, full MFA again).")
    parser.add_argument("--cdp", default="",
                        help="Attach to a Chrome you already started with "
                             "--remote-debugging-port (e.g. http://127.0.0.1:9222) instead of "
                             "launching a browser. Nothing is closed at the end.")
    parser.add_argument("--start-url", default=START_URL,
                        help="Wizard entry page, if VA.gov renames the route.")
    parser.add_argument("--wait-login", type=int, default=DEFAULT_WAIT_LOGIN_SECONDS,
                        help="Seconds to wait for you to finish ID.me sign-in.")
    parser.add_argument("--step-timeout-ms", type=int, default=DEFAULT_STEP_TIMEOUT_MS,
                        help="Per-step timeout for finding/clicking an element.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk the wizard but stop before pressing 'Download report'.")
    parser.add_argument("--pause", action="store_true",
                        help="Confirm each step in the terminal before it happens.")
    parser.add_argument("--keep-open", action="store_true",
                        help="Leave the browser open at the end (e.g. to download by hand).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sync_playwright = _load_playwright()
    except PlaywrightMissing as exc:
        print(f"✖ {exc}", file=sys.stderr)
        return 2

    if args.cdp:
        print(
            f"Attaching to the browser at {args.cdp}. It is left running afterwards, and "
            "its existing VA.gov session is used as-is.\n"
        )
    else:
        print(
            "This opens a visible browser. Sign in at VA.gov yourself — including the SMS "
            "code — and leave the rest to the script.\n"
        )
    try:
        with sync_playwright() as playwright:
            browser, page, owned = _open_browser(playwright, args)
            try:
                downloader = VaGovDownloader(
                    page,
                    out_dir=args.out,
                    artifacts_dir=args.artifacts_dir,
                    wait_login_seconds=args.wait_login,
                    step_timeout_ms=args.step_timeout_ms,
                    dry_run=args.dry_run,
                    pause=args.pause,
                    start_url=args.start_url,
                )
                saved = downloader.run()
            finally:
                if owned:
                    if args.keep_open:
                        input("Leaving the browser open — press Enter to close it: ")
                    browser.close()
    except DownloadError as exc:
        print(f"✖ {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n✖ Aborted.", file=sys.stderr)
        return 130

    if args.dry_run:
        print("✔ Dry run finished: the wizard was walked but nothing was downloaded.")
        return 0
    print(f"✔ Saved {saved} ({saved.stat().st_size:,} bytes)")
    print(f"  Upload it in the app (record source 'Upload files') or pass it to the pipeline.")
    return 0


def _open_browser(playwright: Any, args: argparse.Namespace) -> tuple[Any, Any, bool]:
    """Open the browser for the run, returning ``(browser, page, owned)``.

    Three modes: attach to a Chrome the human already signed in to (``--cdp``),
    launch a throwaway Chromium (``--no-persist``), or launch Chromium with a
    persistent profile so the VA.gov session can be reused (the default).

    ``owned`` is False when the browser belongs to the human: an attached Chrome is
    never closed out from under them, and never navigated while they are signing in
    (see ``VaGovDownloader._open_wizard``).
    """
    if args.cdp:
        browser = playwright.chromium.connect_over_cdp(args.cdp)
        contexts = browser.contexts
        context = contexts[0] if contexts else browser.new_context()
        page = context.pages[0] if context.pages else context.new_page()
        return browser, page, False
    if args.no_persist:
        browser = playwright.chromium.launch(headless=False)
        return browser, browser.new_page(), True
    context = playwright.chromium.launch_persistent_context(
        str(args.profile_dir), headless=False
    )
    return context, (context.pages[0] if context.pages else context.new_page()), True


if __name__ == "__main__":
    raise SystemExit(main())
