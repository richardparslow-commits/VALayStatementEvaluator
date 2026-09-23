"""Watch for outputs/cfile-ptsd/statement.md; on arrival run mechanical review checks.

Pure-stdlib, no LLM calls, no network. Detached (own session) so it survives
the tool session that launched it. Writes REVIEW_MECHANICAL.md next to the
statement, then exits. Self-terminates after WATCH_TIMEOUT_S in case the run
dies and no statement ever appears.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "cfile-ptsd"
STATEMENT = OUT / "statement.md"
REPORT = OUT / "REVIEW_MECHANICAL.md"
LOG = OUT / "watch.log"
PIDFILE = OUT / "watch.pid"
WATCH_TIMEOUT_S = 8 * 3600

OBServation_KEYWORDS: dict[str, list[str]] = {
    "night terrors (freq ~2x/wk)": [r"scream", r"night(?:mare| terror)", r"twice a week|two (?:times|nights) a week|~?2x"],
    "sweating / unresponsive": [r"sweat", r"(?:un)?responsive|did not respond|no response"],
    "withdrawn next morning": [r"withdrawn|won'?t (?:talk|discuss)|would not talk|quiet"],
    "socks / bending": [r"sock", r"bend"],
    "laundry abandoned": [r"laundry|basket"],
    "stairs + sitting": [r"stair", r"sit"],
    "games since 2019": [r"2019", r"game|season"],
    "waits in car (noise)": [r"car|driveway"],
    "garage retreat": [r"garage"],
    "crowd noise intolerance": [r"talk(?:ing)? at once|noise|loud|crowd"],
}

JARGON_PATTERNS: list[tuple[str, str]] = [
    (r"\bdiagnos(?:is|ed|e[sd]?)\b", "diagnosis language"),
    (r"\bPTSD\b(?![^.]*\bclaim\b)", "bare 'PTSD' outside claim context — check framing"),
    (r"\b(?:depress(?:ion|ed)|anxiety disorder|bipolar|schizophrenia|psychosis)\b", "psychiatric labels"),
    (r"\b(?:arthritis|degenerat\w*|herniat\w*|sciatica|torn|rotator cuff|TBI|traumatic brain injury|sleep apnea)\b", "medical condition names"),
    (r"\b\d+\s*degrees\b|\brange of motion\b|\bROM\b", "range-of-motion value"),
    (r"\b\d+\s*%\b|\bpercent(?:age)? (?:disabled|rating)\b|\brating percent", "rating percentage"),
    (r"\bservice[- ]connected\b", "legal conclusion term"),
    (r"\bTDIU\b|\bunemployab\w*|\bmeets the criteria\b", "legal-conclusion phrasing"),
    (r"\bcaused by\b|\bdue to (?:his|the)\b|\bas a (?:direct )?result of\b", "causation assertion — verify attribution"),
    (r"\bprescri\w*|\bmedication name\b", "medication specifics — verify attribution"),
]

CERT_PATTERN = re.compile(r"certif\w*", re.I)
FORM_PATTERN = re.compile(r"21-10210|21\s*-\s*10210", re.I)


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def notify(title: str, message: str) -> None:
    """Best-effort macOS desktop notification; never raises.

    The report file is the source of truth — this only saves the user from
    re-polling. Silent no-op on non-macOS hosts or when osascript is absent.
    Fixed message strings only (no statement text), so nothing user-supplied
    reaches a shell parser.
    """
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            check=False, capture_output=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def run_checks() -> str:
    text = STATEMENT.read_text(encoding="utf-8")
    lines: list[str] = ["# Mechanical review — statement.md", "",
                        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append(f"Length: {len(text):,} chars, {text.count(chr(10)) + 1} lines")

    lines += ["", "## Required elements (topic L)", ""]
    lines.append(f"- Certification word present: {'YES' if CERT_PATTERN.search(text) else 'NO — MISSING'}")
    lines.append(f"- Form 21-10210 referenced: {'YES' if FORM_PATTERN.search(text) else 'NO — check header'}")
    first_person = len(re.findall(r"\bI\b", text))
    lines.append(f"- First-person 'I' count: {first_person} ({'plausible' if first_person > 10 else 'LOW — voice check'})")

    lines += ["", "## Jargon / competence sweep (advisory — judge in context)", ""]
    hits = 0
    for pat, label in JARGON_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            hits += 1
            ctx = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            lines.append(f"- [{label}] …{ctx}…")
    if not hits:
        lines.append("- None found.")

    lines += ["", "## Bracketed placeholders (record facts for witness to verify)", ""]
    ph = re.findall(r"\[Confirm[^\]]*\]", text, re.I)
    lines.append(f"- Count: {len(ph)}")
    for p in ph:
        lines.append(f"  - {p[:120]}")

    lines += ["", "## Observation coverage matrix", ""]
    low = text.lower()
    for label, pats in OBServation_KEYWORDS.items():
        found = any(re.search(p, low) for p in pats)
        lines.append(f"- {'OK ' if found else 'MISSING'} — {label}")

    lines += ["", "## Next step", ""]
    lines.append("Mechanical pass only. Run the judgment review (REVIEW_RUBRIC.md) ")
    lines.append("against grounding.md for record-corroboration and topic honesty.")
    return "\n".join(lines) + "\n"


def main() -> int:
    PIDFILE.write_text(str(os.getpid()))
    log(f"watcher started (pid {os.getpid()}), waiting up to {WATCH_TIMEOUT_S}s for {STATEMENT.name}")
    deadline = time.time() + WATCH_TIMEOUT_S
    while time.time() < deadline:
        if STATEMENT.exists():
            time.sleep(5)  # let grounding.md land too
            report = run_checks()
            REPORT.write_text(report, encoding="utf-8")
            log(f"statement detected; mechanical report written to {REPORT.name}")
            notify(
                "VA statement ready",
                f"statement.md landed — mechanical review written to {REPORT.name}",
            )
            return 0
        time.sleep(20)
    log("timed out without statement.md — run may have died; check rerun2.log")
    notify("VA statement watcher timed out", "No statement.md after 8 hours — check rerun2.log")
    return 1


if __name__ == "__main__":
    if OUT.exists() and "--launch" in sys.argv:
        p = subprocess.Popen(
            [str(Path(sys.executable)), str(Path(__file__).resolve())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, cwd=str(ROOT),
        )
        PIDFILE.write_text(str(p.pid))
        print("detached watcher pid:", p.pid)
        sys.exit(0)
    sys.exit(main())
