"""Build identity: which commit is this process running?

The app is deployed from git (Streamlit Cloud) and from Docker images, and the
one question a stale deployment hides is "is this the code on main?" — asked
twice this month when a freshly merged feature (the Agent API pivot, the
witness-credentials step) appeared missing because the running build predated
it. This module answers it: a short commit SHA when it can be resolved, empty
otherwise, plus whether the answer came from a real checkout.

Resolution order:

1. ``VA_LSE_BUILD_SHA`` — a deployment that knows its own SHA (a Docker build
   passing ``--build-arg``, a platform injecting it) wins outright;
2. ``git rev-parse`` in the project root — true for git-based deploys and any
   local clone;
3. empty — a source upload or an image built without the build arg. The About
   tab says so explicitly rather than showing nothing, because "unknown" is
   exactly the state that masquerades as "the feature is missing".
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BUILD_SHA_ENV = "VA_LSE_BUILD_SHA"


def _sha_from_env() -> str:
    """An explicitly injected build SHA, if one is set and SHA-shaped."""
    value = os.environ.get(BUILD_SHA_ENV, "").strip()
    # A SHA is hex; anything else in the variable is a mistake worth ignoring
    # rather than displaying (e.g. "latest" pasted into a build arg).
    if value and len(value) >= 7 and all(c in "0123456789abcdefABCDEF" for c in value):
        return value
    return ""


def _sha_from_git() -> str:
    """The checked-out commit, when the app runs from a git clone."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        # No git binary, no .git directory (a Docker image excludes it), or a
        # sandbox that forbids subprocesses — all mean "cannot know".
        return ""
    sha = result.stdout.strip()
    return sha if sha else ""


def build_sha() -> str:
    """The running build's short commit SHA, or ``""`` when unknowable."""
    return _sha_from_env() or _sha_from_git()


def build_source() -> str:
    """Where the SHA came from: ``environment``, ``git``, or ``unknown``."""
    if _sha_from_env():
        return "environment"
    if _sha_from_git():
        return "git"
    return "unknown"
