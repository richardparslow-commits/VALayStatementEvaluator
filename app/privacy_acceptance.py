"""Structured actual-host/account attestations; never automatic certification."""
from __future__ import annotations
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Mapping

CHECKS = tuple(f"P{n:02}" for n in range(1, 12))
BINDING_FIELDS = ("reviewed_revision", "deployment_url", "issuer", "participant_notice",
                  "provider_base_url", "models", "model_profiles", "local_log_retention_days",
                  "provider_terms", "retention_policy", "privacy_review", "ingestion_security",
                  "quota_policy", "spending_controls")


def configuration_sha256(approval: Mapping[str, Any]) -> str:
    data = {key: approval[key] for key in BINDING_FIELDS}
    data["text_export_policy"] = approval.get("text_export_policy")
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _reference(value: Any) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError


def validate(approval: Mapping[str, Any]) -> None:
    """Validate structure/freshness/binding, not observation truth or legality.

    Private external evidence is referenced and hashed, never loaded into the
    app. Actual account/image inspection and reviewer authorization still need
    independent release acceptance.
    """
    value = approval.get("privacy_acceptance")
    keys = {"schema_version", "status", "reviewed_at", "operator", "independent_reviewer",
            "configuration_sha256", "images", "checks"}
    if not isinstance(value, dict) or set(value) != keys or type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["status"] != "ACCEPTED":
        raise ValueError
    for key in ("operator", "independent_reviewer"):
        _reference(value[key])
    if value["operator"].casefold() == value["independent_reviewer"].casefold() or value["configuration_sha256"] != configuration_sha256(approval):
        raise ValueError
    from .pilot import _approval_time
    now = datetime.now(timezone.utc)
    reviewed = _approval_time(value["reviewed_at"])
    if not 0 <= (now - reviewed).total_seconds() <= 30 * 86400:
        raise ValueError
    images = value["images"]
    if not isinstance(images, dict) or set(images) != {"application", "parser", "parser_launcher", "proxy"} or any(not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image) for image in images.values()):
        raise ValueError
    configured_parser = os.getenv("VA_LSE_PARSER_IMAGE", "")
    if configured_parser and images["parser"] != configured_parser:
        raise ValueError
    checks = value["checks"]
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError
    for check in checks.values():
        if not isinstance(check, dict) or set(check) != {"status", "evidence_ref", "evidence_sha256"} or check["status"] != "OBSERVED_PASS":
            raise ValueError
        _reference(check["evidence_ref"])
        if not isinstance(check["evidence_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", check["evidence_sha256"]):
            raise ValueError
