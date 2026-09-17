"""Offline tests for OpenTelemetry tracing (app/tracing.py).

Tracing is opt-in and no-ops when the OpenTelemetry packages are missing, so
these tests cover both halves: the default (nothing configured, nothing built,
nothing exported) and the enabled path, driven through a real ``TracerProvider``
with an in-memory exporter — no collector and no network.

The point of most of these is the part that is easy to get subtly wrong and hard
to notice: that spans nest (including from a thread-pool worker), that the queue
boundary continues a trace, that annotations never carry record/statement text,
and that shutdown flushes rather than dropping the trace of the run that just
finished.
"""
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app import shutdown  # noqa: E402
from app import tracing  # noqa: E402
from app.documents import document_from_text  # noqa: E402
from app.job_payload import (  # noqa: E402
    KIND_EVALUATE,
    EvaluateJob,
    decode_job,
    encode_job,
)
from app.logging_config import set_request_id  # noqa: E402


def _in_memory_exporter() -> Any:
    """An ``InMemorySpanExporter``, or None when the SDK is not installed."""
    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return InMemorySpanExporter()


@contextmanager
def _tracing_on(*, chunk_spans: bool = False, llm_calls: bool = False):
    """Enable tracing against an in-memory exporter for the duration of a test."""
    exporter = _in_memory_exporter()
    if exporter is None:  # pragma: no cover - CI installs the SDK
        raise unittest.SkipTest("opentelemetry-sdk is not installed")
    with (
        patch.object(config, "TRACING_ENABLED", True),
        patch.object(config, "TRACING_CHUNK_SPANS", chunk_spans),
        patch.object(config, "TRACING_LLM_CALLS", llm_calls),
        patch.dict(os.environ, {"OTEL_SDK_DISABLED": ""}),
        patch.object(tracing, "_build_exporter", return_value=exporter),
    ):
        tracing.reset_tracing_for_tests()
        try:
            assert tracing.setup_tracing(role="test")
            yield exporter
        finally:
            tracing.reset_tracing_for_tests()


def _finished(exporter: Any) -> list[Any]:
    """Flush the batch processor, then return the exported spans."""
    tracing.flush_tracing()
    return list(exporter.get_finished_spans())


def _spans_by_name(exporter: Any) -> dict[str, Any]:
    return {span.name: span for span in _finished(exporter)}


def _attrs(span: Any) -> dict[str, Any]:
    """Span attributes without ``request.id`` (asserted separately).

    The request id comes from a ContextVar that another test may have set and
    left in place, so including it would make these assertions order-dependent.
    """
    return {k: v for k, v in dict(span.attributes).items() if k != "request.id"}


class TestDisabledByDefault(unittest.TestCase):
    """The default install must behave exactly as it did before tracing existed."""

    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)

    def test_not_enabled_without_the_env_flag(self) -> None:
        with patch.object(config, "TRACING_ENABLED", False):
            self.assertFalse(tracing.is_enabled())
            self.assertFalse(tracing.setup_tracing())
        self.assertFalse(tracing.is_active())

    def test_spans_are_no_ops(self) -> None:
        with patch.object(config, "TRACING_ENABLED", False):
            with tracing.phase_span("claims", statement_text="not traced") as span:
                self.assertFalse(span.is_recording())
                span.set_attribute("anything", 1)
                span.add_event("event")
            self.assertEqual(tracing.inject_trace_context(), {})

    def test_sdk_disabled_env_overrides_the_flag(self) -> None:
        with (
            patch.object(config, "TRACING_ENABLED", True),
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}),
        ):
            self.assertFalse(tracing.is_enabled())
            self.assertFalse(tracing.setup_tracing())
            self.assertIn("OTEL_SDK_DISABLED", tracing.unavailable_reason())

    def test_health_reports_why_tracing_is_off(self) -> None:
        with patch.object(config, "TRACING_ENABLED", False):
            health = tracing.tracing_health()
        self.assertFalse(health["enabled"])
        self.assertFalse(health["active"])
        self.assertIn("VA_LSE_TRACING", str(health["reason"]))

    def test_flush_and_shutdown_are_safe_before_setup(self) -> None:
        self.assertFalse(tracing.flush_tracing())
        tracing.shutdown_tracing()  # must not raise
        tracing.shutdown_tracing()


class TestSpanTree(unittest.TestCase):
    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)

    def test_run_and_phase_spans_nest(self) -> None:
        with _tracing_on() as exporter:
            with tracing.run_span("evaluate", files=2, pages=10):
                with tracing.phase_span("records:review", files=2):
                    pass
                with tracing.phase_span("claims"):
                    pass
            spans = _spans_by_name(exporter)

        root = spans["run:evaluate"]
        self.assertEqual(_attrs(root), {"files": 2, "pages": 10})
        self.assertIsNone(root.parent)
        for name in ("records:review", "claims"):
            self.assertEqual(spans[name].context.trace_id, root.context.trace_id)
            self.assertEqual(spans[name].parent.span_id, root.context.span_id)

    def test_request_id_is_attached_for_log_correlation(self) -> None:
        token = set_request_id("req_trace_test")
        self.addCleanup(set_request_id, "")
        with _tracing_on() as exporter:
            with tracing.phase_span("rubric"):
                pass
            spans = _spans_by_name(exporter)
        self.assertEqual(dict(spans["rubric"].attributes)["request.id"], "req_trace_test")
        del token

    def test_span_context_is_usable_across_threads(self) -> None:
        """Pipeline work runs in a ThreadPoolExecutor; spans must still nest."""
        with _tracing_on() as exporter:
            with tracing.run_span("evaluate"):
                parent = tracing.current_span_context()

                def chunk() -> None:
                    with tracing.use_parent(parent):
                        with tracing.phase_span("records:digest.chunk", chunk=1):
                            pass

                thread = threading.Thread(target=chunk)
                thread.start()
                thread.join()
            spans = _spans_by_name(exporter)

        root = spans["run:evaluate"]
        child = spans["records:digest.chunk"]
        self.assertEqual(_attrs(child), {"chunk": 1})
        self.assertEqual(child.context.trace_id, root.context.trace_id)
        self.assertEqual(child.parent.span_id, root.context.span_id)

    def test_many_threads_share_one_trace(self) -> None:
        with _tracing_on() as exporter:
            with tracing.run_span("evaluate"):
                parent = tracing.current_span_context()
                with ThreadPoolExecutor(max_workers=3) as pool:
                    list(
                        pool.map(
                            lambda i: _chunk_span(parent, i),
                            range(4),
                        )
                    )
            spans = [s for s in _finished(exporter) if s.name == "records:digest.chunk"]
        self.assertEqual(len(spans), 4)
        self.assertEqual(len({s.context.trace_id for s in spans}), 1)


def _chunk_span(parent: Any, index: int) -> None:
    with tracing.use_parent(parent):
        with tracing.phase_span("records:digest.chunk", chunk=index):
            pass


class TestPiiScreening(unittest.TestCase):
    """Span attributes leave the deployment, so free text must never reach them."""

    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)

    def test_forbidden_and_suffixed_keys_are_dropped(self) -> None:
        with _tracing_on() as exporter:
            with tracing.phase_span(
                "claims",
                statement_text="vet was diagnosed with PTSD",
                chunk_text="record body",
                prompt="prompt body",
                records="raw records",
                pages=3,
            ):
                pass
            spans = _spans_by_name(exporter)
        self.assertEqual(_attrs(spans["claims"]), {"pages": 3})

    def test_long_strings_are_truncated(self) -> None:
        with _tracing_on() as exporter:
            with tracing.phase_span("claims", model="m" * 500):
                pass
            spans = _spans_by_name(exporter)
        self.assertEqual(len(str(spans["claims"].attributes["model"])), 200)

    def test_non_primitive_values_do_not_reach_the_sdk(self) -> None:
        with _tracing_on() as exporter:
            with tracing.phase_span("claims", payload={"statement": "x"}, flag=True):
                pass
            spans = _spans_by_name(exporter)
        attributes = _attrs(spans["claims"])
        self.assertEqual(attributes["payload"], "dict")
        self.assertTrue(attributes["flag"])

    def test_none_attributes_are_skipped(self) -> None:
        with _tracing_on() as exporter:
            with tracing.phase_span("claims", condition=None, claims=2):
                pass
            spans = _spans_by_name(exporter)
        self.assertEqual(_attrs(spans["claims"]), {"claims": 2})


class TestErrors(unittest.TestCase):
    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)

    def test_failure_marks_the_span_and_still_propagates(self) -> None:
        from opentelemetry.trace import StatusCode

        with _tracing_on() as exporter:
            with self.assertRaises(ValueError):
                with tracing.phase_span("claims"):
                    raise ValueError("model returned junk")
            spans = _spans_by_name(exporter)

        span = spans["claims"]
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(_attrs(span)["error.class"], "ValueError")

    def test_setup_failure_degrades_to_no_op(self) -> None:
        with (
            patch.object(config, "TRACING_ENABLED", True),
            patch.dict(os.environ, {"OTEL_SDK_DISABLED": ""}),
            patch.object(tracing, "_build_exporter", side_effect=RuntimeError("no collector")),
        ):
            tracing.reset_tracing_for_tests()
            self.addCleanup(tracing.reset_tracing_for_tests)
            self.assertFalse(tracing.setup_tracing())
            with tracing.phase_span("claims") as span:
                self.assertFalse(span.is_recording())
            health = tracing.tracing_health()
        self.assertFalse(health["active"])
        self.assertIn("setup failed", str(health["reason"]))


class TestQueuePropagation(unittest.TestCase):
    """The web pod's span and the worker's spans must be one trace, not two."""

    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)

    def test_inject_returns_nothing_outside_a_span(self) -> None:
        with _tracing_on():
            self.assertEqual(tracing.inject_trace_context(), {})

    def test_worker_span_is_a_child_of_the_submit_span(self) -> None:
        with _tracing_on() as exporter:
            with tracing.phase_span("queue:submit", kind=KIND_EVALUATE):
                # The submit span is what the payload inherits.
                with tracing.phase_span("inner") as span:
                    carrier = tracing.inject_trace_context()
                submit_trace_id = span.context.trace_id
                submit_span_id = span.context.span_id
            self.assertIn("traceparent", carrier)
            # Worker side: adopt the carrier, then run the pipeline.
            with tracing.attach_trace_context(carrier):
                with tracing.run_span("evaluate"):
                    pass
            spans = _spans_by_name(exporter)

        run = spans["run:evaluate"]
        self.assertEqual(run.context.trace_id, submit_trace_id)
        self.assertEqual(run.parent.span_id, submit_span_id)

    def test_missing_or_empty_carrier_is_a_no_op(self) -> None:
        with _tracing_on() as exporter:
            with tracing.attach_trace_context(None):
                with tracing.run_span("evaluate"):
                    pass
            with tracing.attach_trace_context({}):
                with tracing.run_span("evaluate"):
                    pass
            spans = [s for s in _finished(exporter) if s.name == "run:evaluate"]
        self.assertEqual(len(spans), 2)
        self.assertIsNone(spans[0].parent)
        self.assertIsNone(spans[1].parent)


class TestJobPayloadCarriesTraceContext(unittest.TestCase):
    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)

    @staticmethod
    def _job() -> EvaluateJob:
        return EvaluateJob(
            statement_text="I served in the Gulf War.",
            records=[document_from_text("records.txt", "page one")],
        )

    def test_envelope_omits_the_key_when_tracing_is_off(self) -> None:
        """A tracing-off deployment must produce byte-identical payloads."""
        with patch.object(config, "TRACING_ENABLED", False):
            first = encode_job(KIND_EVALUATE, self._job())
        self.assertNotIn("trace_context", first)
        decoded = decode_job(KIND_EVALUATE, first)
        self.assertEqual(decoded.trace_context, {})

    def test_envelope_carries_the_submit_span(self) -> None:
        with _tracing_on():
            with tracing.phase_span("queue:submit") as submit:
                raw = encode_job(KIND_EVALUATE, self._job())
            decoded = decode_job(KIND_EVALUATE, raw)

        self.assertIn("traceparent", decoded.trace_context)
        self.assertIn(format(submit.context.trace_id, "032x"), decoded.trace_context["traceparent"])


class TestShutdownAndHealth(unittest.TestCase):
    def setUp(self) -> None:
        tracing.reset_tracing_for_tests()
        self.addCleanup(tracing.reset_tracing_for_tests)
        shutdown.reset_for_tests()
        self.addCleanup(shutdown.reset_for_tests)

    def test_flush_exports_without_tearing_down(self) -> None:
        with _tracing_on() as exporter:
            with tracing.phase_span("claims"):
                pass
            self.assertTrue(tracing.flush_tracing())
            self.assertIn("claims", _spans_by_name(exporter))
            # Still active: flushing is not shutdown.
            self.assertTrue(tracing.is_active())
            self.assertTrue(tracing.setup_tracing())

    def test_graceful_shutdown_flushes_buffered_spans(self) -> None:
        with _tracing_on() as exporter:
            with tracing.run_span("evaluate"):
                with tracing.phase_span("claims"):
                    pass
            # No explicit flush here: the drain path has to do it, or the trace of
            # the run that just finished is lost with the process.
            shutdown.request_shutdown(source="test", grace_seconds=0)
            names = {span.name for span in exporter.get_finished_spans()}
        self.assertEqual(names, {"run:evaluate", "claims"})
        self.assertFalse(tracing.is_active())

    def test_health_payload_includes_tracing(self) -> None:
        from app.health import _health_payload

        with _tracing_on():
            payload = _health_payload()
        self.assertTrue(payload["tracing"]["enabled"])
        self.assertTrue(payload["tracing"]["active"])
        self.assertEqual(payload["tracing"]["service_name"], config.TRACING_SERVICE_NAME)


if __name__ == "__main__":
    unittest.main()
