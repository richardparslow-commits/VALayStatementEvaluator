# Required inputs at pipeline boundaries

Evaluation and drafting share `app/request_validation.py`. Invalid inputs raise
a field-specific, user-safe error before a run quota is reserved or a model is
called. The UI checks requests before the endpoint gate and checks the final
input again after adding saved follow-up answers. Queue encoding checks before
blob writes; workers validate decoded requests before constructing a model
client. Saved submission retries validate their retained payload before enqueue.

Evaluation requires a nonblank text statement and extracted medical records.
Drafting requires nonblank text observations, condition, claim type, and extracted
medical records. Required text must contain a printable, visible, non-whitespace character. This check uses
the sanitizer’s invisible-character set without copying/sanitizing whole pages. Numbers, booleans, containers and null
values are rejected rather than converted into strings or empty defaults.

The payload ceiling for statements and observations is 120,000 characters. This
preserves drafting's existing ceiling and gives both paths bounded room for
follow-up answers. It is distinct from the existing 60,000-character recommended
limit and 80,000-character prompt budget: those still carry their confirmation
and truncation warnings. The post-append check applies the hard ceiling so saved
answers cannot require a confirmation checkbox that the original-input UI never
rendered. Passing input validation does not mean every character
will reach every prompt.

Condition, claim type, filenames and ordinary witness values are bounded at 500
characters. Optional `aa_*` intake answers allow 4,000 characters; their existing
prompt rendering still uses its smaller budget. Witness maps allow at most 64
text fields with short text keys. Missing witness details and empty optional
values remain valid; evaluation may omit the witness map entirely.

Record inputs must be a nonempty list of extracted documents with nonblank page
text and valid page/block addresses. Source page count and total UTF-8 extracted
text obey `MAX_RECORD_PAGES` and `MAX_TOTAL_UPLOAD_BYTES`. The controlled pilot's
existing stricter page and complete-coverage admission gate still applies.
Queued requests use a strict page reader so malformed members cannot disappear
through the tolerant saved-result deserializer. Inline and externalized records
have the same request contract. Result decoding keeps its legacy compatibility.

The batch script validates its configuration and selected source records before
its credential probe, checks extracted groups before medical review, and validates
final-phase inputs before merge/summarization calls. Mixed readable/unreadable
parts retain the existing batch skip and coverage behavior when usable text
remains; no usable records stops the run before any provider call. If legacy
facts have no recoverable source filenames, current readable parts may satisfy
the input requirement, but those facts retain their unresolved provenance warning.
Those current parts do not establish bindings for old facts. The script preserves
raw witness-JSON types so an invalid number or null cannot become invented witness
text.

Rejection logs distinguish missing fields, invalid payloads and capacity limits.
Errors contain labels, policy limits and error kinds, without repeating submitted
values. Drafting preserves its `DraftingPayloadError` API; queue boundaries wrap
contract failures in `PayloadError`; evaluation exposes `RequestValidationError`.

`tests/test_request_validation.py` covers direct, UI, queue, worker, saved-retry,
blob-write and batch paths with synthetic inputs and fake models. This contract
does not establish factual truth, parser isolation, live deployment acceptance,
or approval to enable queues or persistent batch storage for real information.
