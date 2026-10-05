# Passive-file ingestion acceptance

Status: **actual-host acceptance OPEN**. This addresses ingestion audit A05 in
code, including the encoding ambiguity relevant to recognizing binary content.
It does not close the remaining parsing/understanding or release review issues.
Use invented records for these checks. No real records belong in test fixtures,
public scanners, CI logs, issues, or acceptance evidence.

## Enforced file policy

The policy is unconditional in `app/ingestion_policy.py`. Both the ordinary
reader and the secret-free, isolated parser child apply it. The child's empty
environment cannot disable it. Pilot startup continues to require the reviewed
isolated parser image, private launcher, and existing admission/resource limits.
Changing parser code requires rebuilding and reviewing that immutable image.
Complete [dedicated parser engine acceptance](PARSER_ENGINE_ACCEPTANCE.md) too:
the launcher requires private mutual TLS to a different daemon on a separate
host/VM, without the application Docker socket or credentials. CI validates TLS
and daemon/container boundaries; actual host separation remains unverified.

| Input | Required validation and refusal behavior |
| --- | --- |
| TXT / MD | Strict UTF-8, optionally UTF-8 BOM, or BOM-declared UTF-16/32. No guessed code page or replacement decoding. Recognized binary signatures, NUL, unsupported control characters and malformed Unicode are refused. Legacy exports must be converted explicitly to UTF-8. |
| Source blocks | TXT/MD decoded text and each extracted DOCX story retain contiguous half-open Unicode character spans. Blocks prefer paragraph/whitespace boundaries and never split non-whitespace tokens. Reconstruction preserves source characters without inserted separators within a story. Oversized tokens are explicitly refused. Normalized model/retrieval views are separate. Parser document schema 4 and saved-job decoding check span length, continuity, story identity and order; legacy saved blocks have unknown coverage and require a fresh upload. |
| PDF | Header and strict pypdf structural parse. Password protection, actions (including annotation hyperlinks), JavaScript, file specifications/attachments, XFA, active media and collections are refused before text extraction. A bounded traversal checks dictionaries reachable from the trailer without inflating content streams. |
| DOCX | ZIP header, required OPC content types, root office-document relationship and main Word XML. Every non-directory part has an allowed content type and is read with per-part and cumulative actual-byte limits. Only the listed passive XML part types and PNG/JPEG/GIF/BMP/TIFF image signatures are supported. |
| DOCX active content | Macro/embedded-object/unsupported part types, active or unknown relationships, external targets (including hyperlinks/templates), active Word elements and DDE/include/link/embed field instructions are refused. Internal relative relationships must resolve to an existing part inside the package. |
| XML inside DOCX | DTD/entity declarations and processing instructions are refused through parser callbacks; malformed XML produces a fixed refusal. Per-part limits are 200,000 elements and 128 levels. These checks complement the byte, CPU, memory and wall-time sandbox limits. |
| ZIP of records | All member identities checked before any expansion; existing member/count/ratio/declared and actual expansion limits remain. Hidden filenames and `__MACOSX` paths follow the same policy: supported records are read, and every unreadable/unsupported file has an explicit refusal. Nested archives are skipped and not recursively expanded. Pilot admission refuses a batch containing skips, including cached reruns; legacy archive cache entries are recomputed. Only stored/deflated, unencrypted regular files/directories are supported. |
| Labels | Bounded relative names with `/` separators; no empty, `.`/`..`, absolute/drive/UNC paths, backslashes, colon/bracket citation delimiters, control/format/surrogate characters or padded components. ZIP NUL truncation, links/special files, duplicate names, case or Unicode-normalization collisions are refused. Safe nested Unicode names are preserved. |

Labels are display/citation identifiers, never destinations for filesystem
writes. Input staging uses fixed opaque job filenames in a private temporary
directory. ZIP/DOCX parts are read in memory; no `extractall`, shell execution,
macro evaluation, PDF action execution or relationship fetching is introduced.
Unsupported files are refused; this change does not silently rewrite originals.

PDF graph limits are 100,000 visited/pending entries and 128 levels. This is a
reachable-structure policy, **not an exhaustive scan of every byte, unreachable
PDF object or polyglot payload**. Image checks inspect signatures, not full image
decoding. Neither these checks nor isolation prove a document malware-free.

## Malware decision required before real information

The approval manifest now requires a ninth non-empty reference,
`ingestion_security`. Missing/blank evidence closes admission. Changing that
reference invalidates existing participant consent. The example remains empty.

The named security owner must record a revision/image-specific decision either:

1. Accept this narrow passive-format policy and the documented residual malware
   risk for the controlled pilot, with the reviewed OS sandbox and ingress limits;
   or
2. Require an additional private antivirus/CDR service. Keep admission closed until
   that service is implemented, reviewed and tested, including failure/timeout
   refusal, private routing, no retention/logging of records and a process for
   any transformed document's fidelity/provenance validation.

No antivirus/CDR service has been implemented or tested in this change. An
operator approval reference records a review; its presence alone cannot verify
the review's contents. Never claim scanner coverage from these format tests.

## Actual-host checks and evidence

Bind a private evidence packet to the reviewed commit/tree and running parser,
launcher/web/proxy image digests. Record a named security owner, UTC window and
non-secret results. The following checks remain OPEN until observed on that host:

- Through the authenticated uploader, submit valid synthetic UTF-8, UTF-16/32
  BOM text, passive PDF/DOCX and nested records ZIP. Confirm source labels and
  critical dates, numbers, negation and Unicode symbols survive extraction.
- Rename a PNG/ZIP/executable fixture as TXT and spoof its MIME type. Confirm no
  document reaches the evaluation workflow. Refuse damaged/binary text without
  guessing an encoding or replacement characters.
- Submit PDF JavaScript, attachment, hyperlink/action, XFA/media and encrypted
  fixtures; DOCX macro, embedded object, external template/hyperlink, DTD/entity,
  malformed XML, image mismatch and escaping relationship fixtures. Confirm
  refusal before provider work and responsive recovery for the next valid file.
- Submit unsafe/duplicate/NUL/case/Unicode-colliding ZIP names, links, encryption,
  unsupported compression and bounded bomb fixtures. Confirm no member writes,
  cross-owner access, partial pilot analysis or provider attempts.
- Submit a visible note plus hidden correction, including a supported record under
  `__MACOSX`. Confirm both contents, hashes and source addresses survive the actual
  isolated parser. Add an unreadable hidden file or unsupported sidecar; confirm
  a named refusal blocks pilot use on the first read and cached rerun. Remove the
  refused member and confirm the corrected bundle can proceed. Rebuild/review the
  parser image and refresh revision-specific approval before using the new code.
- Observe CPU, memory and wall-time kill/cleanup on hostile synthetic input using
  the existing resource/isolation acceptance procedure. Verify staging deletion
  after success, refusal and timeout; no document content in host logs.
- Submit a TXT/MD and DOCX story with an ISO date, signed decimal dose, negation
  and combining Unicode text crossing the old 4,000-character seam. Confirm exact
  source reconstruction, original byte identity, intact evidence tokens and
  sequential span/citation addresses through the actual parser. Confirm a token
  longer than the configured block limit is explicitly refused. Deploy the same
  schema-4 application/parser version, rebuild/review the immutable parser image,
  and refresh revision-specific approval before pilot admission.
- Remove `ingestion_security` and change it during an existing consent session.
  Confirm respectively closed admission and required renewed consent.

Retain only fixture IDs/hashes, outcomes, timings, numeric resource measurements,
build/image identities and review references. Cross-reference the resource,
privacy and deployment acceptance packets. Existing approved real-data gates are
not replaced by this checklist.

## Documentation cross-check

- [pypdf PdfReader](https://pypdf.readthedocs.io/en/latest/modules/PdfReader.html):
  strict parsing, encryption and object access. Strict parsing is not malware
  scanning; the application implements its own refusal policy.
- [Python ElementTree](https://docs.python.org/3.12/library/xml.etree.elementtree.html):
  custom TreeBuilder callbacks can intercept document types. Functional fixtures
  verify refusal before entity expansion on supported Python 3.12/3.13 runtimes.
- [Microsoft packaging relationships](https://learn.microsoft.com/en-us/dotnet/api/system.io.packaging.packagerelationship.targetmode):
  internal/external relationship semantics. This pilot refuses external targets
  rather than fetching them.
