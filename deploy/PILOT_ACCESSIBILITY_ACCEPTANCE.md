# R13: accessibility and participant comprehension acceptance

**Status: unapproved preparation. Real-information admission remains NO-GO.**
Automated tests and a local invented-data component check are limited evidence.
They do not establish actual screen-reader behavior, complete application access,
participant understanding or conformance to an accessibility standard.

## Bind the intended scope and actual release

Name the operator, accessibility tester, independent accessibility reviewer and
comprehension facilitator. Use private participant aliases; keep identities,
access needs and observations outside Git, images, public PRs and application logs.
Agree the private evidence handling procedure and participant authorization before
conducting any participant study. Use wholly invented case materials for these
checks, including screenshots and recordings. This document authorizes no real
record upload, participant contact, account change or provider charge.

For the clean reviewed release, prepare an offline worksheet:

```sh
mkdir -p accessibility-evidence
python scripts/accessibility_review.py --out accessibility-evidence/draft.json
```

The tool checks all tracked release bytes against HEAD and records the full
revision, source tree and file hashes. It creates a new file exclusively and
refuses changed, missing or linked source. It reads no participant observations,
contacts no host/provider and marks every check `not_run`. Its hash identifies
the draft; it is not a signature or evidence of a completed test.

Record actual private-host reference, effective configuration hash, immutable
web/parser/launcher/proxy image IDs and the synthetic task set. Build a matrix
from the intended participants' actual browser, OS, device, input method,
assistive tool and versions/settings. Include any magnification, text scaling,
high contrast or touch settings they use. Select combinations based on those
needs; do not assume that the developer's desktop browser represents everyone.
Keep untested intended combinations excluded from admission. Rebind and retest
relevant checks after release, theme, configuration or supported-scope changes.

## Observe each actual task

Use the worksheet's A01–A12 identifiers. For each applicable matrix row, record
tester alias, time, exact steps, observed outcome, evidence reference and any
failure. `not_run` and `incomplete` are not passes. Maintain original failures
and retest history rather than replacing them with a final pass.

1. **Keyboard and focus (A02–A03).** Complete sign-in, consent, file selection,
   accepted/refused upload, field entry, run, cancel, tabs, result navigation,
   original passage selection, supporting-source selection, editing, review,
   approved copy, clear and logout without a pointer. Observe logical order,
   visible focus, no trap, reachable controls and predictable focus after reruns,
   errors, edits and cancellation. Confirm that opening help or changing the
   passage being read cannot confirm a factual review or change supporting links.
2. **Assistive tools (A04–A05).** Use each intended screen reader on the actual
   host with its actual browser/settings. Observe spoken headings, field labels,
   selected/disabled states, text and warnings; read an entire original passage
   with the full-passage reader without depending on the source table. Trigger
   refusal, validation error, run progress, completion, cancellation and changed
   review requirements. Record whether changes are announced at the right time
   without lost focus, duplicate disruptive announcements or missing messages.
   Inspecting an ARIA role alone does not prove that it is spoken correctly.
3. **Visual and narrow layouts (A06–A07).** Measure actual text/background,
   control and focus contrast in the supported themes. Check meaning without
   color, 200% text resize, 400% browser zoom and reflow at a 320 CSS-pixel width.
   Check the intended mobile/touch settings and long filenames/passages. No
   whole-page horizontal scrolling should be needed for ordinary text/controls;
   where a data table requires two-dimensional navigation, confirm the complete
   original text is also reachable through the full-passage reader. Observe
   occlusion, clipping, touch targets and accessibility names. Do not infer zoom
   or contrast acceptance from a single screenshot or viewport-width check.
4. **Errors and incomplete work (A08).** Use invented malformed/oversized uploads,
   missing/unreadable pages and synthetic incomplete/blocked results. Confirm
   participants can find what failed and what is needed next, while access,
   quota and factual-review restrictions still apply. Follow the R12 approved
   failure procedure; obtain separate authorization for any paid provider test.

The software supplies escaped alert/status messages, native read-only tables
without CSV download toolbars and a literal full-passage reader. The result
explanation is available in About, Draft and Evaluate.
These changes address observed component barriers; Streamlit reruns and external
identity/file-picker behavior still require the actual task checks above.

## Supervised teach-back with invented cases (A09–A11)

Invite intended participants only through a separately authorized private process.
Ask them to complete tasks and explain the meaning in their own words. Avoid
coaching the expected answer before their first attempt. Record misunderstood
meaning, task assistance and corrective wording, then repeat affected tasks.
Reading a warning or checking a box is not evidence that it was understood.

| Invented task / prompt | Required understanding or action |
| --- | --- |
| A witness describes an event absent from the supplied pages; the label is NOT FOUND. What does this tell you? | The model found no matching text in the supplied readable material; it does not prove the event was false. Check missing/unreadable pages and the witness account. |
| A quote contains a denial by the patient while the witness describes an observation; the label is CONTRADICTED. What should you compare? | Compare exact wording, timing, uncertainty and who said/observed each thing. The label does not determine which account is true. |
| A passage is labeled SUPPORTED or PARTIALLY SUPPORTED. Is a matching quote enough? | Read the original passage and check meaning, numbers, dates and attribution; a quote match is not a factual or legal decision. |
| The result has unreadable pages or incomplete checks. Can you treat the whole review as complete? | Identify the missing coverage and resolve it through the approved workflow; partial output remains partial. |
| The draft contains `[Confirm: date]` or changes “around 2020” to an exact date. What is needed? | Resolve with the witness and original source; preserve uncertainty rather than inventing a date. |
| A reviewed sentence is edited, a supporting source is changed or the original context changes. Does prior review carry over? | Review the new exact text and source meaning again; previous review is invalidated. Changing only the passage being read is not approval. |
| You want to save or send the text, then clear the case. What is allowed and what is erased? | File exports are disabled. Copy only reviewed text to an operator-approved destination. Clear releases session data and registered uploads; it cannot erase originals, provider copies or material already copied elsewhere. |

A participant must complete the permitted workflow and demonstrate these safety
meanings using the access methods they need. Misunderstood meaning, inaccessible
required controls, missing warnings and unresolved task barriers block admission
for the affected scope. An independent reviewer must verify any claimed
inapplicability with a narrow rationale; simply omitting an intended participant
or device from the worksheet is not acceptance.

## Independent decision (A12)

The reviewer checks the completed private worksheet, actual observations,
participant scope, failures/retests and release identity. The operator and
reviewer authenticate dated, expiring acceptance with the exact release and
accepted participant/matrix scope. Preserve their decision and evidence in the
approved private store. The worksheet tool does not validate signatures or
convert a draft to approval; a manual independent decision remains required.

Keep `NO-GO` until every applicable check has accepted evidence and no unresolved
admission-blocking failure. Include the privately accepted reference in the release
acceptance bundle referenced by the existing `deployment_validation` approval
field only after that review; the example
approval stays expired and incomplete. All other R07–R12 acceptance gates still
apply. File exports remain excluded; if later enabled, repeat access testing and
separately review rendered PDF/DOCX readability and layout.

The procedure uses [W3C keyboard guidance](https://www.w3.org/WAI/WCAG22/Understanding/keyboard.html),
[status-message guidance](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html),
[reflow guidance](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html) and
[contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)
as test references. These references and preparation do not certify conformance.
