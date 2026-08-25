# 🎖️ VA Lay Statement Evaluator

A Streamlit application that performs **exhaustive medical-record review** to:

1. **Evaluate** an existing lay/witness statement (VA Form 21-10210 style) — extracting every
   factual claim, verifying each claim against the medical records, scoring the statement on an
   8-dimension rubric drawn from VA lay-evidence law, auditing it against the **topic
   checklist** (hazards and dangers, caregiver necessity, personal care, medication and
   financial management, household safety, errands and driving, before/after progression,
   observable behaviors, family impact, medication side effects), and then suggesting how to
   improve it: a prioritized improvement plan plus a proposed rewrite with record-grounded
   corrections; and
2. **Draft** a new, factually grounded statement from a witness's own observations — checking
   the observations against the topic checklist, asking follow-up questions for applicable
   topics not yet covered, and flagging anything that conflicts with or cannot be verified in
   the records.

> ⚠️ **Disclaimer.** This tool is an educational and drafting aid. It is **not** legal, medical,
> or claims advice. No output should be submitted to the VA without the witness personally
> verifying every fact. For accredited help see
> [www.va.gov/ogc/apps/accreditation](https://www.va.gov/ogc/apps/accreditation).

## Legal foundation baked into the tool

- Competent lay evidence: 38 C.F.R. § 3.159(a)(2); *Jandreau v. Nicholson*, 492 F.3d 1372 (Fed. Cir. 2007)
- Benefit of the doubt: 38 U.S.C. § 5107(b); *Gilbert v. Derwinski*
- Combat presumption & duty to consider all lay evidence: 38 U.S.C. § 1154(a); *Buchanan v. Nicholson*, 451 F.3d 1331
- Absence from records ≠ negative evidence: *Barr v. Nicholson*; *Buczynski v. Shinseki*
- Nexus lay competence limits: *Layno v. Brown*; *Kahana v. Shinseki*; *Davidson v. Shinseki*

Key design rule: a claim absent from the records is reported as **NOT FOUND**, never as
contradicted; only an explicit record conflict is **CONTRADICTED**.

## Architecture

```
run_app.py                Streamlit launcher
app/
  main.py                 UI: Evaluate / Draft / About tabs
  config.py               Settings (.env), knowledge-file loader
  llm.py                  OpenAI-compatible client (retry, JSON parsing)
  documents.py            TXT/MD/DOCX/PDF extraction, page-aware chunking
  medical_review.py       Exhaustive chunked record review -> fact digest
  evaluate.py             Claim extraction -> verification -> rubric scoring -> topic coverage
                          audit -> improvement suggestions & proposed rewrite -> report
  draft.py                Grounding + topic coverage -> draft -> self-review pipeline
  knowledge/              legal_framework.md, evaluation_rubric.md, drafting_guide.md,
                          topic_checklist.md
scripts/
  extract_pdfs.py         Build reference_docs/extracted/*.txt from source PDFs
  smoke_test.py           End-to-end pipeline test against the live LLM endpoint
tests/                    Offline unit tests (no API key required)
examples/                 Fictional sample statement + sample medical records
```

Long documents are processed in overlapping, page-labelled chunks so reviews are exhaustive
regardless of record length.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then put your API key in .env (never commit .env)
```

### Environment variables (`.env`)

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | API key for the gateway | (required) |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL | Alibaba MaaS gateway |
| `LLM_MODEL_MAIN` | Heavy model (analysis/scoring/drafting) | `qwen3.7-max` |
| `LLM_MODEL_FAST` | Light model (fact extraction/merging) | `qwen3.7-flash` |

All four can also be overridden live in the app sidebar. Model availability depends on your
gateway workspace; check `GET {base_url}/models`.

## Run

```bash
streamlit run run_app.py
```

### Evaluate a statement

1. Upload or paste the lay statement.
2. Upload the veteran's medical records (PDF/TXT/MD/DOCX, multiple files OK).
3. Click **Run exhaustive evaluation** — watch chunked record review, claim verification,
   rubric scoring, improvement drafting, and report generation progress.
4. Review the verdict table (✅ supported / 🟡 partial / ❌ contradicted / ⚪ not found),
   scores, and the prioritized improvement plan.
5. Review the **proposed rewrite**: a change-by-change table (original → suggested → why),
   the revised statement with `[Confirm: ...]` placeholders, and downloads for both the
   report and the revised statement.

### Draft a statement

1. Upload the veteran's medical records.
2. Enter witness details and bulleted firsthand observations.
3. Click **Draft the statement** — the app grounds every observation in the records, flags
   conflicts, suggests strengthening questions, drafts the statement, and self-reviews it.
4. Resolve every bracketed `[Confirm: ...]` placeholder with the witness before signing.
   Submit on VA Form 21-10210 (one form per witness).

## Tests

```bash
python -m unittest discover -s tests -v        # offline unit tests
python scripts/smoke_test.py all               # live end-to-end (needs valid .env)
```

## Security notes

- `.env`, `.venv/`, and `outputs/` are git-ignored.
- Medical records stay local: they are only sent to the configured LLM endpoint.
