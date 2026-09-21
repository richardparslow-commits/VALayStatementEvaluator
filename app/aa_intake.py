"""The 16-question Aid & Attendance intake: question data and prompt block.

Why this exists
---------------
A&A (and SMC) awards turn on the *observational* matrix in 38 CFR 3.352
(ADL assistance) and 3.354 (supervision need, extra-schedular markers). Lay
statements usually under-report exactly those observations, not because the
caregiver did not live them but because nobody asked. This module is the
asking: sixteen questions — the classic A&A dependency set plus the
PTSD-supervision items — whose answers become a structured prompt block that
rides alongside the witness's free-form observations.

Design constraints
------------------
* **Observations, not conclusions.** Every question asks what the witness
  has *seen*, and the block's framing instructs the model to treat the
  answers as first-hand observations. The regulatory mapping (3.352/3.354)
  lives HERE, in the analysis framing — never in the witness's mouth. A
  statement that parrots "substitute for institutional care" reads as
  coached; a statement that reports the Friday flashback incident lets the
  rater reach the conclusion.
* **String-typed transport.** Answers flatten into the ``witness`` dict as
  ``aa_<slug>`` keys, exactly like the credential fields. ``dict[str, str]``
  is what the job queue serializes, so intake answers survive queued runs
  for free.
* **Empty is a valid state.** A form answered with the defaults serializes
  to nothing and every prompt is byte-identical to the pre-intake pipeline.
* **Every field is sanitized** on its way into a prompt — the block sits
  inside ``<<<``/``>>>`` delimiters, so a pasted delimiter sequence or
  injection directive must not break out.
"""
from __future__ import annotations

from dataclasses import dataclass

from .prompt_sanitize import sanitize_for_prompt

# ---------------------------------------------------------------------------
# Question data
# ---------------------------------------------------------------------------

FREQUENCY_CHOICES = ("Daily", "A few times a week", "About weekly", "A few times a month", "Rarely", "Never")
YESNO_CHOICES = ("No", "Yes", "Sometimes")


@dataclass(frozen=True)
class IntakeQuestion:
    """One intake question, renderable and parseable by key."""

    slug: str                 # stable key; becomes witness["aa_<slug>"]
    text: str                 # what the witness is asked
    kind: str                 # "frequency" | "yesno" | "detail"
    reg: str                  # the 38 CFR criterion the answer evidences (analysis framing)
    help: str = ""            # one-line example / clarification in the UI
    choices: tuple[str, ...] = ()  # for kind == "frequency" | "yesno"


def _q(
    slug: str, text: str, kind: str, reg: str, help: str = "",  # noqa: A002 - DOM spelling
    choices: tuple[str, ...] = (),
) -> IntakeQuestion:
    return IntakeQuestion(slug=slug, text=text, kind=kind, reg=reg, help=help, choices=choices)


#: The 16 mandatory A&A questions, in wizard order.
INTAKE_QUESTIONS: tuple[IntakeQuestion, ...] = (
    _q(
        "care_provider_frequency", "Who provides the care, and exactly how often is it required?",
        "detail", "3.352(a) — regular assistance",
        "e.g. 'Me (spouse), every morning and evening, about an hour total'",
    ),
    _q(
        "cleanliness", "Is the veteran unable to keep themselves ordinarily clean and presentable without someone telling them to?",
        "frequency", "3.352(a) — keep ordinarily clean and presentable",
        "What actually happens on days no one prompts them?", FREQUENCY_CHOICES,
    ),
    _q(
        "dressing", "Does the veteran need physical assistance to dress or undress?",
        "frequency", "3.352(a) — dressing/undressing",
        "Buttons, zippers, braces, balance while stepping into trousers", FREQUENCY_CHOICES,
    ),
    _q(
        "eating", "Does the veteran require assistance with eating or feeding?",
        "frequency", "3.352(a) — eating/feeding",
        help="Cutting food, prompting to continue, leaving meals half-finished", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "wants_of_nature", "How does the veteran's mental state impact their ability to attend to the 'wants of nature' (toileting)?",
        "detail", "3.352(a) — attends to the wants of nature",
        "e.g. 'During flashback days he will not leave the bedroom; twice he has soiled himself rather than cross the hall'",
    ),
    _q(
        "prosthetics", "Does the veteran frequently need help adjusting special prosthetic or orthopedic appliances?",
        "frequency", "3.352(a) — adjusting prosthetic/orthopedic appliances",
        help="Braces, hearing aids, CPAP, wheelchair transfers", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "hazard_supervision", "Does the veteran's mental incapacity require regular supervision to protect them from daily hazards?",
        "frequency", "3.354(a) — protection from hazards/danger",
        help="Stove left on, wandering, door locks, answering the door to strangers", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "bedridden", "Is the veteran considered permanently bedridden?",
        "yesno", "3.352(b) — bedridden",
        help="Or essentially confined to bed/the immediate home", choices=YESNO_CHOICES,
    ),
    _q(
        "healthcare_supervision", "Does the veteran require daily, specialized healthcare supervision?",
        "frequency", "3.354(b) — daily supervision by medical professionals",
        help="Wound care, oxygen, insulin, medication side-effect monitoring", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "no_assistance_consequence", "What would happen if this assistance were not provided?",
        "detail", "3.354(e) — would require hospitalization/nursing home",
        "Say it concretely: what did happen on the days help was late or unavailable?",
    ),
    _q(
        "adl_reminders", "How often do you remind the veteran to do things related to ADLs?",
        "frequency", "3.352(a) — prompting through ADLs",
        help="Showering, changing clothes, brushing teeth, taking meals", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "flashback_care", "What does the caregiver actually do during severe flashbacks or panic attacks?",
        "detail", "3.354(a) — supervision to protect from hazards (PTSD)",
        "What you say and do, how long it lasts, what he is unable to do during it",
    ),
    _q(
        "medication_supervision", "Does the veteran require supervision to safely manage and take psychiatric medications or other medications?",
        "frequency", "3.354(b) — regular monitoring of psychiatric medication",
        help="Double-dosing, refusing doses, side effects you manage", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "eating_safety", "Does the veteran require supervision to ensure they eat safely and consistently?",
        "frequency", "3.352(a) — nourishment",
        help="Forgetting meals, weight loss, food left uneaten", choices=FREQUENCY_CHOICES,
    ),
    _q(
        "care_substitute", "Does the caregiver's care replace what would otherwise be institutional (inpatient) psychiatric care?",
        "detail", "3.354(a) — protection from self-harm/harm to others",
        "Describe the incidents: what has happened, or nearly happened, when symptoms peaked",
    ),
    _q(
        "left_alone", "What would immediately happen if the veteran were left completely alone for 24 to 48 hours?",
        "detail", "3.354(e) — regular (daily) assistance required",
        "Base it on real episodes, not the worst thing you can imagine",
    ),
)

_QUESTION_BY_SLUG = {q.slug: q for q in INTAKE_QUESTIONS}

# The witness-dict prefix for every intake answer ("aa_" + slug).
KEY_PREFIX = "aa_"


def answers_from_witness(witness: dict[str, str] | None) -> dict[str, str]:
    """Extract intake answers (``aa_*`` keys) from a witness dict.

    ``None`` is accepted and yields ``{}`` — callers thread an optional
    witness dict and should not each have to remember ``witness or {}``.
    Returns only the keys present and non-empty; a witness dict with no
    intake answers yields ``{}`` and the whole feature is a no-op.
    """
    if not witness:
        return {}
    return {
        key[len(KEY_PREFIX):]: value.strip()
        for key, value in witness.items()
        if key.startswith(KEY_PREFIX) and str(value or "").strip()
    }


# Frequency/yes-no answers that mean the behaviour does NOT occur. These are
# still rendered — a "no" is also evidence — but they are marked so the model
# does not read them as support for the A&A criterion.
_NEGATIVE_ANSWERS = frozenset({"no", "never"})


def _norm(answer: str) -> str:
    return " ".join(str(answer).split()).casefold()


def care_observation_block(witness: dict[str, str]) -> str:
    """Render the intake answers as the CARE OBSERVATIONS prompt section.

    Empty string when no answers are present, so prompts for runs without
    the intake form stay byte-identical to before. Each rendered answer
    names the 38 CFR criterion it evidences — the mapping is *analysis
    framing for the model*, deliberately not phrasing the witness should
    repeat (the block says so explicitly). Every free-text answer is
    sanitized; the fixed choice sets are already controlled strings but are
    passed through the same gate for uniformity.
    """
    answers = answers_from_witness(witness)
    if not answers:
        return ""
    lines = ["AID & ATTENDANCE CARE OBSERVATIONS (structured intake):",
             "The witness answered a fixed intake questionnaire. Treat each answer as",
             "a first-hand observation to weave into the statement concretely (with",
             "frequency, examples, and consequences). The regulation citations below",
             "are analysis notes for YOU, not language for the witness to repeat —",
             "the statement must never quote regulatory phrases or conclusions; it",
             "reports what was seen and let the rater apply the criteria."]
    for question in INTAKE_QUESTIONS:
        answer = answers.get(question.slug)
        if not answer:
            continue
        text = sanitize_for_prompt(answer, max_chars=1_000)
        marker = "not reported" if _norm(answer) in _NEGATIVE_ANSWERS else "reported"
        lines.append(
            f"- [{marker}] {question.text} → {text}  (evidences 38 CFR {question.reg})"
        )
    return "\n".join(lines)


#: Render the full block into the evaluate RECOMMENDATIONS_USER slot —
#: the model is told what the witness *did* report so it can spot the
#: gap between the statement and what the witness actually knows.
def care_gaps_text(witness: dict[str, str]) -> str:
    """The intake block as a coverage-gap input for the evaluation pipeline.

    Empty string when the caller supplied no intake answers (every existing
    caller), so the recommendations prompt is byte-identical to the
    pre-intake pipeline. When answers exist, the block names what the
    witness reports about each A&A criterion — the recommendations model
    uses it to recommend the statement cover observations the witness has
    already attested to, which is the highest-impact edit available.
    """
    return care_observation_block(witness)


def completed_count(witness: dict[str, str]) -> int:
    """How many of the 16 questions have a non-empty answer (UI progress).

    Every non-empty answer counts, including deliberate "No"/"Never"
    choices — an explicit negative is an answered question and renders in
    the block as ``[not reported]`` evidence.
    """
    return len(answers_from_witness(witness))
