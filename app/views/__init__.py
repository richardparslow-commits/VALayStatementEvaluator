"""View layer: Streamlit tab rendering extracted from the app entry point.

Modules:
- ``shared``        — cross-tab helpers (correlation ids, uploads, record
  sources, usage watchdog, progress widgets, audit metadata).
- ``sidebar``       — LLM/Fetch settings and the credit-calibration widget.
- ``evaluate_view`` — Evaluate tab (statement input, run flow, results).
- ``draft_view``    — Draft tab (inputs, run flow, results).
- ``about_view``    — About/Guide tab (static content).

Business logic stays in ``app.evaluate`` / ``app.draft`` /
``app.medical_review`` / ``app.documents``; views only render and orchestrate.
"""
