# J-Space

J-Space is a small research codebase for testing whether Jacobian-lens signals can recover information that a language model's visible chain of thought omits.

The main experiment applies [J-Lens](https://github.com/anthropics/jacobian-lens) to a pinned MonitorBench preference dataset and compares hidden-state token predictions with behavioural changes under critique. Follow-up scripts test deliberately controlled hidden objectives: answer-cue concealment, acrostic encoding, and a punctuation watermark.

## Result in brief

In the completed main study, J-Lens produced more threshold detections than the lexical baseline, but it did not show a reliable population-level advantage. Manual review found no valid case in which J-Lens recovered information genuinely omitted from the visible chain of thought. This is a controlled null result, not evidence that the method can never work.

The follow-up code is staged and gate-driven so that expensive J-Lens scoring is run only after the target hidden behaviour has been established.

## Public source layout

- `monitorbench_experiment.py` — pinned MonitorBench download, verification, generation, scoring, and analysis utilities.
- `followup/` — answer-cue concealment experiment and tests.
- `secondary_objective/` — acrostic hidden-objective experiment and tests.
- `secondary_objective_watermark/` — punctuation-watermark experiment and tests.

This is intentionally a script-only release. Notebooks, local configurations, model outputs, datasets, cached models, figures, frozen result bundles, and internal working notes are not included. Consequently, the repository documents the implementation but does not by itself reproduce the frozen study results.

## Environment

Python 3.10 or newer is recommended. Create an isolated environment, then install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch 'transformers>=5.5' accelerate huggingface-hub numpy 'pandas>=1.3' 'pyyaml>=6'
pip install 'jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e'
```

Some scripts download public model or benchmark assets. Review the relevant provider terms and resource requirements before running them.

## Running the tests

From the repository root:

```bash
(cd followup && python -m unittest -q test_followup.py)
(cd secondary_objective && python -m unittest -q test_acrostic.py)
(cd secondary_objective_watermark && python -m unittest -q test_watermark.py)
```

## Running experiments

The runners expect a local YAML configuration that is deliberately not published. Use the CLI help to inspect each command and provide your own configuration path:

```bash
python followup/run_followup.py --help
python secondary_objective/run_acrostic.py --help
python secondary_objective_watermark/run_watermark.py --help
```

Keep credentials outside configuration files and source control. Public model downloads can use the standard Hugging Face cache; if authentication is required, supply it through the provider's normal local credential mechanism rather than embedding tokens in experiment files.

## Research caution

Jacobians, token rankings, and behavioural overlaps are diagnostic signals, not direct proof of a model's internal reasoning. Claims should be supported by preregistered thresholds, negative controls, blinded review where possible, and manual inspection of candidate cases.
