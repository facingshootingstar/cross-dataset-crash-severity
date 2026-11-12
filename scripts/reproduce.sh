#!/usr/bin/env bash
# Runs the whole study from the public raw data. The locked-year step refuses to run twice.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m tools.download_data
$PY -m tools.prepare_data
$PY -m tools.permutation_control --repeats
$PY -m tools.within_domain
$PY -m tools.transfer
$PY -m tools.shap_ranks
$PY -m tools.ablations
$PY -m tools.pooled_sources
$PY -m tools.locked_year
$PY -m tools.design_inference
$PY -m tools.fewshot_draws
$PY -m tools.shift_and_crosswalk
$PY -m tools.native_ceiling
$PY -m tools.learning_curve
$PY -m tools.apply_criteria
$PY -m tools.feature_alignment
$PY -m tools.seed_ladder
