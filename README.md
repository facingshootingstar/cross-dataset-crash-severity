# Transferring road crash severity models across countries

Code for the paper by Nguyen Van Trong Hieu and Pham Thanh Tri (FPT University), *"Transferring Road Crash Severity Models Across Countries: Evidence from Great Britain, the United States and France"*.

The study harmonises three police-reported injury sources onto 12 shared features and a killed-or-seriously-injured (KSI) target, then measures what happens when a model trained in one country is used in another, and how many local records are needed to do better.

| Source | Country | Unit | Training years |
|---|---|---|---|
| STATS19 (DfT) | Great Britain | collision | 2016-2021 |
| CRSS (NHTSA) | United States | crash, with survey weight | 2016-2021 |
| BAAC (ONISR) | France | collision | 2019-2021 |

All sources validate on 2022 and test on 2023. The year 2024 is scored once, with frozen models.

## Layout

```
crashtransfer/  paths, schema, harmonise (three codebooks), data (splits, design matrix), models, metrics,
                transfer (importance weights, few-shot samples), survey (Rao-Wu replicates), adaptation (CORAL, DANN)
tools/          one script per analysis, listed below
scripts/        reproduce.sh runs everything in order
docs/protocol/  the study protocol and its amendment for France
results/        every result file of the paper, as written by `scripts/reproduce.sh`; crosswalk.csv and crosswalk_baac.csv document every field mapping
```

## Installation

Python 3.12; a GPU is used by `tools/feature_alignment.py` when available.

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

## Data

`tools/download_data.py` fetches the public files into `data/raw/` and records their SHA-256 (about 5 GB). The data are not redistributed here. Sources: DfT Road Safety Data, NHTSA CRSS, and the French annual injury-collision files on data.gouv.fr.

## Reproducing the study

```bash
bash scripts/reproduce.sh
```

Run single steps from the repository root, for example `python -m tools.within_domain`.

| Script | Output |
|---|---|
| `prepare_data.py` | harmonised tables, yearly counts, label audits |
| `permutation_control.py` | label-permutation control; `--repeats` adds five permutations with and without real validation labels in model selection |
| `within_domain.py` | T0 for five learners, recalibration, choice of the primary model |
| `transfer.py` | transfer ladder T0-T4 and T2b |
| `shap_ranks.py` | mean absolute SHAP ranks, Spearman's rho, Kendall's W |
| `ablations.py` | dropped feature groups, STATS19 injury-based reporting |
| `pooled_sources.py` | leave-one-country-out, refit on the French training years |
| `locked_year.py` | 2024; each output can be written only once |
| `design_inference.py` | CRSS design-based intervals, paired differences, driver-group audit |
| `fewshot_draws.py` | few-shot results over five draws |
| `shift_and_crosswalk.py` | proxy A-distance by feature, alternative crosswalks |
| `native_ceiling.py` | ceiling with native fields of each country |
| `learning_curve.py` | learning curve in N and correction of the KSI prior |
| `apply_criteria.py` | pre-specified criteria of the amendment |
| `feature_alignment.py` | classical CORAL, Deep CORAL, DANN |
| `seed_ladder.py` | T0-T3 and the prior correction over five model seeds; expected against observed KSI counts, KSI share in the top 5/10/20% of risk |

Set `CRASH_SMOKE=1500` to run on 1,500 rows per source and year as a quick check. `CRASH_RAW`, `CRASH_INTERIM` and `CRASH_RESULTS` move the data and output folders.

## Notation

| Paper | Results files |
|---|---|
| T0 | target model, tier `T0` |
| T1 | zero-shot, tier `T1` |
| T2 | isotonic recalibration on target 2022, tier `T2` |
| T2b | EM prior correction, tier `T2b` |
| T3 | importance weighting, tier `T3` |
| T4 | 1, 5 or 10% of target training data, `target-only retrain` or `source + few-shot mix` |
| prior correction | arm `prior_official`, with `prior_minus20` and `prior_plus20` for a misspecified share |
| France | `BAAC` |

File names such as `rq_verdicts_baac.json` keep the question labels of the protocol.

## Protocol

`docs/protocol/PROTOCOL.md` has SHA-256 `5b018639c40ccd6a41320f48ee546a748790f21dd1370f28018958bb871e821f` and `docs/protocol/PROTOCOL_BAAC.md` has SHA-256 `456a9691d20ca89911bed693b4c72798b6037138a48a78530aae4bbcae9bcc92`. These are internal locks, not entries in an external registry. Folder names in the protocol (`results/`, `src/`) refer to an earlier layout.

## License

MIT.
