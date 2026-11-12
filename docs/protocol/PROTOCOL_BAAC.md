# PROTOCOL AMENDMENT A1 — third jurisdiction: France BAAC (ONISR)

Status: LOCKED 2026-11-06, before any model was fit on, or scored on, BAAC rows. sha256 of this file recorded in `scores/RUNLOG.md`; any later change is a new amendment, not an edit.
`PROTOCOL.md` is unchanged and still governs every STATS19 × CRSS number already in the paper. Nothing in this amendment re-runs, overwrites or re-selects anything listed in `SO_LIEU_CON_THIEU.md` (within_domain_metrics, transfer_matrix, locked_2024.json, primary_model.json, the frozen STATS19/CRSS joblib models).

Transparency note: before drafting, the following was inspected on BAAC: file headers and encodings, the official field description (`raw/manuals/baac_description-des-bases-de-donnees-annuelles.pdf`), row counts, missingness of the 12 shared features, and the KSI and fatal share per year (2016 0.442, 2017 0.449, 2018 0.385, 2019–2024 0.341–0.359; fatal 0.052–0.059). No model has been fit on BAAC and no feature–label association has been looked at.

## A1.1 Why BAAC
Person-level police census of injury collisions (like STATS19, so weight = 1), with hour, driver birth year, speed limit (`vma`), built-up area flag, junction type, light, weather and vehicle category — 12/12 shared features mappable. Download: data.gouv.fr dataset 53698f4ca3a729239d2036df, URLs in `raw/baac/_urls.tsv`, hashes in `raw/baac/SHA256SUMS.txt`.

## A1.2 Research questions
| ID | Question | Rejected if |
| --- | --- | --- |
| RQ6 | Does a model trained on two countries transfer to the third (leave-one-country-out, LOCO) better than the best single source? | On 2 of 3 held-out countries, pooled T1 AUROC 95% CI overlaps the best single-source T1 CI |
| RQ7 | Is the transfer calibration gap mainly label (prior) shift? | EM prior correction (Saerens et al. 2002, unlabeled target 2022) does not lower ECE versus T1 on at least 4 of the 6 directed pairs |
| RQ8 | Are feature importances stable across three countries? | Kendall's W of mean-abs-SHAP ranks (12 features, 3 countries) has a bootstrap 95% CI including 0 (small n: say underpowered) |
| RQ9 | How many labelled target collisions N does a new country need? Learning curve N ∈ {100, 250, 500, 1000, 2500, 5000, 10000, 25000}, 5 draws, arms: target-only, source + Platt on N, source fine-tuned (continued boosting) on N, source + N mixed. N* = smallest N whose mean target-test AUROC is within 0.01 of T0 | On 4 of 6 directed pairs no transfer arm reaches T0 − 0.01 at a smaller N than target-only |
| RQ10 | Does correcting only the KSI prior with a published national KSI share (no microdata) repair calibration? Prior-shift adjustment p' ∝ p·π_t/π_s, π_t = target train-year KSI share (stand-in for the official statistic), plus ±20% misspecification and EM estimate | ECE after correction not lower than T1 on 4 of 6 pairs |
| RQ1–RQ5 | Repeated for BAAC within-domain and the four new directed pairs (BAAC→UK, BAAC→US, UK→FR, US→FR) | as in PROTOCOL.md |

RQ9 and RQ10 were first run on STATS19 ↔ CRSS as post-hoc analyses (script `code/run_13_posthoc_learning_curve.py`, labelled post-hoc in the paper) before this amendment is locked; for the four BAAC pairs they are pre-registered here, with the same script and grid unchanged. Intended recipe for a new country: (1) prior correction from the published KSI share, (2) collect N* labelled collisions, (3) fine-tune the foreign model.

## A1.3 Window and split for BAAC
| Split | Years | Role |
| --- | --- | --- |
| Train | 2019–2021 | fit, class weights, encoders |
| Validation | 2022 | early stopping, calibration, EM priors (labels unused for EM) |
| Test | 2023 | all main tables |
| Locked | 2024 | scored once, after code freeze, into a new file `scores/locked_2024_baac.json` |

Reason for starting at 2019: (a) "blessé hospitalisé" is documented by ONISR as not comparable before 2018 and the KSI share drops from 0.44 (2016–2017) to 0.36 (2019+); (b) `vma` (speed limit) does not exist before 2019; (c) the file layout changes in 2019. BAAC 2016–2018 is used only in a label-drift audit (KSI share by year, analogous to the STATS19 IBRS audit), never for fitting. STATS19 and CRSS keep their PROTOCOL.md split (train 2016–2021). Sensitivity: STATS19/CRSS models refit on 2019–2021 only, so that all three sources use the same train years.

## A1.4 Target and scope
Unit = collision (`Num_Acc`). User `grav`: 2 killed (within 30 days) → Fatal; 3 hospitalised > 24 h → Serious; 4 slight → Slight; 1 uninjured ignored. Collision severity = worst user. KSI = Fatal + Serious. Scope: metropolitan France (department code not starting 97/98); overseas departments dropped and counted. From 2021 BAAC includes users who fled; missing `grav` is not imputed. The French Serious definition (hospitalised > 24 h) differs from STATS19 Serious and KABCO A; this is a pre-declared label mismatch, discussed, never "corrected" by relabelling.

## A1.5 Features
Same 12 features, same names, null when unmappable. The crosswalk is in `code/harmonise_baac.py` (header) and `scores/crosswalk_baac.csv`. Speed limit is converted km/h → mph (× 0.621371), not rounded to UK/US bands, and only legal limits {20, 30, 50, 70, 80, 90, 110, 130} km/h are kept. Notable semantic risks: urban_rural from `agg` (built-up area, legal definition) — shifted; junction from `int` — close; dusk/dawn → not dark (as CRSS); vans (VU) → goods (as STATS19/CRSS).

## A1.6 Models and transfer
Primary model and hyperparameter grid: the same as PROTOCOL.md (XGBoost primary, plus majority, logistic, RF, LightGBM), seed 42, early stopping on 2022. Existing frozen STATS19/CRSS models are reused unchanged for UK→FR and US→FR.
- T0 BAAC within-domain; T1 zero-shot; T2 isotonic on target 2022; T3 importance weights with PAD; T4 1/5/10% of target train (BAAC: 2019–2021); for all four new directed pairs.
- New T2b: EM prior correction (Saerens 2002) using target 2022 features only; reported for all 6 directed pairs. The existing 2 pairs get T2b as a post-hoc addition, labelled post-hoc in the paper.
- LOCO: for each held-out country, train XGBoost on the other two sources' train years; each source rescaled to equal total weight (CRSS keeps its WEIGHT inside its share); early stopping on the two sources' 2022; evaluate T1/T2/T2b on the held-out country's 2023.
- SHAP: as PROTOCOL.md (20,000 rows of 2023), three countries, Kendall's W plus the three pairwise Spearman rho.
Output: `scores/transfer_matrix_3way.csv`, `scores/loco.csv`, `scores/within_domain_baac.csv`. Existing score files are not overwritten.

## A1.7 Controls
Negative control on BAAC (permuted train labels, test 2023 AUROC; stop if > 0.60). Ablations as PROTOCOL.md §7 within BAAC. Leakage checklist of PROTOCOL.md §8 applies, plus: no `dep`, `com`, `adr`, `lat`, `long`, `gps`, `id_*`, `secu*`, `etatp`, `locp`, `actp` in X.

## A1.8 Locked 2024
Every result involving BAAC is scored on 2024 once, after the code that produces the 2023 tables is frozen, with the model chosen on 2023. Existing `scores/locked_2024.json` is not touched. `code/run_06_locked.py` is not re-run; a new `code/run_14_locked_baac.py` refuses to run twice (same mechanism).

## A1.9 Writing rules
Paper numbers come only from `scores/`. The paper must state that BAAC was added after the STATS19 × CRSS results were known (amendment, not original design), and must report the BAAC 2016–2018 label break.
