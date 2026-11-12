# PROTOCOL — Cross-Dataset ML for Road Accident Severity (STATS19 × CRSS)

Status: LOCKED before any model fit. Its sha256 is recorded in `results/RUNLOG.md`.
Source plan: `D:\Downloads\plan-cross-dataset-crash-severity.md`. Where this file is more specific, this file wins.

Transparency note: before locking, only row counts, severity value counts and WEIGHT sums of CRSS were inspected (to confirm column names and code meanings). No model has been fit and no STATS19 severity mix has been looked at for the main window.

## 1. Research questions and rejection criteria
| ID | Question | Rejected if |
| --- | --- | --- |
| RQ1 | On STATS19, does the model separate KSI better than logistic regression and majority? | Bootstrap 95% CI of AUROC(KSI) overlaps logistic's |
| RQ2 | On CRSS injury-only, does it hold under weights? | Weighted AUROC collapses to chance while unweighted looks good |
| RQ3 | Calibration shift vs discrimination shift under domain change | T2 does not lower ECE, or T2 raises AUROC by > 0.02 (check leakage) |
| RQ4 | Which features are stable across countries? | Spearman rho of mean abs SHAP over shared features indistinguishable from 0 (small n: say underpowered, not "unrelated") |
| RQ5 | How much target data (1/5/10%) closes the gap? | No level closes >= 50% of the T1 -> target-only gap |

Mechanism hypotheses (not results): H1 lighting, hour, VRU, speed limit transfer partially. H2 urban/rural and junction shift meaning between DfT and NHTSA. H3 injury-based reporting (IBRS) causes internal label shift in STATS19.

## 2. Window and split (fixed)
| Split | Years | Role |
| --- | --- | --- |
| Train | 2016–2021 | fit, class weights, scalers/encoders |
| Validation | 2022 | early stopping, calibration, hyperparameters |
| Test | 2023 | all main tables |
| Locked | 2024 | evaluated once, after code freeze, models chosen on 2023 |

2025 STATS19 is final in the catalogue but is outside the window and unused. No random split. Few-shot: 1/5/10% of target train years (2016–2021) only, seed 42.

## 3. Data and target
- STATS19: collision, vehicle, casualty, 2016–2024, official DfT files only. Unit = collision. Severity: 1 Fatal, 2 Serious, 3 Slight (read the column name from the file: `accident_severity` or `collision_severity`). Record `collision_injury_based` rate by year when the column exists.
- CRSS: accident, vehicle, person CSV, 2016–2024, NHTSA only. Unit = crash. Weight column = `WEIGHT` (Accident file; confirmed in manual DOT HS 813 796, p.44). Severity column = `MAXSEV_IM` (imputed maximum injury severity; `MAX_SEV` is the non-imputed version, manual p.83).
- CRSS code rule (pre-registered): 4 Fatal -> Fatal; 3 Suspected Serious (A) -> Serious; 2 Minor (B) and 1 Possible (C) -> Slight; 0 No Apparent Injury (PDO) dropped; 5 Injured-severity-unknown, 6 Died prior to crash, 8 No person involved dropped. Report the count and percent dropped per code and year.
- Main task: 3-class Fatal/Serious/Slight. Headline task: binary KSI = Fatal + Serious.
- Not used for training: US-Accidents, CISS, FARS, any Kaggle/blog repack.

## 4. Features
Shared, null when unmappable, no mean-filling: hour, day_of_week, month, lighting_dark, weather_coarse (clear/rain/snow/fog/other), speed_limit_mph, junction (binary), n_vehicles, urban_rural, vru_involved (pedestrian/pedal cycle/motorcycle), vehicle_type_mode (car/goods/bus/motorcycle/other), driver_age_band. driver_sex only in the fairness appendix.
Forbidden in X: lat/long/easting/northing, police force, PSU, raw index, any domain indicator, injury or casualty counts or severities, hospital flags. `n_vehicles` is post-event (scene triage framing); drop it if the abstract says "pre-crash".
`results/crosswalk.csv`: source_column, codebook_value, shared_name, transform, semantic_risk in {close, shifted, source-only}.

## 5. Models and metrics
Models, same features, same split: majority, logistic regression (class_weight balanced), shallow random forest, XGBoost, LightGBM. Seed 42. Early stopping on 2022.
Primary metrics: AUPRC (binary KSI), recall of Serious, ECE (15 bins). Also: recall of Fatal, macro-F1, AUROC (KSI), log loss, Brier. Accuracy is never a headline number.
CRSS: weighted column is primary; unweighted is reported alongside.
Uncertainty: 200 bootstrap resamples on the test collisions (CRSS: weighted resampling), 95% CI.
Calibration: none vs Platt vs isotonic, fit on 2022 only, evaluated on 2023. Export reliability bins as CSV.
SHAP: TreeSHAP on 20,000 rows sampled from 2023 (seed 42), mean abs SHAP, rank, 50 bootstraps for rank intervals, Spearman rho UK vs US. No causal reading.

## 6. Transfer (both directions, shared features only)
T0 within-domain. T1 zero-shot. T2 source model + isotonic on target validation. T3 CORAL or domain-classifier importance weights, with proxy A-distance. T4 1/5/10% target train mix versus target-only retrain. Output `results/transfer_matrix.csv`.

## 7. Controls and ablations
Negative control: train labels permuted, XGBoost, test 2023 AUROC reported; stop if > 0.60.
Ablations: drop VRU, drop speed_limit, drop hour (retrain within UK and US). STATS19 IBRS sensitivity: indicator or restricted years if the column is dense enough.

## 8. Leakage checklist (to be ticked in RUNLOG before training)
No injury/casualty/hospital fields in X; no coordinates; no domain column; encoders/scalers fit on train years of the training source only; calibration on 2022 only; 2023 not used for early stopping; 2024 untouched until the locked eval; class weights from train only; CRSS WEIGHT never overwritten; negative control run.

## 9. Stop criteria
Official download fails with no declared fallback; WEIGHT/severity columns cannot be confirmed; negative control AUROC > 0.60; T2 raises AUROC by > 0.02; Fatal recall = 0 everywhere (then the abstract may claim KSI only). Negative transfer results are valid results.

## 10. Writing rules
Paper numbers come only from `results/`. Missing metric: `XX.XX` plus footnote "not computed". Do not call CRSS a census. State that a 2026 cross-jurisdiction paper used US-Accidents (traffic-delay severity), not police injury outcomes. Discussion must cover PDO support mismatch, KABCO vs STATS19 label mismatch, IBRS discontinuity, CRSS weights, and non-deployability.
