# Analysis note B2: police force areas and départements as small new jurisdictions

Written and locked on 2026-11-12 before the analysis is run. Post hoc with respect to PROTOCOL.md and its amendments:
it tests the guidance drawn from the learning curve on real small units instead of uniform samples of a country.

Units: every STATS19 police force area and every metropolitan French département with at least 100 collisions in
the training years (2016-2021 for STATS19, 2019-2021 for BAAC) and at least 300 collisions in the 2023 test year.
N is the number of the unit's training-year collisions.

Arms, all scored on the unit's 2023 collisions with the primary model:
- foreign zero-shot: a model trained on another country (STATS19 units: CRSS and BAAC; BAAC units: STATS19 and CRSS);
- foreign + Platt: Platt scaling of the foreign model on the unit's training-year collisions;
- foreign fine-tuned: boosting continued on the unit's training-year collisions (as in the learning curve);
- local only: a model trained on the unit's training-year collisions alone (15% holdout for early stopping);
- national: the model of the unit's own country (its training data include the unit), as a reference.

Readings: AUPRC, AUROC, ECE. The learning curve predicts that local only reaches the foreign zero-shot model once
N is about 2,500 or more, that fine-tuning beats local only up to N of about 1,000, and that Platt scaling lowers
ECE against zero-shot. The analysis reports, by N band (< 1,000; 1,000-2,500; > 2,500), the share of units in which
each prediction holds.
