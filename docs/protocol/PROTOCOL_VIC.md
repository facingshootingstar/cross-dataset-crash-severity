# Amendment A2: Victoria (Australia) as a fourth jurisdiction

Written and locked on 2026-11-12, before any model is fitted or scored on Victorian rows. Until the lock only file
headers, code lists, row counts, yearly KSI and fatal shares and the share of missing speed zones were inspected.
The protocol of the main study (PROTOCOL.md) and of amendment A1 (PROTOCOL_BAAC.md) apply unless stated here.

## A2.1 Source
Victoria road crash data, Department of Transport and Planning, data.vic.gov.au, licence CC BY 4.0, files Accident,
Person, Vehicle, Node and Atmospheric Condition (download of 2026-11-12). The unit is the accident. Severity codes
1 (fatal), 2 (serious injury) and 3 (other injury) are kept; code 4 (non injury, four records) is dropped. KSI is
code 1 or 2. Serious injury is recorded by the police and is not relabelled; it is a fourth severity instrument.

Pre-lock audit (injury accidents per year, KSI share): 2016 15,779, 0.385; 2017 13,351, 0.431; 2018 12,765, 0.451;
2019 14,398, 0.412; 2020 12,170, 0.397; 2021 14,012, 0.373; 2022 15,041, 0.333; 2023 15,567, 0.346;
2024 15,395, 0.334. The KSI share peaks in 2017-2018 and falls afterwards; this drift is part of the study and no
year is removed because of it. Speed zone is missing (codes 777, 888, 999) for 6-10% of accidents.

## A2.2 Split
Train 2016-2021, validation 2022 (early stopping, model choice, recalibration), test 2023, locked 2024 scored once.
Years before 2016 and after 2024 are not used. Weight 1 for every accident.

## A2.3 Crosswalk to the 12 shared features
| Feature | Rule |
|---|---|
| hour | hour of ACCIDENT_TIME |
| day_of_week | from ACCIDENT_DATE, Sunday = 1 (the DAY_OF_WEEK field changes coding across years) |
| month | from ACCIDENT_DATE |
| lighting_dark | LIGHT_CONDITION 3-6 dark = 1; 1 day and 2 dusk/dawn = 0; 9 missing |
| weather_coarse | first ATMOSPH_COND: 1 clear, 2 rain, 3 snow, 4 fog, 5-7 other, 9 missing |
| speed_limit_mph | SPEED_ZONE in km/h divided by 1.609344; 777, 888, 999 missing |
| junction | ROAD_GEOMETRY 1-4 = 1; 5 and 6 = 0; 7-9 missing |
| n_vehicles | NO_OF_VEHICLES |
| urban_rural | DEG_URBAN_NAME of the first node: RURAL_VICTORIA = rural, every other value = urban |
| vru_involved | a pedestrian, motorcyclist, pillion passenger, bicyclist or e-scooter rider among the persons, or a motorcycle, moped, scooter or bicycle among the vehicles |
| vehicle_type_mode | VEHICLE_TYPE: car 01-03; goods 04-07, 60-63, 71, 72 (utility vehicles as goods, as US pickups); bus 08-09; motorcycle 10-12; other 14-17, 20, 21, 27; bicycle 13 and codes 18, 19, 99 missing; most common type per accident, same tie-break as the other sources |
| driver_age_band | AGE_GROUP of the driver (seating position D) of vehicle A, assigned to the shared band containing the group's midpoint: 0-4, 5-12, 13-15 to <16; 16-17, 18-21 to 16-20; 22-25 to 21-25; 26-29, 30-39 to 26-35; 40-49 to 36-45; 50-59 to 46-55; 60-64 to 56-65; 65-69 and 70+ to 66-75 (>75 cannot be formed) |

Every Victorian field is treated as shifted in meaning except hour, day of week and month.

## A2.4 Analyses
Within Victoria: the five learners of the main protocol. Transfer: the six new directed pairs (Victoria to and from
STATS19, CRSS and BAAC) with T0, T1, T2, T3 and the prior correction over five model seeds; learning curve with the
same N grid, arms and five draws; locked 2024 with T0, T1 and T2. Readings as in the main study.

## A2.5 Predictions and failure rules (new pairs only)
| | Prediction | Fails if |
|---|---|---|
| P11 | Zero-shot transfer loses ranking | the paired 95% interval of T0 minus T1 AUPRC includes 0 on 3 or more of the 6 pairs |
| P12 | Isotonic recalibration on 2022 lowers ECE | ECE after T2 is not lower than T1 on 2 or more of the 6 pairs |
| P13 | A target-only model reaches the zero-shot model with at most 2,500 labelled collisions | this fails on 3 or more of the 6 pairs |
| P14 | Fine-tuning beats target-only at N of 1,000 or less | the mean over draws is not higher at every N of 1,000 or less on 3 or more of the 6 pairs |
| P15 | Correcting the KSI prior with the training-year share lowers ECE | ECE is not lower than T1 on 3 or more of the 6 pairs |
| P16 | Within Victoria, XGBoost ranks KSI above logistic regression | the bootstrap 95% intervals of AUROC overlap |
