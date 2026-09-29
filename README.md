# Churn data pipeline

This is a small daily pipeline for a churn prediction model. It ingests synthetic CSV files, checks whether each file is trustworthy before cleaning can hide defects, stores passing data as Parquet, and trains only when the gate allows it.

I kept the model simple on purpose. The point of this project is the data contract: a bad daily file must be visible, recoverable, and unable to influence a model.

## Run it

Use Python 3.14 or another version supported by the pinned dependencies.

```bash
make setup
make demo
```

`make demo` clears only generated demo data, reports, state, model artifacts, and logs. It then generates the six daily files and processes them from scratch.

To run the tests:

```bash
make test
```

To rerun the pipeline without regenerating data:

```bash
make run
```

A rerun after a completed demo logs `nothing to process` and exits successfully. The Makefile uses `.venv/bin/python` by default, and you can override it with `make test PYTHON=python` if you already activated an environment.

## What the demo does

The generator creates six deterministic daily files in `data/raw`.

1. Sep 24 passes, trains, and becomes the first champion.
2. Sep 25 passes, retrains, and compares the challenger with the champion.
3. Sep 26 passes with numeric drift warnings.
4. Sep 27 fails several independent checks and moves to quarantine.
5. Sep 28 fails the minimum-row check and moves to quarantine.
6. Sep 29 passes and resumes training without manual cleanup.

The normal files have realistic small defects: roughly 2% missing session minutes, 0.75% exact duplicates, and 1% plan-type casing or whitespace variants. Those defects are repaired and reported. They are not failures. If a passing file has a few rows without `user_id` or `plan_type`, cleaning drops and counts those rows because training requires both values.

The bad Sep 27 file has a missing required column, 35% missing sessions, negative login-day values, and an unknown `PRO_v2` plan. It demonstrates that the checks fail independently and together.

The synthetic label comes from a logistic probability. More days since login and more support tickets raise churn risk. More sessions and longer sessions lower it. The generator prints a quick signal check so the model is demonstrably better than a dummy baseline without turning this into a model-tuning exercise.

## Design decisions

### Pipeline flow

```text
                         data/raw/user_activity_YYYY-MM-DD.csv
                                            |
                                            v
                    filename date + SHA-256 hash + manifest lookup
                               |                         |
                               |                         -> known hash: skip
                               v
                            read CSV
                               |
                 unreadable or empty? -> report -> quarantine -> manifest
                               |
                               v
       raw schema checks + pre-imputation quality and drift checks
                               |
                  error-severity failure? -> report -> quarantine -> manifest
                               |
                               v
      normalize plan type, drop exact duplicates, repair small defects
                               |
                               v
                  atomic Parquet write -> processed history -> manifest
                               |
                               v
                          training gate
                  /                           \
        gate denies training                     gate allows training
                  |                                      |
                  v                                      v
             log skip reason          train challenger + dummy baseline
                                                     |
                                                     v
                     current champion and challenger score the same holdout
                                                     |
                                                     v
                             promote only when improvement clears the margin
                                                     |
                                                     v
                                  versioned model artifacts + metrics JSON
```

The file date comes from the filename, never from the system clock. Files are processed in date order. Each SHA-256 hash is recorded in `state/manifest.json`, so identical reruns are skipped. A corrected file with the same name but new contents has a new hash and is processed again.

For a passing file, the pipeline writes a date-keyed Parquet file atomically. For a failing file, it writes the JSON report and moves the untouched original CSV to `data/quarantine`. Expected validation failures are normal outcomes, not process crashes.

### Why the gate can be trusted

The most important choice is that quality checks see values before imputation. For example, a file with 35% missing `sessions_last_30d` values fails before median imputation could make it look complete. Cleaning repairs small defects and counts any rows dropped for missing training keys. Validation decides whether the file is acceptable at all.

Training is allowed only when all three conditions are true:

- Validation has no error-severity failures.
- The processed Parquet write succeeded.
- The processed history has at least 1,000 rows and 100 positive churn labels.

Warnings do not block a run. Drift is a warning because a warning is useful evidence, while automatically rejecting a valid file based on a noisy comparison would be too aggressive.

## Validation

Error checks block storage and training:

- Required columns and values coercible to the expected numeric forms.
- At least 1,000 rows.
- Pre-imputation null rate at most 5% per column.
- Duplicate rate at most 5%.
- Out-of-range rate at most 1% for configured numeric bounds.
- Unknown normalized plan types at most 0.5%.
- Valid binary, non-null churn labels and a 5% to 60% overall churn rate.
- A `date` value that matches the filename date in every row.

Warning checks include numeric mean shift and PSI, plan-type proportion shift, large row-count or churn-rate movement, extra columns, zero-variance features, conflicting same-day user records, and date gaps.

All thresholds live in [`config.yaml`](config.yaml). Reports record the check name, stage, severity, observed value, threshold, and result.

### Training and promotion

The model is a scikit-learn pipeline with numeric scaling, one-hot encoding for `plan_type`, and class-balanced logistic regression. A `DummyClassifier` baseline is reported alongside ROC AUC and PR AUC.

When at least two valid dates are available, the newest valid day is held out. On a one-day first run, the split is grouped by `user_id`. This avoids evaluating a user in both train and test data. Preprocessing is fitted only on training data because it lives inside the sklearn pipeline.

The first trained model is the champion. Later models are challengers. The current champion and new challenger are both evaluated on the same current holdout. A challenger replaces the champion only if its ROC AUC improves by the configured margin of 0.01. This avoids promoting a model because of a tiny, likely noisy score difference.

## Files produced by a demo

- `data/processed/` contains four Parquet files for the passing days.
- `data/quarantine/` contains the original Sep 27 and Sep 28 CSV files.
- `logs/reports/` contains one JSON report per input file.
- `state/manifest.json` records hashes and outcomes.
- `models/` contains challenger artifacts, the current champion, and metrics JSON files.

Real outputs from a completed run are committed under [`docs/sample_outputs`](docs/sample_outputs): a passing report, a failing report, and the run log.

## Configuration and automation

`config.yaml` holds all active validation, drift, training, and promotion thresholds. The generator's data distributions are intentionally kept in `scripts/generate_data.py`, where their relationship to the scenario is easy to inspect.

The GitHub Actions workflow runs on pushes to `main`, on demand, and on a daily UTC schedule. It regenerates synthetic inputs, runs the demo and tests, then uploads reports and metrics as an artifact. It does not claim to receive live production data.

## Assumptions

- Each file is one daily user-activity snapshot and its label is already available. Real churn labels would arrive after an outcome window.
- `user_id` can recur across days, which is why the holdout design matters.
- New plan types are treated as an error until someone intentionally updates the allowed list.
- A local process runs the pipeline. Concurrent runs are not supported.

## Known limitations

- Drift compares with the latest earlier passing day. This can be noisy and can miss slow drift. A fixed training-window baseline would be a useful addition.
- The pipeline does not protect against a file still being uploaded. A done marker or file-age rule is the practical next step.
- Late corrections replace the processed data for their own date, but this project does not automatically replay every later model run.
- The retained history is small enough for this take-home. A real system would likely use a rolling training window.
- Hand-written validation is appropriate here because every check is small and explainable. A larger schema would justify Pandera or Great Expectations.

## Next steps

Add a fixed training-window drift baseline, an upstream done-marker rule, replay semantics for late corrections, a rolling training window, and alerting for quarantine outcomes.

For the deeper edge-case decisions, see [`docs/design_notes.md`](docs/design_notes.md).
