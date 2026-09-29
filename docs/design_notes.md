# Design notes

## Goal and boundaries

The pipeline's job is to make a safe training decision for each daily file. It is intentionally not a full production data platform. The generator, storage, checks, model, and tests are all small enough to explain in a short walkthrough.

The data is synthetic. The label is present in the same daily file only to make the exercise runnable. In a real churn system, features and labels would arrive on different schedules.

## Ingestion and files

The pipeline derives a date from `user_activity_YYYY-MM-DD.csv`. It accepts an optional suffix after the date so two candidates for a date can coexist during a correction. It does not use the current time to decide a file's data date.

Every file is hashed before processing. A hash already in the manifest is skipped, so repeating a run is safe. The same name with different contents has a new hash and is processed again. If several unrecorded candidates have the same date, the newest modification time wins and older candidates are recorded as superseded. This prevents two versions from being written as processed data for one date.

Files are processed in date order. A gap in dates is a warning. The drift reference is the latest earlier passing processed day, not merely the prior item encountered in a directory.

Unreadable files, wrong delimiters that leave the schema unusable, empty files, header-only files, and invalid filenames become quarantine outcomes. Header whitespace and a UTF-8 BOM are repaired because they are ordinary export artifacts. Column renaming and case changes are not repaired because hiding an upstream schema change would be unsafe.

One bad file does not stop the batch. An expected quarantine exits successfully. An unexpected write or model error is allowed to fail the command instead of being mistaken for an ordinary data-quality event.

This repository does not address an upload still in progress. In a deployed setting I would require a done marker or a minimum file age before pickup. The pipeline also does not coordinate simultaneous runs. A scheduler concurrency group or lock file is the right control there.

## Schema, cleaning, and validation

The raw CSV is checked for required fields before cleaning. Numeric coercion uses invalid values as missing values. This means text such as `N/A` counts against the original null-rate check rather than being silently repaired.

The validation profile is built after harmless normalization but before imputation. It captures nulls, exact duplicates, invalid ranges, normalized plan types, labels, dates, and summary values. That sequence matters. If a file has many missing sessions, it fails because of the original defect rate even though a median exists.

Cleaning trims and lowercases `plan_type`, drops only exact duplicate rows, turns impossible configured values into missing values, and imputes small numeric missing rates with that file's median. It drops and counts rows missing `user_id` or `plan_type` when their pre-clean null rates pass the gate, since those rows cannot enter training safely. It does not guess which of two conflicting records for one user is correct. That situation is recorded as a warning.

Range checking catches impossible values, not ordinary in-range outliers. Values inside configured bounds are retained. Small null repair can still be problematic during real distribution changes, so the report retains cleaning counts and the pre-imputation profile.

The error checks are intentionally strict: missing required columns, insufficient rows, null rates, duplicate rates, impossible-value rates, unknown plan values, labels, churn plausibility, and date mismatch block the file. The warning checks are intentionally nonblocking: drift, added columns, zero variance, conflicting users, and gaps provide a review signal without making the system reject a usable day.

PSI uses quantile bins from the reference and epsilon-protected proportions. It is skipped when either input is too small or the reference has no usable variance. A previous-day reference can create false alarms and can miss gradual drift. A fixed baseline alongside the day-to-day comparison is the next improvement.

## Storage and state

Passing data is written to a temporary Parquet file in the target directory and then renamed. The manifest and reports use the same temporary-file pattern. This prevents a partial output from appearing complete after a crash.

A failing raw CSV is moved byte-for-byte to `data/quarantine` and its report is retained under `logs/reports`. The manifest connects the file hash, status, report path, and quarantine or processed path. Quarantine is evidence, not garbage.

If the manifest is lost, the pipeline can reprocess raw files. Date-keyed Parquet outputs are atomically replaced, so this is safe for this small project, though it can repeat training. A real deployment would back up state and make model-run lineage more explicit.

## Training decisions

There is one gate function. It is reached after a successful processed write and requires a passing report, enough accumulated rows, and enough positive labels. An end-to-end spy test asserts training is never called for a quarantined file. The manifest marks a passing file complete only after training finishes or is deliberately skipped, so an unexpected training failure can be retried.

The first usable training run uses a group split by `user_id`. Once multiple dates exist, the latest valid date is held out. A random row split would leak recurring users across train and test data. A holdout with only one class is skipped because ROC AUC would be undefined.

Scaling and encoding are inside the sklearn pipeline, so they fit only the training rows. The encoder ignores unseen categories at inference. Logistic regression with balanced class weights is sufficient for the exercise. PR AUC is reported with ROC AUC because accuracy would be misleading for a churn label with roughly 20% to 30% positives.

The first model is champion. Later challengers are scored on the same holdout as the champion. A challenger needs to improve ROC AUC by the configured margin before promotion. This makes promotion conservative. The cost is that a slightly better model may remain inactive, but the score and decision remain in the metrics JSON.

Training history grows without bound in this take-home. A rolling window is the next practical change. Model artifacts record their feature list, dates, row counts, seed, and pandas and scikit-learn versions because loading a model under a different scikit-learn version may fail.

## Costs of the design

- Pre-imputation null checks can quarantine data that cleaning could technically complete. The tradeoff favors preventing a silently degraded model over running on time.
- Quarantine can become an ignored folder. The run log summarizes it; alerting and ownership would be the next operational work.
- Previous-day drift is cheap and understandable but is not a complete monitoring strategy.
- A promotion margin can retain a marginally weaker champion.
- Hand-written validation is easy to explain here but will become maintenance work as the schema expands.
- The simple model can underperform a tuned model. That is deliberate because data quality, not model selection, is the subject of this project.

## Review checklist

For a quick walkthrough, run `make demo`, open the two quarantine files and their reports, inspect `state/manifest.json`, and compare a metrics JSON with `models/champion.joblib`. Then run the pipeline once more and show the `nothing to process` log line.
