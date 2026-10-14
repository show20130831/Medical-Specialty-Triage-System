# Model Training

The served model is produced by the `triage_training` package. The data pipeline and
TF-IDF baseline run on any CPU; fine-tuning needs a CUDA GPU. Every run records the
Git commit, configuration, dataset hashes, and package versions, so a model can be
traced back to the exact code and data that produced it.

## Pipeline

```text
mtsamples.csv
  -> clean: drop rows without specialty or description
  -> label: 40 source specialties -> 8 groups (configs/taxonomy.yaml)
  -> deduplicate: one row per normalised description, resolve conflicting labels
  -> split: stratified train / val / test with zero shared descriptions
  -> validate: label coverage, class sizes, blank text, split overlap
  -> baseline (TF-IDF + logistic regression) and fine-tuning (PubMedBERT)
  -> export: serving bundle + metrics + per-row scores (no text)
```

| Command | Output |
| --- | --- |
| `python -m triage_training prepare-data` | `outputs/data/dedup_seed42/` splits and `data_report.json` |
| `python -m triage_training baseline` | `outputs/runs/baseline_*/` metrics and per-row probabilities |
| `python -m triage_training finetune` | `outputs/runs/pubmedbert_*/model/` serving bundle |
| `python -m triage_training finetune --smoke` | Tiny pipeline check that finishes on CPU |
| `python -m triage_training evaluate-model` | Scores an existing model bundle on a prepared split, e.g. to check CPU parity |
| `python -m triage_training report` | `reports/model_evaluation.md` statistical evaluation |
| `python -m triage_training summarize-splits` | `reports/split_variation.md` variation across data split seeds |
| `python -m triage_training validate-model` | `reports/release_validation.md` release gate decision |
| `python -m triage_training monitor-drift` | Drift check of a `/metrics` snapshot against the reference |

All hyperparameters live in [configs/train.yaml](../configs/train.yaml). The loader
rejects missing, unknown, or mistyped keys. `--seed` overrides the training seed for
repeated runs; the split seed stays fixed so all runs share the same test set.

## Data decisions

MTSamples lists many transcriptions under several specialties, so the same description
appears more than once, sometimes with different labels.

- 1,492 of 3,687 labeled rows are duplicates after normalising case, punctuation, and
  whitespace.
- 409 descriptions carry conflicting labels. Labels from a direct specialty mapping
  are trusted over keyword-rule labels; 209 that remain tied are dropped.
- The result is 1,986 unique descriptions split 1,270 / 318 / 398 with no shared
  description between splits.

Splitting rows instead of unique descriptions (2,359 / 590 / 738) shares 365 normalised
descriptions between train and test. That row-level split can be built for comparison with
`prepare-data --keep-duplicates`, which writes `outputs/data/rowlevel_seed42/`.

`outputs/` contains medical text and model weights and is excluded from Git and Docker.

## Training on a Linux GPU server

```bash
git clone https://github.com/show20130831/Medical-Specialty-Triage-System.git
cd Medical-Specialty-Triage-System
python3 -m venv .venv && source .venv/bin/activate
pip install "torch==2.14.0"            # Linux wheels from PyPI include CUDA
pip install -e ".[train,inference,finetune]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Copy the raw dataset from your own machine:

```bash
scp data/mtsamples.csv <user>@<server>:~/Medical-Specialty-Triage-System/data/
```

Run long jobs inside `tmux` so they survive an SSH disconnect:

```bash
tmux new -s train
nvidia-smi                               # confirm the shared GPU is idle
python -m triage_training prepare-data
python -m triage_training baseline
python -m triage_training finetune --smoke
python -m triage_training finetune
# detach with Ctrl+B then D; reattach later with: tmux attach -t train
```

On an RTX 4090 the trainer uses bf16 automatically; older GPUs use fp16 and CPUs use
fp32. Commit any code change before a real run so `run_manifest.json` records a clean
commit instead of `"dirty": true`.

## Run outputs

```text
outputs/runs/pubmedbert_dedup_seed42_trainseed42_<UTC>/
├── model/                  # serving bundle: weights, tokenizer, training_config.json,
│                           # metrics.json, data_report.json, reload_check.json
├── metrics.json            # val/test accuracy, macro F1, per-class report, confusion matrix
├── predictions_val.npz     # source_row, y_true, logits (no text)
├── predictions_test.npz
├── run_manifest.json       # git commit, config, dataset hashes, environment, precision
└── trainer_log.json        # loss and validation curve
```

Copy the bundle back to the serving machine. Move any existing
`models/pubmedbert_description/` aside first, otherwise `scp -r` nests the new `model/`
inside it:

```bash
scp -r <user>@<server>:~/Medical-Specialty-Triage-System/outputs/runs/<run>/model \
    models/pubmedbert_description
```

## Evaluation report

`report` combines saved per-row scores into an aggregate report that is safe to
commit:

- Test accuracy and macro F1 with 95% bootstrap intervals, and seed variation
- Fine-tuned model vs. baseline: paired bootstrap difference and exact McNemar test
- Per-class metrics, frequent confusions, and accuracy by label provenance
  (direct specialty mapping vs. keyword rules)
- Temperature scaling fitted on validation, ECE and NLL before and after, and the
  confidence threshold below which the API should answer "uncertain"
- Leakage study on the row-level split: accuracy on test rows seen in train,
  unseen, and with conflicting labels

```bash
python -m triage_training prepare-data --keep-duplicates
python -m triage_training finetune --dataset rowlevel_seed42
python -m triage_training report     --model outputs/runs/pubmedbert_dedup_seed42_trainseed42_<UTC>     --baseline outputs/runs/baseline_dedup_seed42_trainseed42_<UTC>     --seed-run outputs/runs/pubmedbert_dedup_seed42_trainseed1_<UTC>     --seed-run outputs/runs/pubmedbert_dedup_seed42_trainseed2_<UTC>     --rowlevel outputs/runs/baseline_rowlevel_seed42_trainseed42_<UTC>     --rowlevel outputs/runs/pubmedbert_rowlevel_seed42_trainseed42_<UTC>
```

`report` also writes `calibration.json` (temperature and uncertainty threshold) and
`monitoring_reference.json` (expected prediction profile for drift checks) into the
primary run's `model/` bundle. Release validation and monitoring are described in
[validation-and-operations.md](validation-and-operations.md).
