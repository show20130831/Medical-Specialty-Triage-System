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

The original row-level split (2,359 / 590 / 738) shared 365 normalised descriptions
between train and test. It can be rebuilt for comparison with
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
