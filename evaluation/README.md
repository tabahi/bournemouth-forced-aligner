# Evaluation

Measures how well an aligner places phoneme boundaries on TIMIT and Buckeye, for the BFA
models and for the Montreal Forced Aligner (MFA), against the corpora's hand labels.

This page explains how the harness works. The current results and their analysis are in
[docs/evaluation.md](../docs/evaluation.md).

There are two steps, and one file configures both:

```bash
python evaluation/process_alignments.py     # 1. align: every experiment in evaluate.yaml
python evaluation/compute_metrics.py        # 2. score: one CSV per dataset, one row per experiment
```

Both can be run from any directory, and both can be run again at any time: an experiment that
is already aligned is not aligned again, and one that is already in the CSV is not scored
again. Adding a model or a setting to the yaml only costs the new runs.

## Before the first run

1. **Corpora.** Set `paths.audio_roots` in `evaluate.yaml` to where TIMIT and Buckeye are.
2. **Ground truth.** Each parser reads a corpus and writes the transcripts and reference
   boundaries the other steps use (`tmp/data/<dataset>_data/` and
   `tmp/paths/<dataset>_paths_all.json`):

   ```bash
   python evaluation/dataset/timit_parser.py
   python evaluation/dataset/buckeye_parser.py
   ```

   Run them again only when a parser changes. Alignments made before that no longer match
   the new segments and have to be redone (`run.existing: redo`).
3. **MFA**, if it is in your experiments, in its own conda environment:

   ```bash
   conda create -n mfa -c conda-forge montreal-forced-aligner
   ```

   The environment's name goes in `mfa.conda_env`. You do not activate it: the sweep runs in
   your usual environment and calls `conda run -n mfa mfa ...` itself. The acoustic model and
   dictionary are downloaded on first use.

## evaluate.yaml

Every option is described in the file, with its possible values. The sections are:

| section | what it holds |
|---|---|
| `paths` | corpus audio, the local release directory, the results CSVs |
| `models` | every model an experiment may name: published BFA models, local checkpoints, MFA |
| `run` | what to do with an experiment that already exists, logging, threads |
| `bfa` | the value of every BFA setting, for experiments that do not set it themselves |
| `mfa` | conda environment, number of jobs, extra arguments |
| `experiments` | the list of runs |
| `scoring` | tolerances, offsets and the other scorer options |

Every key is required and an unknown key is an error, so the file is always the complete
description of a run.

### Experiments

`experiments` is a list. In each entry any key can be one value or a list, and the entry
stands for every combination; the entries run one after another.

```yaml
experiments:
  - name: released models and MFA
    dataset: [timit, buckeye]
    model: [CUPE-3-en, mbfa-latin, mfa-english_us_arpa]      # 2 x 3 = 6 runs

  - name: boundary bonus
    dataset: timit
    subset: 100                                              # 100 random clips, same for all
    model: [CUPE-3-en, mbfa-latin]
    boundary_bonus_weight: [0, 15, 30]                       # 2 x 3 = 6 runs
```

- `dataset` and `model` are required. `subset` is optional (`-1`, every clip).
- Any key of the `bfa` section can be added to an entry to override it or sweep it.
- `enabled: false` keeps an entry in the file without running it.
- MFA takes none of the BFA settings, so in an entry that sweeps them it runs once.
- A run described twice, in one entry or two, is done once.

### What a run is called

Each experiment gets a name made of the dataset, the model and its settings, and its
alignments go to a directory of that name:

```
tmp/alignments/timit__CUPE-3-en__b30_d6_s10_grid/
tmp/alignments/buckeye__mbfa-latin__b0_d6_s10_grid_n10/
tmp/alignments/timit__mfa-english_us_arpa/
```

| part | meaning |
|---|---|
| `b30` | `boundary_bonus_weight` 30 |
| `d6` | `min_phone_frames` 6 |
| `s10` | `soft_phone_frames` 10 |
| `grid` / `subframe` | `sub_frame_onsets` off / on |
| `n10` | `subset` 10; absent for the full corpus |
| `dur`, `bs`, `cov`, `lang`, `bct`, `lcl`, `chunk`, `seg` | `duration_max`, `batch_size`, `coverage_min`, `lang`, `bad_confidence_threshold`, `low_confidence_level`, `chunk_audio`, `segment_max_seconds`. Each appears only when it differs from its usual value (30, 16, 0.6, en-us, 0.6, null, true, 120) |

The decoder values in the name are the ones that actually ran. `null` in the yaml means "the
value the model was released with", so `boundary_bonus_weight: null` and `30` give the same
name for a model released at 30, and that run is done once.

Inside the directory:

```
<clip>.json       one alignment per audio file
_manifest.json    the full configuration and the run's statistics
_log.txt          the run's log
```

The name is the cache key. To throw a run away, delete its directory.

## Step 1: aligning

```bash
python evaluation/process_alignments.py                    # everything enabled in the yaml
python evaluation/process_alignments.py --list             # what would run, and what is cached
python evaluation/process_alignments.py --only 'timit__*'  # only names matching the pattern
python evaluation/process_alignments.py --config other.yaml
python evaluation/process_alignments.py 4 0                # shell 1 of 4, see below
```

`--list` prints each experiment as `new`, `partial` (started, not finished) or `done`.

With `run.existing: skip`, a `done` experiment is skipped and a `partial` one continues where
it stopped. With `redo`, the directory is deleted and the experiment is aligned from scratch.

If one experiment fails, the rest still run; the failures are listed at the end and the exit
code is 1.

**In parallel.** `process_alignments.py N K` runs every N-th experiment starting at the K-th,
so N shells with K = 0 … N-1 cover the list between them:

```bash
python evaluation/process_alignments.py 4 0 &
python evaluation/process_alignments.py 4 1 &
python evaluation/process_alignments.py 4 2 &
python evaluation/process_alignments.py 4 3 &
```

Do not let two shells run MFA at the same time; MFA keeps one shared database per user.

**Cost** on one CPU process: a full TIMIT run (6,300 clips) is a little over an hour per BFA
model. A Buckeye clip is a whole interview of about 130 segments, so use `subset` for a quick
look. MFA took about a minute for 20 TIMIT clips, most of it start-up.

## Step 2: scoring

```bash
python evaluation/compute_metrics.py                    # score what is not in the CSV yet
python evaluation/compute_metrics.py --list             # what is on disk, and what is scored
python evaluation/compute_metrics.py --only '*__n100'
python evaluation/compute_metrics.py --rescore          # recompute rows that already exist
python evaluation/compute_metrics.py --rescore --yes    # ... without asking
```

The scorer does not read the `experiments` list. It looks at every finished run under
`tmp/alignments/`, reads the CSVs, and scores only what has no row yet. Each dataset has its
own file, `evaluation/results/results_timit.csv` and `results_buckeye.csv`
(`paths.results_csv` in the yaml).

A row is replaced in two cases, and in both the scorer asks first:

- the experiment was aligned again after its row was written;
- you passed `--rescore`, which is what to do after changing the `scoring` section or
  re-running a parser.

A copy of the CSV is kept beside it as `.bak` before any row is replaced. Without a terminal
to ask on (a batch job), rows are only replaced with `--yes`.

For each experiment scored it also prints the full breakdown and writes
`tmp/plots/<experiment>.json` (the detail, including `all_distances`: for every hand-labelled
boundary, the distance to the nearest predicted onset) and `.png` (the histogram of those
distances).

## Step 3: the figures for the docs

[docs/evaluation.md](../docs/evaluation.md) shows the runs in `results/` and the plots in
`plots/`. To refresh them after a new full-corpus run, copy that run's `tmp/plots/<experiment>.png`
and `.json` into `evaluation/plots/`, then redraw the two comparison figures:

```bash
python evaluation/plot_summary.py      # evaluation/plots/summary_f1.png, summary_distance.png
```

It draws one bar group or line per method from the full-corpus rows (`cfg_subset` -1) of each CSV.
Only the `.png` files in `plots/` are committed. The JSONs are ~3 MB each and stay local.

### Reading the CSV

One file per dataset, one row per experiment. The columns, in order:

| columns | meaning |
|---|---|
| `experiment` | the run's name, as above |
| `method`, `dataset`, `cfg_model`, `cfg_ckpt`, `phoneme_set` | `BFA` or `MFA`, the corpus, the model |
| `notes` | a short label; `*` marks a decoder value left to the model |
| `cfg_*` | the settings as they ran, and when the alignment finished |
| `f1_<t>ms`, `r_value_<t>ms`, `recall_<t>ms`, `precision_<t>ms` | boundary scores at each tolerance |
| `files`, `total_known`, `total_pred`, `detection_ratio` | how much was scored |
| `abs_err_*`, `signed_err_median`, `missed_pct`, `fakes_pct` | timing error of the matched boundaries, in ms |
| `ex_*` | the same scores with gross errors left out; not comparable to a published F1 |
| `mn_*` | mean error per manner class: filled for every method, so use these to compare models |
| `err_*`, `std_*` | mean error per phoneme group of the ph66 / std set |

The headline number is `f1_25ms`: the F1 of phoneme onsets within 25 ms of a hand-labelled
one. Only compare rows with the same `files` count; a 100-clip row
reads a few points away from a full-corpus one.

How boundaries are matched is described at the top of `compute_metrics.py`. How the corpus
labels become reference boundaries is in [LABEL_MAPPING.md](LABEL_MAPPING.md).

## Files

| file | role |
|---|---|
| `evaluate.yaml` | the configuration |
| `process_alignments.py` | step 1 |
| `align_with_mfa.py` | the MFA part of step 1; not run directly |
| `compute_metrics.py` | step 2 |
| `plot_summary.py` | step 3: the comparison figures in docs/evaluation.md |
| `config.py` | reads and checks the yaml |
| `layout.py` | where files go |
| `dataset/` | the corpus parsers (`timit_parser.py`, `buckeye_parser.py`), the label tables (`std_labels.py`), and the training probe cache (`probe_cache.py`) |
| `LABEL_MAPPING.md` | how each corpus's labels become reference boundaries |
| `results/results_<dataset>.csv` | the scores |
| `plots/` | the histograms and figures that docs/evaluation.md shows |
