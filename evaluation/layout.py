"""
Where the evaluation pipeline puts its files.

ONE definition, imported by both `process_alignments.py` (which writes the alignments) and
`compute_metrics.py` (which reads them back). The two agreeing by coincidence is exactly how a
sweep ends up scoring an empty set and reporting every run as "never aligned".

    tmp/alignments/<experiment>/<clip>.json       one alignment per clip - BFA and MFA alike
    tmp/alignments/<experiment>/_manifest.json    the configuration that produced them
    tmp/alignments/<experiment>/_log.txt          that run's log
    tmp/mfa/                                      MFA's scratch corpus, removed after a run
    tmp/plots/<experiment>.json / .png            the scorer's detail and error histogram
    evaluation/results/results_<dataset>.csv      the scores, one row per experiment (the
                                                  path is `paths.results_csv` in the yaml)
    tmp/data/<dataset>_data/…                     THE CORPUS: the SRT transcripts. Nothing
                                                  this pipeline writes ever lands here.

`<experiment>` names the run by what it is - `<dataset>__<model>__<key settings>`, built by
`process_alignments.experiment_id` - so the directory is the cache key: the same configuration
always lands in the same place, and `rm -rf tmp/alignments/<experiment>` drops exactly one run
together with its record.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every alignment file, one directory per experiment.
OUTPUT_DIR = os.path.join(REPO_ROOT, "tmp", "alignments")

# The record of one run, kept INSIDE its directory so the two cannot come apart: a manifest
# that outlives its alignments describes files that are not there.
MANIFEST_NAME = "_manifest.json"
LOG_NAME = "_log.txt"

# Where the segment audio and TextGrids of an MFA run sit while MFA works on them.
MFA_WORK_DIR = os.path.join(REPO_ROOT, "tmp", "mfa")

# Per-experiment scoring detail and error histograms.
PLOTS_DIR = os.path.join(REPO_ROOT, "tmp", "plots")

# The corpus trees the parsers wrote. Read-only as far as this pipeline is concerned.
DATASET_SRT_ROOTS = {
    "timit": os.path.join(REPO_ROOT, "tmp", "data", "timit_data"),
    "buckeye": os.path.join(REPO_ROOT, "tmp", "data", "buckeye_data"),
}

# The paths lists, as written by `evaluation/dataset/*_parser.py`. Scoring only reads `text`
# and `boundaries` from the SRTs they point at, which no phoneme set has a say in, so one list
# serves ph66 and std runs alike.
PATHS_LISTS = {
    dataset: os.path.join(REPO_ROOT, "tmp", "paths", f"{dataset}_paths_all.json")
    for dataset in ("timit", "buckeye")
}


def resolve_srt(srt_path):
    """
    A paths list's `srt` entry as an absolute path.

    The parsers store it relative to the repository root, so every consumer would otherwise
    have to be run from there. Anchoring it here means neither script cares about the working
    directory.
    """
    if os.path.isabs(srt_path):
        return srt_path
    return os.path.join(REPO_ROOT, srt_path)


def resolve_repo_path(path):
    """A path from `evaluate.yaml` as an absolute one; relative paths start at the repo root."""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def experiment_dir(experiment):
    """The directory holding one experiment's alignments. Deleting it deletes that run."""
    return os.path.join(OUTPUT_DIR, experiment)


def manifest_path(experiment):
    """Where the manifest of one experiment lives: `tmp/alignments/<experiment>/_manifest.json`."""
    return os.path.join(experiment_dir(experiment), MANIFEST_NAME)


def log_path(experiment):
    return os.path.join(experiment_dir(experiment), LOG_NAME)


def list_experiments():
    """Every experiment on disk that has a manifest, by name."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(name for name in os.listdir(OUTPUT_DIR)
                  if os.path.isfile(manifest_path(name)))


def clip_stem(dataset, srt_path):
    """
    A unique, flat, filesystem-safe name for one clip.

    Taken from the SRT's path relative to the corpus root, because that is the only thing both
    corpora have that is reliably unique: TIMIT's basenames are NOT (`SX61.srt.json` exists
    once per speaker), and Buckeye's `clip_id` is a positional index into the paths list, so it
    would rename every clip the moment the list is regenerated.

        timit    all/MABC0/SX61/SX61.srt.json  ->  all_MABC0_SX61
        buckeye  s01/s0103a.srt.json           ->  s01_s0103a

    A leaf that merely repeats its parent directory is dropped, which is what keeps TIMIT's
    `.../SX61/SX61` from reading `all_MABC0_SX61_SX61`. `assert_unique_stems` checks the whole
    paths list rather than trusting that rule.
    """
    root = DATASET_SRT_ROOTS.get(dataset)
    path = resolve_srt(srt_path)
    rel = os.path.relpath(path, root) if root else os.path.basename(path)

    for suffix in ('.srt.json', '.srt'):
        if rel.endswith(suffix):
            rel = rel[:-len(suffix)]
            break

    parts = [p for p in rel.split(os.sep) if p not in ('', '.', '..')]
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        parts = parts[:-1]
    return "_".join(parts)


def alignment_path(experiment, dataset, srt_path):
    """
    Where one experiment's alignment for `srt_path` lives. The extension is always `.json`;
    which run a file belongs to is said by its directory.
    """
    return os.path.join(experiment_dir(experiment), clip_stem(dataset, srt_path) + ".json")


def assert_unique_stems(dataset, paths_list):
    """
    Refuse a paths list whose clips do not get distinct names.

    A collision would silently overwrite one clip's alignment with another's, and the scorer
    would read the survivor against the wrong ground truth - a wrong number rather than a
    missing one, which is far worse.
    """
    seen = {}
    for entry in paths_list:
        stem = clip_stem(dataset, entry["srt"])
        if stem in seen:
            raise ValueError(
                f"clip_stem collision for {dataset}: {entry['srt']} and {seen[stem]} both "
                f"map to {stem!r}. Their alignments would overwrite each other.")
        seen[stem] = entry["srt"]
    return len(seen)
