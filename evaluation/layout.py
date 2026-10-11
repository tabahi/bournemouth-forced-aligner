"""
Where the scoring pipeline puts its files.

ONE definition, imported by both `process_alignments.py` (which writes the alignments) and
`compute_metrics.py` (which reads them back). The two agreeing by coincidence is exactly how a
sweep ends up scoring an empty set and reporting every extension as "never aligned".

    tmp/alignments/<dataset>/<bfN>/<clip>.json    every BFA alignment, and nothing else
    tmp/manifests/bf<N>.json                      what configuration produced each <bfN>
    tmp/manifests/logs/…                          per-experiment logs
    tmp/manifests/notes_<dataset>.txt             the running notes
    tmp/plots/                                    error histograms
    evaluation/results/*.csv                      the scores themselves - IN the repository,
                                                  and committed, because a score is a result
                                                  and the rest of this is scratch
    tmp/data/<dataset>_data/…                     THE CORPUS: SRT transcripts and MFA's own
                                                  `.mfa.json` alignments. Nothing this
                                                  pipeline writes ever lands here.

The split exists so a sweep is disposable: `rm -rf tmp/alignments` throws away every BFA
alignment ever produced without touching the corpus or the record of what was run, and
`rm -rf tmp/alignments/timit/bf3` throws away exactly one experiment.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every `.bfN` alignment file, one directory per (dataset, experiment).
OUTPUT_DIR = os.path.join(REPO_ROOT, "tmp", "alignments")

# Manifests, logs and notes - the record of what was run, which outlives the alignments.
MANIFESTS = os.path.join(REPO_ROOT, "tmp", "manifests")
MANIFEST_DIR = MANIFESTS
LOG_DIR = os.path.join(MANIFESTS, "logs")

# The scored CSV rows. Inside the package and version-controlled: everything else here is
# regenerable scratch, but a score is the output of a run that may not be repeatable (a
# checkpoint gets replaced, a corpus moves), so it belongs in the history.
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")

# Per-extension error histograms. Images, so they stay out of the repository.
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


def vs_tag(vs_ext):
    """`.bf3.json` -> `bf3`, `.mfa.json` -> `mfa`."""
    return str(vs_ext).replace(".json", "").lstrip(".")


def experiment_dir(dataset, vs_ext):
    """The directory holding one experiment's alignments. Deleting it deletes that run."""
    return os.path.join(OUTPUT_DIR, dataset, vs_tag(vs_ext))


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


def alignment_path(dataset, srt_path, vs_ext):
    """
    Where one experiment's alignment for `srt_path` lives.

    BFA runs go under `OUTPUT_DIR`. MFA alignments stay beside the SRT as `<id>.mfa.json`,
    because that is where MFA itself wrote them and the corpus is not ours to reorganise.
    """
    tag = vs_tag(vs_ext)
    if tag.startswith("bf"):
        return os.path.join(experiment_dir(dataset, vs_ext),
                            clip_stem(dataset, srt_path) + ".json")

    # Beside the SRT. `srt_path.replace('.srt', vs_ext)` would give `<id>.mfa.json.json`,
    # since the stored name is `<id>.srt.json`.
    path = resolve_srt(srt_path)
    for suffix in ('.srt.json', '.srt'):
        if path.endswith(suffix):
            return path[:-len(suffix)] + vs_ext
    return path + vs_ext


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
