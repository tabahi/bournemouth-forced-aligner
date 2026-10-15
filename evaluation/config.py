"""
Reads `evaluation/evaluate.yaml`, the one place the evaluation is configured.

Both `process_alignments.py` and `compute_metrics.py` load it through here. Nothing has a
default in the code: a key that is missing or misspelt is an error naming it, so the file on
disk is always the whole description of what a run does.
"""

import os

import yaml

from evaluation.layout import REPO_ROOT, DATASET_SRT_ROOTS, resolve_repo_path

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "evaluation", "evaluate.yaml")

# The decoder settings. `null` on any of them means "what the model was released/trained with".
DECODER_AXES = ("boundary_bonus_weight", "min_phone_frames", "soft_phone_frames",
                "sub_frame_onsets")
# Passed to `BFA` as they are.
ALIGNER_KWARGS = ("lang", "device", "bad_confidence_threshold", "low_confidence_level",
                  "chunk_audio", "segment_max_seconds")
# Used by the sweep itself.
RUN_SETTINGS = ("duration_max", "batch_size", "coverage_min")
# Everything under `bfa:`, and everything an `experiments:` entry may override or sweep.
BFA_SETTINGS = DECODER_AXES + ALIGNER_KWARGS + RUN_SETTINGS

SECTIONS = {
    "paths": ("audio_roots", "releases_dir", "results_csv"),
    "run": ("existing", "vocab_analysis", "warn_level", "torch_threads"),
    "bfa": BFA_SETTINGS,
    "mfa": ("conda_env", "num_jobs", "extra_args", "keep_workdir"),
    "scoring": ("tolerance_ms", "align_max_dist", "offset_start", "offset_end",
                "use_end_boundaries", "measure_group_errors", "merge_two_token_phones",
                "limit_clips", "plots"),
}
EXPERIMENT_KEYS = ("name", "enabled", "dataset", "subset", "model") + BFA_SETTINGS
MODEL_KEYS = {
    "bfa": ("model_name", "ckpt_path"),                    # exactly one of the two
    "mfa": ("method", "acoustic_model", "dictionary"),     # all three
}


def _check_keys(where, found, allowed, required=None):
    unknown = sorted(set(found) - set(allowed))
    if unknown:
        raise KeyError(f"evaluate.yaml: unknown key(s) {unknown} in {where}; "
                       f"it takes {sorted(allowed)}")
    missing = sorted(set(allowed if required is None else required) - set(found))
    if missing:
        raise KeyError(f"evaluate.yaml: {where} is missing {missing}")


def load_config(path=None):
    """The parsed, checked configuration. Paths in it are absolute afterwards."""
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    _check_keys("the top level", cfg, tuple(SECTIONS) + ("models", "experiments"))
    for section, keys in SECTIONS.items():
        _check_keys(f"`{section}:`", cfg[section] or {}, keys)

    paths = cfg["paths"]
    unknown = sorted(set(paths["audio_roots"]) - set(DATASET_SRT_ROOTS))
    if unknown:
        raise KeyError(f"evaluate.yaml: paths.audio_roots names {unknown}, but the datasets "
                       f"are {sorted(DATASET_SRT_ROOTS)}")
    if "{dataset}" not in str(paths["results_csv"]):
        raise ValueError("evaluate.yaml: paths.results_csv needs `{dataset}` in it, e.g. "
                         "evaluation/results/results_{dataset}.csv - each dataset has its own file")
    paths["results_csv"] = resolve_repo_path(paths["results_csv"])

    if cfg["run"]["existing"] not in ("skip", "redo"):
        raise ValueError(f"evaluate.yaml: run.existing is {cfg['run']['existing']!r}; "
                         f"it is 'skip' or 'redo'")

    for name, entry in cfg["models"].items():
        if os.sep in name or name.startswith((".", "_")) or "__" in name:
            raise ValueError(f"evaluate.yaml: model name {name!r} becomes part of a directory "
                             f"name, so no '/', no '__', and no leading '.' or '_'")
        if entry.get("method") == "mfa":
            _check_keys(f"models.{name}", entry, MODEL_KEYS["mfa"])
        else:
            _check_keys(f"models.{name}", entry, MODEL_KEYS["bfa"], required=())
            if len(entry) != 1:
                raise ValueError(f"evaluate.yaml: models.{name} must set exactly one of "
                                 f"model_name / ckpt_path, got {sorted(entry)}")
            if "ckpt_path" in entry:
                entry["ckpt_path"] = resolve_repo_path(
                    entry["ckpt_path"].replace("{releases_dir}", str(paths["releases_dir"])))

    if not cfg["experiments"]:
        raise ValueError("evaluate.yaml: `experiments:` is empty")
    for n, entry in enumerate(cfg["experiments"]):
        _check_keys(f"experiments[{n}]", entry, EXPERIMENT_KEYS, required=("dataset", "model"))

    cfg["config_path"] = os.path.abspath(path)
    return cfg


def results_csv_path(cfg, dataset):
    """The results CSV of one dataset."""
    return cfg["paths"]["results_csv"].replace("{dataset}", dataset)
