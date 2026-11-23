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
# Decoder settings BFA() has no argument for: they come from the checkpoint's `decoder` record,
# and the sweep overrides them on the built aligner's `decoder_hp`. Optional everywhere (absent
# or `null` = the checkpoint's value), so older yamls and cached run names are unchanged.
HP_OVERRIDES = ("soft_frame_cost", "boundary_contrast_frames", "sil_skippable",
                "optional_sil", "class_prior_norm")
# Passed to `BFA` as they are.
ALIGNER_KWARGS = ("lang", "device", "bad_confidence_threshold", "low_confidence_level",
                  "chunk_audio", "segment_max_seconds")
# Used by the sweep itself.
RUN_SETTINGS = ("duration_max", "batch_size", "coverage_min")
# Everything under `bfa:`, and everything an `experiments:` entry may override or sweep.
BFA_SETTINGS = DECODER_AXES + ALIGNER_KWARGS + RUN_SETTINGS
# Passed to `BFA`, but optional: absent or `null` = False, what every run before v2.0.2 used
# (BFA's own default became True then), so older yamls and run names are unchanged.
OPTIONAL_KWARGS = ("punctuation_sil", "decoder")
# `decoder:` (optional, a mapping) goes to `BFA(decoder=...)`: any decoder setting, e.g. the
# pause / non-speech ones of 2.1 (`{word_sil: false}`). It is not swept: one mapping per entry.

SECTIONS = {
    "paths": ("audio_roots", "releases_dir", "results_csv"),
    "run": ("existing", "vocab_analysis", "warn_level", "torch_threads"),
    "bfa": BFA_SETTINGS + HP_OVERRIDES + OPTIONAL_KWARGS,
    "mfa": ("conda_env", "num_jobs", "extra_args", "keep_workdir"),
    # optional: only a yaml with a `method: maps` model needs it
    "maps": ("conda_env", "repo_dir", "num_jobs", "keep_workdir"),
    # optional: only a yaml with a `method: ctc` model needs it
    "ctc": ("device",),
    # optional: only a yaml with a `method: charsiu` model needs it
    "charsiu": ("device",),
    "scoring": ("tolerance_ms", "align_max_dist", "offset_start", "offset_end",
                "use_end_boundaries", "measure_group_errors", "merge_two_token_phones",
                "limit_clips", "plots"),
}
EXPERIMENT_KEYS = ("name", "enabled", "dataset", "subset", "split", "model") + BFA_SETTINGS \
    + HP_OVERRIDES + OPTIONAL_KWARGS
# `split:` of an experiment. `test` is the clips MAPS's English model was not trained on.
SPLITS = ("all", "test")
OPTIONAL_SECTIONS = ("maps", "ctc", "charsiu")
MODEL_KEYS = {
    "bfa": ("model_name", "ckpt_path"),                    # exactly one of the two
    "mfa": ("method", "acoustic_model", "dictionary", "speaker_folders"),   # all four
    "maps": ("method", "acoustic_model", "dictionary", "g2p_model", "interpolate",
             "trained_on"),                                                  # all six
    "ctc": ("method", "hf_model", "revision", "compounds", "lang"),         # all five
    "charsiu": ("method", "hf_model", "revision", "decoder"),               # all four
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

    _check_keys("the top level", cfg, tuple(SECTIONS) + ("models", "experiments"),
                required=[s for s in SECTIONS if s not in OPTIONAL_SECTIONS]
                + ["models", "experiments"])
    for section, keys in SECTIONS.items():
        if section not in cfg:
            continue
        required = BFA_SETTINGS if section == "bfa" else None
        _check_keys(f"`{section}:`", cfg[section] or {}, keys, required=required)

    paths = cfg["paths"]
    unknown = sorted(set(paths["audio_roots"]) - set(DATASET_SRT_ROOTS))
    if unknown:
        raise KeyError(f"evaluate.yaml: paths.audio_roots names {unknown}, but the datasets "
                       f"are {sorted(DATASET_SRT_ROOTS)}")
    if "{dataset}" not in str(paths["results_csv"]):
        raise ValueError("evaluate.yaml: paths.results_csv needs `{dataset}` in it, e.g. "
                         "local/results/results_{dataset}.csv - each dataset has its own file")
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
        elif entry.get("method") == "maps":
            _check_keys(f"models.{name}", entry, MODEL_KEYS["maps"])
            if "maps" not in cfg:
                raise KeyError(f"evaluate.yaml: models.{name} is a MAPS model, so the yaml "
                               f"needs a `maps:` section")
            entry["trained_on"] = list(entry["trained_on"] or [])
        elif entry.get("method") == "ctc":
            _check_keys(f"models.{name}", entry, MODEL_KEYS["ctc"])
            if "ctc" not in cfg:
                raise KeyError(f"evaluate.yaml: models.{name} is a CTC model, so the yaml "
                               f"needs a `ctc:` section")
            if entry["compounds"] not in ("split", "native"):
                raise ValueError(f"evaluate.yaml: models.{name}: compounds "
                                 f"{entry['compounds']!r} is one of split | native")
        elif entry.get("method") == "charsiu":
            _check_keys(f"models.{name}", entry, MODEL_KEYS["charsiu"])
            if "charsiu" not in cfg:
                raise KeyError(f"evaluate.yaml: models.{name} is a Charsiu model, so the yaml "
                               f"needs a `charsiu:` section")
            if entry["decoder"] not in ("viterbi", "argmax"):
                raise ValueError(f"evaluate.yaml: models.{name}: decoder "
                                 f"{entry['decoder']!r} is one of viterbi | argmax")
        elif "method" in entry:
            raise ValueError(f"evaluate.yaml: models.{name}: method {entry['method']!r} is "
                             f"not one of mfa | maps | ctc | charsiu (a BFA model has no `method`)")
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
        for split in (entry["split"] if isinstance(entry.get("split"), list)
                      else [entry.get("split", "all")]):
            if split not in SPLITS:
                raise ValueError(f"evaluate.yaml: experiments[{n}] split {split!r}; "
                                 f"it is one of {SPLITS}")

    cfg["config_path"] = os.path.abspath(path)
    return cfg


def results_csv_path(cfg, dataset):
    """The results CSV of one dataset."""
    return cfg["paths"]["results_csv"].replace("{dataset}", dataset)
