"""
Parameter sweep: run BFA over a corpus once per configuration, writing one `.bfN.json` of
alignments per audio file plus a manifest describing exactly what produced it.

`compute_metrics.py` reads those manifests back to fill the CSV's `notes` and `cfg_*` columns,
so every scored row is labelled by the code that ran it rather than by memory days later.

Every path it reads or writes is anchored at the repository root by `evaluation/layout.py`, so
the working directory does not matter:

    python evaluation/process_alignments.py
    python evaluation/process_alignments.py 10 3     # one of 10 parallel shells, see below

Alignments land in `tmp/alignments/<dataset>/<bfN>/`, one directory per experiment, so a run
can be thrown away with `rm -rf` without touching the corpus. See `layout.py`.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

# Sweep the LOCAL checkout, not whatever `pip install bournemouth-forced-aligner` last put on
# the path: this script exists to measure the code in this repository. `sys.path[0]` is
# `evaluation/` when the file is run directly, so the repo root has to be prepended explicitly -
# without it every run dies at the first `from bournemouth_aligner import ...`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.layout import (OUTPUT_DIR, MANIFESTS, MANIFEST_DIR, LOG_DIR, PATHS_LISTS,
                            alignment_path, assert_unique_stems, experiment_dir, resolve_srt,
                            vs_tag)


# =====================================================================================
# CONFIGURATION - everything that decides WHAT runs lives in this block and nowhere else
# =====================================================================================


# Where everything goes is defined ONCE, in `layout.py`, and imported by the scorer too - if
# the two ever disagree the sweep writes somewhere the scorer does not read, and every
# extension is reported as "never aligned":
#   OUTPUT_DIR    tmp/alignments/<dataset>/<bfN>/   the .bfN files, and nothing else
#   MANIFESTS     tmp/manifests/                    manifests, logs and notes
# The corpus under tmp/data/ keeps only the SRTs and MFA's own .mfa.json alignments.

# ---- corpus audio roots --------------------------------------------------------------
# The paths lists store audio with the corpus root replaced by `~` - a PLACEHOLDER written by
# `evaluation/dataset/timit_parser.py` (`wav_path.replace(timit_base_dir, '~')`), NOT the home
# directory. It has to be substituted back before the file can be opened; `os.path.expanduser`
# resolves it to $HOME and every single file then fails to load.
DATASET_AUDIO_ROOTS = {

    # timit base dir: expect subdirs `TRAIN` and `TEST`, each with `DR1` ... `DR8` subdirs, each with `SX*`, `SI*`, `SA*` subdirs, each with `*.WAV` and `*.PHN` files
    "timit": "/mnt/intelpa-2/rehman/speech_data_small/timit/raw/TIMIT",

    # buckeye base dir: expect s01, s02, ... s40 subdirs, each with sXXXXa/b/*.wav, *.phones, *.words
    "buckeye": "/mnt/intelpa-2/rehman/speech_data_small/buckeye_corpus",     
}

# ---- the acoustic model -------------------------------------------------------------
# ONE entry per selectable model. Each is either a PUBLISHED name (`model_name`, downloaded by
# BFA from its Hugging Face repository on first use) or a LOCAL path (`ckpt_path`: a release
# directory or a training checkpoint file) - never both, because
# `PhonemeTimestampAligner.load_model` gives `ckpt_path` priority over `model_name` and
# silently ignores the other.
#
# The PHONEME SET is not declared here: the model decides it (p3cupe is ph66, p4mbfa is std),
# and `checkpoint_identity` reads it back for the manifest so every scored row says which set
# it was aligned in. Both families are scored against the same ground-truth boundaries, so
# their rows are directly comparable.
#
# Only the released models are listed (bfa_models/p4mbfa/hf_release.yaml says which training
# run each one is and why). The September sweeps over training-run checkpoints (pa07*, pa08*,
# pa09*, ma02a_e30) are gone from here; their manifests stay in tmp/manifests as the record.
bfa_models = "/mnt/intelpa-2/rehman/codes/bfa_models"
# `export_hf.py release` builds every release here before `push` uploads it.
RELEASES_DIR = f"{bfa_models}/tmp/training/selected_models_for_hf"

PUBLISHED = (
    # ---- p3cupe (ph66 phoneme set)
    "CUPE-3-en",              # pa07f e0: English, LibriSpeech 500 h - the English ph66 preset
    "CUPE-3-en-base",         # pa01d e3: English, the root of the lineage. By name only
    "CUPE-3-mls7",            # pa07a e0: CUPE-3-en-base + 7 MLS languages - every other preset
    # ---- p4mbfa (std phoneme set), one per language group
    "mbfa-latin",             # ma04a e6: the std default, trained at the most noise
    "mbfa-latin-benchmark",   # ma02a e8: best latin model on TIMIT. By name only
    # TIMIT and Buckeye are English, so the three below cannot be scored on them -
    # `_validate_experiment` refuses a model whose language group is not the corpus's.
    "mbfa-cyrillic",          # mr01a e10
    "mbfa-cjk",               # mc01a e13
    "mbfa-japanese",          # mj01a e8
)

MODELS = {
    # ---- as a user gets them: downloaded from https://huggingface.co/Tabahi/<name>
    **{name: {"model_name": name} for name in PUBLISHED},
    # ---- the same releases from the local build, for checking one BEFORE it is pushed.
    # Identical weights once pushed, so sweep one source or the other, not both.
    **{f"{name}-local": {"ckpt_path": f"{RELEASES_DIR}/{name}"} for name in PUBLISHED},
}

# ---- BFA settings held CONSTANT across the grid --------------------------------------
BFA_FIXED = dict(
    lang="en-us",
    device="cpu",                 # CPU on purpose: does not contend with training on the GPUs.
                                  # The Viterbi runs on CPU regardless - it is a python loop
                                  # over frames on small tensors, ~3x faster there than on CUDA.
    bad_confidence_threshold=0.9,
    low_confidence_level=None,    # None = the model family's own level (0.6 CUPE-3, 0.1 mbfa)
    chunk_audio=True,
    segment_max_seconds=120,      # TIMIT/Buckeye segments are far below this, so nothing chunks
)

# ---- the grid ------------------------------------------------------------------------
# The cartesian product of these is the experiment list. Keep single-element lists for the
# axes you are not sweeping - that keeps the shape of the sweep visible.
#
# `None` on a decoder axis means "whatever the checkpoint was trained with", which is BFA's own
# default. It is a legitimate value to sweep, but see `_check_for_duplicates`: on a checkpoint
# trained at bonus 30, the grid [None, 30.0] is two runs of the same configuration under two
# different labels.
EXPERIMENT_GRID = dict(
    dataset=["timit", "buckeye"],         # "timit" | "buckeye"
    # -1 = every clip. Buckeye clips are WHOLE INTERVIEWS - ~7 minutes each, 40 segments - so
    # a handful of them is already a few thousand boundaries; TIMIT clips are ~2.5 s, so it
    # takes ~100 of those to say anything.
    subset=[-1],
    # keys of MODELS: the five that can align English
    model=["CUPE-3-en", "CUPE-3-en-base", "CUPE-3-mls7", "mbfa-latin", "mbfa-latin-benchmark"],
    boundary_bonus_weight=[0.00, None],    # None = the checkpoint's trained value
    min_phone_frames=[None],              # in 5 ms frames; None = the checkpoint's value
    soft_phone_frames=[None],             # None = the checkpoint's value
    sub_frame_onsets=[False,],              # False = onsets on the 5 ms frame grid
)

# First `.bfN.json` suffix to allocate. Each experiment takes the next N, and re-using an N
# that holds a DIFFERENT config deletes that run's alignments first (see `write_manifest`).
# bf1 - bf36 hold the September p3cupe decoder sweeps over training-run checkpoints (Buckeye,
# 10 clips); the released-model suite starts after them so those stay on disk.
VS_COUNTER_START = 37

DURATION_MAX_SEGMENTS = 30
REDO = True                 # re-align even when the vs file already exists
BATCH_SIZE = 16
COVERAGE_MIN = 0.6          # segments below this coverage ratio are dropped from the output

RUN_CLEANUP = False         # DESTRUCTIVE: deletes old .bfN.json files. See clean_up().
RUN_VOCAB_ANALYSIS = False  # one-off; phonemization does not depend on any swept parameter


# The decoder axes above, mapped to the `PhonemeTimestampAligner` kwargs they set. Kept as one
# table so adding an axis is a single edit and cannot drift out of step with `build_aligner`.
DECODER_AXES = ("boundary_bonus_weight", "min_phone_frames", "soft_phone_frames",
                "sub_frame_onsets")


# =====================================================================================
# Process distribution
# =====================================================================================

# `python evaluation/process_alignments.py N K` runs every N-th experiment of the grid,
# starting at the K-th (0-based) - so N shells with K = 0 .. N-1 cover the grid between them,
# one experiment at a time each. The `.bfN.json` numbering comes from the whole grid, so it is
# the same whichever process runs an experiment.
process_id = 0
total_processes = 1
if len(sys.argv) > 2:
    total_processes = int(sys.argv[1])
    process_id = int(sys.argv[2])
    if not 0 <= process_id < total_processes:
        raise SystemExit(f"process id {process_id} is not in 0 .. {total_processes - 1}")
    # Share the cores: parallel sweeps that each start a full-size torch thread pool only
    # contend (the Viterbi is a single python thread per process either way).
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // (2 * total_processes)))


# =====================================================================================
# Logging helpers
# =====================================================================================

def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def append_text(file_path, text):
    ensure_parent_dir(file_path)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def log_message(message, log_file=None):
    print(message)
    if log_file:
        append_text(log_file, message + "\n")


def log_block(log_file, lines):
    if not log_file:
        return
    append_text(log_file, "\n".join(lines) + "\n")


def line():
    print("------------------------------------------------------------------")



# =====================================================================================
# Experiment resolution, validation and manifests
# =====================================================================================

def resolve_model(model_key):
    """
    Turn a MODELS key into the exact `PhonemeTimestampAligner` kwargs, and a label.

    Returns `(kwargs, label)` where kwargs has exactly ONE of `model_name` / `ckpt_path` set,
    so the constructor's priority rule can never make the other one a lie.
    """
    if model_key not in MODELS:
        raise KeyError(f"unknown model {model_key!r}; MODELS has {sorted(MODELS)}")
    entry = dict(MODELS[model_key])
    if ('model_name' in entry) == ('ckpt_path' in entry):
        raise ValueError(f"MODELS[{model_key!r}] must set exactly one of model_name / "
                         f"ckpt_path, got {sorted(entry)}")
    if 'ckpt_path' in entry and not os.path.exists(entry['ckpt_path']):
        raise FileNotFoundError(f"MODELS[{model_key!r}] checkpoint not found: "
                                f"{entry['ckpt_path']}")
    # BFA wants both kwargs present; the unused one must be explicitly None.
    kwargs = {"model_name": entry.get("model_name"), "ckpt_path": entry.get("ckpt_path")}
    return kwargs, model_key


_CKPT_DEFAULTS_CACHE = {}
_CKPT_IDENTITY_CACHE = {}


def checkpoint_decoder_defaults(model_key):
    """
    The decoder settings the model is published or was trained with - i.e. what a `None` on
    any of `DECODER_AXES` actually resolves to.

    Read straight from the model's own record (a release's config.json, a training
    checkpoint's `hyper_parameters`) rather than by constructing an aligner, so validation
    stays cheap enough to run before the sweep starts. A published model is downloaded here if
    it is not on disk yet.
    """
    if model_key not in _CKPT_DEFAULTS_CACHE:
        _read_checkpoint(model_key)
    return _CKPT_DEFAULTS_CACHE[model_key]


def checkpoint_identity(model_key):
    """
    Which model family and phoneme set a MODELS entry is, read off the model: `family`,
    `phoneme_set` ('ph66', or 'std/<lang_group>'), `lang_group`, for a released p4mbfa model
    the `gold_fingerprint` of the token list it was trained on, and for any released model
    the training run it is (`source_experiment`, `source_checkpoint`).
    """
    if model_key not in _CKPT_IDENTITY_CACHE:
        _read_checkpoint(model_key)
    return _CKPT_IDENTITY_CACHE[model_key]


def model_path(model_key):
    """Where a MODELS entry is on disk; a published `model_name` is downloaded if needed."""
    kwargs = resolve_model(model_key)[0]
    if kwargs["ckpt_path"]:
        return kwargs["ckpt_path"]
    from bournemouth_aligner import hub
    return hub.resolve_model(kwargs["model_name"])


def _read_checkpoint(model_key):
    """One read of the model for both `checkpoint_decoder_defaults` and `_identity`."""
    path = model_path(model_key)
    release = {}
    if os.path.isdir(path):                  # a release: everything is in config.json
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            release = json.load(f)
        hp, config = release["decoder"], release["model_config"]
        fingerprint = release.get("labels", {}).get("gold_fingerprint")
    else:                                    # a training checkpoint
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        hp = ckpt.get('hyper_parameters', {}) or {}
        if hasattr(hp, '__dict__') and not isinstance(hp, dict):
            hp = vars(hp)
        config = ckpt.get('model_config', {}) or {}
        fingerprint = None
        del ckpt
    # No fallbacks: BFA refuses a model that does not carry its decoder settings.
    defaults = {
        "boundary_bonus_weight": float(hp["boundary_bonus_weight"]),
        "min_phone_frames": int(hp["min_phone_frames"]),
        "soft_phone_frames": int(hp["soft_phone_frames"]),
    }
    if 'n_tokens' in config:
        group = config.get('lang_group')
        identity = {"family": "p4mbfa", "phoneme_set": f"std/{group}", "lang_group": group,
                    "gold_fingerprint": fingerprint}
    else:
        identity = {"family": "p3cupe", "phoneme_set": "ph66", "lang_group": None,
                    "gold_fingerprint": None}
    training = release.get("training", {})
    identity["source_experiment"] = training.get("experiment")
    identity["source_checkpoint"] = training.get("source_checkpoint")
    _CKPT_DEFAULTS_CACHE[model_key] = defaults
    _CKPT_IDENTITY_CACHE[model_key] = identity


def effective_decoder_config(exp):
    """
    The experiment's decoder settings with every `None` resolved against the checkpoint.

    This is what actually runs, and therefore what two experiments have to differ in for their
    two result rows to mean anything.
    """
    defaults = checkpoint_decoder_defaults(exp["model"])
    out = {}
    for axis in DECODER_AXES:
        value = exp.get(axis)
        out[axis] = defaults.get(axis, value) if value is None else value
    return out


def _validate_experiment(exp):
    """Reject the values that fail loudly later, or quietly never."""
    resolve_model(exp["model"])          # raises on an unknown key or a missing checkpoint

    if exp["dataset"] not in DATASET_AUDIO_ROOTS:
        raise KeyError(f"unknown dataset {exp['dataset']!r}; "
                       f"DATASET_AUDIO_ROOTS has {sorted(DATASET_AUDIO_ROOTS)}")

    # A p4mbfa model belongs to one language group and BFA raises on a language outside it -
    # per file, so without this a mismatched model "runs" and reports every file as failed.
    group = checkpoint_identity(exp["model"])["lang_group"]
    if group is not None:
        from bournemouth_aligner import presets
        corpus_group = presets.get_std_preset(BFA_FIXED["lang"])[2]
        if group != corpus_group:
            raise ValueError(
                f"model {exp['model']!r} is a '{group}' model, but the corpora are "
                f"{BFA_FIXED['lang']} ('{corpus_group}'): it cannot align them. Take it out "
                f"of EXPERIMENT_GRID['model'].")

    bonus = exp["boundary_bonus_weight"]
    if bonus is not None and float(bonus) < 0:
        raise ValueError(f"boundary_bonus_weight={bonus}: a negative weight rewards a phone for "
                         f"starting where the audio does NOT change, which is not a setting "
                         f"anyone means. Use 0 to switch the bonus off.")

    mpf, spf = exp["min_phone_frames"], exp["soft_phone_frames"]
    if mpf is not None and int(mpf) < 1:
        raise ValueError(f"min_phone_frames={mpf}: must be >= 1 (frames are 5 ms).")
    if spf is not None and int(spf) < 0:
        raise ValueError(f"soft_phone_frames={spf}: must be >= 0.")
    return exp


def _check_for_duplicates(experiments):
    """
    Refuse a grid whose rows are not actually different runs.

    The failure this exists for: every decoder axis accepts `None`, meaning "the checkpoint's
    own trained value". So on a checkpoint trained at bonus 30, `boundary_bonus_weight=[None,
    30.0]` is ONE configuration run twice, and the CSV ends up with two rows that differ only
    in their label - which is exactly the shape of a result that gets believed.
    """
    seen = {}
    for exp in experiments:
        key = (exp["dataset"], exp["subset"], exp["model"],
               tuple(sorted(effective_decoder_config(exp).items())))
        if key in seen:
            raise ValueError(
                f"{exp['vs_ext']} and {seen[key]['vs_ext']} resolve to the SAME configuration "
                f"and would produce two identical rows under different labels:\n"
                f"  {seen[key]['vs_ext']}: " + ", ".join(f"{k}={seen[key][k]!r}" for k in DECODER_AXES) + "\n"
                f"  {exp['vs_ext']}: " + ", ".join(f"{k}={exp[k]!r}" for k in DECODER_AXES) + "\n"
                f"  both resolve to: {effective_decoder_config(exp)}\n"
                f"(`None` means the checkpoint's trained value - see checkpoint_decoder_defaults.)")
        seen[key] = exp

    return experiments


def build_experiments():
    """The grid -> a list of fully-resolved experiments, each with its own vs_ext."""
    keys = list(EXPERIMENT_GRID)
    experiments = []
    for counter, values in enumerate(product(*(EXPERIMENT_GRID[k] for k in keys)),
                                     start=VS_COUNTER_START):
        exp = dict(zip(keys, values))
        exp["vs_ext"] = f".bf{counter}.json"
        experiments.append(_validate_experiment(exp))
    return _check_for_duplicates(experiments)


def _fmt(value, resolved):
    """`30.0` -> `30`, and a `None` axis as `30*` so the label says it came from the checkpoint."""
    text = f"{resolved:g}" if isinstance(resolved, (int, float)) and not isinstance(resolved, bool) \
        else str(resolved)
    return text + ("*" if value is None else "")


def experiment_label(exp):
    """
    The short, self-describing name that lands in the CSV's `notes` column.

    `subset` is part of the label whenever it is not the full set, and that is not cosmetic: a
    100-file row reads several f1 points HIGH against a full-corpus row and can even reorder
    settings. Two rows scored on different amounts of data must not be able to sit next to each
    other looking comparable.
    """
    eff = effective_decoder_config(exp)
    label = (f"{exp['model']}"
             f"-b{_fmt(exp['boundary_bonus_weight'], eff['boundary_bonus_weight'])}"
             f"-d{_fmt(exp['min_phone_frames'], eff['min_phone_frames'])}"
             f"+{_fmt(exp['soft_phone_frames'], eff['soft_phone_frames'])}")
    if not eff["sub_frame_onsets"]:
        label += "-gridonset"
    subset = int(exp.get("subset", -1) or -1)
    if subset > 0:
        label += f"-n{subset}"
    return label


def write_manifest(exp, extra=None):
    """
    Record the full configuration that produced this `vs_ext`, at the moment it is produced.

    `compute_metrics.py` reads this back to fill the CSV's `notes` and `cfg_*` columns, so a
    row is labelled by the code that ran it rather than by memory days later.
    """
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = os.path.join(MANIFEST_DIR, f"{vs_tag(exp['vs_ext'])}.json")

    model_kwargs, _ = resolve_model(exp["model"])
    manifest = {
        "vs_ext": exp["vs_ext"],
        "label": experiment_label(exp),
        "timestamp": timestamp_now(),
        "experiment": exp,
        # What the `None` axes actually resolved to. Without this the manifest records the
        # request rather than the run, and a checkpoint change silently reinterprets every
        # historical row.
        "decoder_effective": effective_decoder_config(exp),
        "model": model_kwargs,
        # Family, phoneme set and language group, from the checkpoint itself. The scorer needs
        # the phoneme set to know which id is silence and which groups the phonemes fall in.
        **checkpoint_identity(exp["model"]),
        "bfa_fixed": BFA_FIXED,
        "duration_max_segments": DURATION_MAX_SEGMENTS,
        "coverage_min": COVERAGE_MIN,
        **(extra or {}),
    }

    # A manifest that already exists with a DIFFERENT config means this `.bfN.json` currently
    # holds another experiment's alignments. Re-using the extension is fine - but the STALE
    # files must go first, or this run silently inherits whichever of them it does not
    # overwrite (a different subset, or files that error out this time), and the CSV row ends
    # up scoring a mixture of two configurations.
    clashed = False
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("experiment") != exp:
            clashed = True
            print(f"\n[WARN] {path} already describes a DIFFERENT experiment:"
                  f"\n         existing: {old.get('label')}  {old.get('experiment')}"
                  f"\n         new:      {manifest['label']}  {exp}"
                  f"\n       Re-using {exp['vs_ext']} - its stale alignment files will be "
                  f"deleted first so the two cannot mix.\n")

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path, clashed


def load_manifest(vs_ext, manifest_dir=MANIFEST_DIR):
    """Read back a manifest by `vs_ext`. Returns None when there is none."""
    path = os.path.join(manifest_dir, f"{vs_tag(vs_ext)}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# =====================================================================================
# Paths lists
# =====================================================================================

def resolve_audio_path(dataset, audio_path):
    """
    Put the corpus root back in front of a stored audio path.

    The parsers write `wav_path.replace(corpus_root, '~')`, so `~` here is a placeholder for a
    directory that has nothing to do with $HOME. Left unresolved, every `torchaudio.load` fails
    and the whole sweep reports 100% failed files.
    """
    root = DATASET_AUDIO_ROOTS.get(dataset)
    if root and audio_path.startswith("~"):
        return root + audio_path[1:]
    return audio_path


def get_paths_list(dataset):
    """
    The paths list for one corpus, as `evaluation/dataset/<dataset>_parser.py` wrote it.

    Checked for clip-name collisions up front: alignments are keyed by `layout.clip_stem`, and
    two clips sharing a stem would overwrite each other's output silently - a wrong score
    rather than a missing one.
    """
    if dataset not in PATHS_LISTS:
        raise KeyError(f"unknown dataset {dataset!r}; PATHS_LISTS has {sorted(PATHS_LISTS)}")
    paths_list_path = PATHS_LISTS[dataset]
    print("paths_list:", paths_list_path)
    if not os.path.exists(paths_list_path):
        raise FileNotFoundError(
            f"{os.path.basename(paths_list_path)} not found at {paths_list_path}. "
            f"Run evaluation/dataset/{dataset}_parser.py to generate it.")

    with open(paths_list_path, "r") as f:
        paths_list = json.load(f)

    assert_unique_stems(dataset, paths_list)
    return paths_list



# =====================================================================================
# Task: BFA alignment  (the one that writes the .bfN.json files)
# =====================================================================================

def build_aligner(exp, log_file=None):
    """
    Construct the aligner for one experiment. Every setting comes from either `exp` (swept) or
    `BFA_FIXED` (held constant) - there are no literals here, which is what makes the manifest
    a complete description of the run.
    """
    from bournemouth_aligner import PhonemeTimestampAligner

    model_kwargs, label = resolve_model(exp["model"])
    eff = effective_decoder_config(exp)
    source = (f"ckpt {os.path.basename(model_kwargs['ckpt_path'])}"
              if model_kwargs['ckpt_path'] else f"published {model_kwargs['model_name']}")
    identity = checkpoint_identity(exp["model"])
    log_message(
        f"[{timestamp_now()}] BFA aligner: model={label} ({source}, "
        f"{identity.get('family', '?')} / {identity.get('phoneme_set', '?')}), "
        + ", ".join(f"{axis}={_fmt(exp[axis], eff[axis])}" for axis in DECODER_AXES)
        + f", vs_ext={exp['vs_ext']}   (* = the checkpoint's trained value)",
        log_file,
    )

    aligner = PhonemeTimestampAligner(
        **model_kwargs,
        duration_max=DURATION_MAX_SEGMENTS,
        # A `None` axis is passed through as None, which is BFA's own "use the checkpoint's
        # trained value" - so the sweep never has to hard-code a default that could drift away
        # from what the model was actually trained with.
        **{axis: exp[axis] for axis in DECODER_AXES},
        **BFA_FIXED,
    )
    aligner.warn_level = 1  # 2 = all warnings, 1 = critical only, 0 = none
    return aligner


def task_bfa_align(paths_list, exp, log_file=None):
    """
    Align every file in `paths_list` and write one `<vs_ext>` json per audio file.

    Segments whose `coverage_ratio` falls below COVERAGE_MIN are dropped from the output; a file
    with no surviving segments is not written at all. Both are counted and reported, because
    they change the denominator of every score computed downstream.
    """
    aligner = build_aligner(exp, log_file=log_file)
    vs_ext = exp["vs_ext"]

    latencies = []
    aligned_files = 0
    total_segments = 0
    failed_segments = 0
    failed_files = 0
    bad_alignment_total = 0
    bad_confidence_total = 0
    confidence_sum = 0.0
    confidence_count = 0
    segment_durations = []

    for i in tqdm(range(len(paths_list))):
        audio_path = resolve_audio_path(exp["dataset"], paths_list[i]['audio'])
        srt_path = resolve_srt(paths_list[i]['srt'])
        vs_path = alignment_path(exp["dataset"], srt_path, vs_ext)

        if os.path.exists(vs_path) and not REDO:
            continue
        if os.path.exists(vs_path):
            os.remove(vs_path)  # ensure fresh alignments rather than a partial overwrite

        try:
            start_time = time.time()

            vs2_data = aligner.process_srt_file(
                srt_path, audio_path,
                ts_out_path=None, extract_embeddings=False, vspt_path=None,
                do_groups=False, batch_size=BATCH_SIZE, debug=False,
            )
            if vs2_data is None:
                log_message(f"Error: BFA alignment failed for {srt_path}, skipping.", log_file)
                failed_files += 1
                continue

            # --- drop low-coverage segments -------------------------------------------
            pop_s = []
            for seg_i, seg in enumerate(vs2_data['segments']):
                segment_durations.append(float(seg['end']) - float(seg['start']))
                confidences = [float(phone['confidence'])
                               for phone in seg.get('phoneme_ts', [])
                               if phone.get('confidence') is not None]
                confidence_sum += sum(confidences)
                confidence_count += len(confidences)
                if seg['coverage_analysis']['bad_alignment']:
                    bad_confidence_total += 1
                if seg['coverage_analysis']['coverage_ratio'] < COVERAGE_MIN:
                    pop_s.append(seg_i)
                    failed_segments += 1
                elif seg['coverage_analysis']['bad_alignment']:
                    # counted but KEPT - dropping these was tried and is not obviously right
                    bad_alignment_total += 1
                total_segments += 1

            if pop_s:
                vs2_data['segments'] = [s for j, s in enumerate(vs2_data['segments'])
                                        if j not in pop_s]

            if len(vs2_data['segments']) == 0:
                log_message(f"WARN: all segments bad in {audio_path}, not writing. "
                            f"SRT: {srt_path}", log_file)
                failed_files += 1
                continue

            ensure_parent_dir(vs_path)
            with open(vs_path, 'w') as f:
                json.dump(vs2_data, f, ensure_ascii=False)

            aligned_files += 1
            latencies.append(time.time() - start_time)

        except ValueError as ex:
            log_message(f"ValueError during BFA alignment for {audio_path} {srt_path}", log_file)
            log_message(str(ex), log_file)
            failed_files += 1
        except Exception as ex:
            log_message(f"ERROR during BFA alignment for {audio_path} {srt_path}", log_file)
            log_message(str(ex), log_file)
            traceback.print_exc()
            failed_files += 1

    # --- summary -------------------------------------------------------------------------
    log_message(f"Done. Files written: {aligned_files}, Total segments: {total_segments}, "
                f"Failed segments: {failed_segments}, Failed files: {failed_files}, "
                f"Flagged bad_alignment (kept): {bad_alignment_total}", log_file)
    log_message(
        f"Totals, segments processed: {aligner.total_segments_processed}\t"
        f"Segments flagged bad: {aligner.total_segments_bad}\t"
        f"Segments with no output: {aligner.total_segments_failed}\t"
        f"Phonemes aligned: {aligner.total_phonemes_aligned}\t"
        f"Target phonemes: {aligner.total_phonemes_target}\t"
        f"Phonemes missed: {aligner.total_phonemes_missed}", log_file)

    if total_segments > 0 and aligner.total_phonemes_target > 0:
        tgt = aligner.total_phonemes_target
        # `Phonemes missed` is dominated by optional silences the speaker did not pause at:
        # forced alignment places every other target, so a few percent here is the phonemizer
        # writing a SIL per comma, not the aligner losing phonemes.
        log_message(
            f"Percentages, segments failed: {failed_segments / total_segments * 100:.2f}%\t"
            f"Phonemes aligned: {aligner.total_phonemes_aligned / tgt * 100:.2f}%\t"
            f"Target phonemes: 100%\t"
            f"Phonemes missed: {aligner.total_phonemes_missed / tgt * 100:.2f}%", log_file)
    else:
        log_message("Percentages unavailable because no segments or targets were processed.",
                    log_file)

    if latencies:
        log_message(f"Average BFA alignment time per file: {np.mean(latencies) * 1000:.2f} ms "
                    f"over {len(latencies)} files", log_file)
    log_message(f"[{timestamp_now()}] Finished BFA alignment: vs_ext={vs_ext}", log_file)

    return {
        "files_written": aligned_files,
        "total_segments": total_segments,
        "failed_segments": failed_segments,
        "failed_files": failed_files,
        "bad_alignment_kept": bad_alignment_total,
        "mean_confidence": (confidence_sum / confidence_count) if confidence_count else None,
        "bad_confidence_segments": bad_confidence_total,
        "bad_confidence_segment_fraction": (bad_confidence_total / total_segments
                             if total_segments else None),
        "segment_duration_avg_s": (float(np.mean(segment_durations))
                        if segment_durations else None),
        "segment_duration_min_s": (float(np.min(segment_durations))
                        if segment_durations else None),
        "segment_duration_max_s": (float(np.max(segment_durations))
                        if segment_durations else None),
        "phonemes_aligned": aligner.total_phonemes_aligned,
        "phonemes_target": aligner.total_phonemes_target,
        "phonemes_missed": aligner.total_phonemes_missed,
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else None,
    }


# =====================================================================================
# Task: phoneme dictionary over the alignments just written
# =====================================================================================

def make_bfa_phoneme_dict(paths_list, dataset, vs_ext=".bf1.json", log_file=None):
    not_found = 0
    missing_ts = 0
    total_ts = 0
    total_duration = 0.0
    phoneme_dict = {}

    for i in tqdm(range(len(paths_list))):
        vs_path = alignment_path(dataset, paths_list[i]['srt'], vs_ext)
        if not os.path.exists(vs_path):
            not_found += 1
            continue

        with open(vs_path, "r") as f:
            vs_data = json.load(f)

        for segment in vs_data['segments']:
            total_duration += segment['end'] - segment['start']
            for pi, phoneme_ts in enumerate(segment['phoneme_ts']):
                total_ts += 1
                if 'ipa_label' not in phoneme_ts:
                    missing_ts += 1
                    if 'phoneme_id' in phoneme_ts:
                        print("Warning: 'ipa_label' missing, removing unusable vs:", vs_path)
                        os.remove(vs_path)
                        break
                    continue

                ipa_label = phoneme_ts['ipa_label']
                if ipa_label == "-":
                    continue

                if ipa_label not in phoneme_dict:
                    phoneme_dict[ipa_label] = {"count": 0, "total_duration": 0.0,
                                               "avg_confidences": 0.0, "avg_duration": 0.0,
                                               "index": -1}
                phoneme_dict[ipa_label]["count"] += 1
                phoneme_dict[ipa_label]["total_duration"] += phoneme_ts['end_ms'] - phoneme_ts['start_ms']
                phoneme_dict[ipa_label]["avg_confidences"] += phoneme_ts['confidence']

                # a following "-" (pause) is charged to the phoneme before it
                nxt = segment['phoneme_ts'][pi + 1] if pi + 1 < len(segment['phoneme_ts']) else None
                if nxt is not None and nxt.get('ipa_label') == "-":
                    phoneme_dict[ipa_label]["total_duration"] += nxt['end_ms'] - nxt['start_ms']

    total_phonemes_duration = sum(v["total_duration"] for v in phoneme_dict.values())
    log_message(f"Phoneme dict summary. Files not found: {not_found}, Missing 'ipa_label': "
                f"{missing_ts}, Total phoneme_ts: {total_ts}, Total duration: "
                f"{total_duration / 60:.2f} minutes, Total phonemes duration: "
                f"{total_phonemes_duration / 60000:.2f} minutes", log_file)

    phoneme_dict = dict(sorted(phoneme_dict.items(), key=lambda kv: kv[1]['count'], reverse=True))
    if "SIL" in phoneme_dict:
        phoneme_dict = {"SIL": phoneme_dict.pop("SIL"), **phoneme_dict}

    for idx, phoneme in enumerate(phoneme_dict):
        stats = phoneme_dict[phoneme]
        stats["index"] = idx
        stats["avg_confidences"] = (stats["avg_confidences"] / stats["count"]) if stats["count"] else 0.0
        if stats["total_duration"] > 0:
            stats["total_duration"] /= 1000  # ms -> s
        stats["avg_duration"] = ((stats["total_duration"] * 1000) / stats["count"]) if stats["count"] else 0.0

    log_block(log_file, ["Phoneme dictionary details:"] + [
        f"  {p}: count={s['count']}, avg_confidence={s['avg_confidences']:.4f}, "
        f"avg_duration_ms={s['avg_duration']:.2f}, total_duration_s={s['total_duration']:.2f}, "
        f"index={s['index']}"
        for p, s in phoneme_dict.items()])

    return phoneme_dict


# =====================================================================================
# One-off utilities - NOT part of the experiment loop
# =====================================================================================

def analyze_phoneme_vocab_mappings(paths_list, log_file=None):
    '''
    Report the phoneme vocab and counts for `paths_list`. Independent of every swept parameter
    (phonemization does not depend on the acoustic model), so run it once, not per experiment.
    '''
    from bournemouth_aligner import PhonemeTimestampAligner

    # phonemizer only - no model needed. This analysis is about espeak's vocabulary and how
    # ph66 reduces it, so it is pinned to that set whatever the default is.
    bfa = PhonemeTimestampAligner(preset=None, model_name=None, ckpt_path=None,
                                  lang="en-us", phoneme_set="ph66")
    P = bfa.phonemizer
    lang = 'en'
    P.set_backend(lang)

    for i in tqdm(range(len(paths_list)), leave=False):
        if paths_list[i]['lang'] != lang:
            continue
        srt_path = resolve_srt(paths_list[i]['srt'])
        if not os.path.exists(srt_path):
            print("SRT file does not exist:", srt_path)
            continue

        with open(srt_path, "r") as file:
            srt_data = json.load(file)
        if "segments" not in srt_data:
            raise ValueError("SRT data does not contain 'segments' key")
        if not srt_data["segments"]:
            print("SRT file is empty or has no segments:", srt_path)
            continue

        for s, segment in enumerate(srt_data["segments"]):
            if not all(k in segment for k in ("text", "start", "end")):
                print("Segment missing required keys:", srt_path, "index:", s)
                continue
            P.phonemize_sentence(segment["text"])

    by_count = lambda d: dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))
    phoneme_vocab_counts = by_count(P.phoneme_vocab_counts)
    phoneme_mapped_vocab_counts = by_count(P.phoneme_vocab_mapped_counts)
    unmapped_phonemes = by_count(P.unmapped_phonemes)

    log_message("Phoneme vocabulary analysis:", log_file)
    log_message(f"  Total words: {P.word_counts}", log_file)
    log_message(f"  Total phonemes: {P.phn_counts}", log_file)
    log_message(f"  Total unique espeak phonemes: {len(P.phoneme_vocab_counts)}", log_file)
    log_message(f"  Total unique mapped phonemes: {len(P.phoneme_vocab_mapped_counts)}", log_file)
    log_message(f"  Total unmapped phonemes: {len(P.unmapped_phonemes)}", log_file)
    log_message(f"  Silence tokens: {P.sil_count}", log_file)
    log_message(f"  Noise tokens: {P.noise_count}", log_file)
    for title, d in (("Espeak vocab counts", phoneme_vocab_counts),
                     ("Mapped vocab counts", phoneme_mapped_vocab_counts),
                     ("Unmapped phonemes", unmapped_phonemes)):
        log_message(f"  {title}:", log_file)
        log_block(log_file, [str(d)])

    return {"phoneme_vocab_counts": phoneme_vocab_counts,
            "phoneme_mapped_vocab_counts": phoneme_mapped_vocab_counts,
            "unmapped_phonemes": unmapped_phonemes}




def analyze_durations(paths_list):
    sum_dur = 0.0
    sum_n = 0
    for i in tqdm(range(len(paths_list)), leave=False):
        duration = paths_list[i]['duration']
        if duration > 0:
            sum_n += 1
            sum_dur += duration

    print("Total duration:", sum_dur / 3600, "hours")
    print("Average duration:", sum_dur / sum_n, "seconds over", sum_n, "files")


def clean_up_extension(dataset, vs_ext, log_file=None):
    """
    Delete one experiment's alignments.

    Now a single `rm -rf` of `tmp/alignments/<dataset>/<bfN>/`, which is the reason the files
    were moved out of the corpus tree. The old per-entry loop could only remove what the
    CURRENT paths list mentions, so a previous run over a different list - a larger subset, a
    regenerated list - left files behind that the scorer then read as part of this experiment.

    Used when an experiment re-uses an extension a different configuration already wrote to:
    `REDO` only overwrites the files this run reaches, and anything it skips or fails on would
    otherwise survive from the previous config.
    """
    import shutil
    target = experiment_dir(dataset, vs_ext)
    removed = 0
    if os.path.isdir(target):
        removed = sum(len(files) for _, _, files in os.walk(target))
        shutil.rmtree(target)
    log_message(f"Removed {removed} stale file(s) from {target} before re-using {vs_ext}.",
                log_file)
    return removed


def clean_up(dataset, vs_range=range(0, 7), keep=()):
    """
    DESTRUCTIVE: delete old `.bfN.json` alignment directories.

    Gated behind RUN_CLEANUP because it removes the inputs of every CSV row produced from those
    extensions - a scored result cannot be re-checked once its alignments are gone. The
    manifests survive, so the record of what was run does too.
    """
    extensions = [f".bf{n}.json" for n in vs_range if f".bf{n}.json" not in keep]
    print(f"Cleaning up {len(extensions)} old vs extensions for {dataset}: {extensions}")

    removed = 0
    for ext in extensions:
        removed += clean_up_extension(dataset, ext)
    print(f"Removed {removed} files")
    global RUN_CLEANUP
    RUN_CLEANUP = False  # only run once per session


# =====================================================================================
# Experiment runner
# =====================================================================================

def run_experiment(exp, e, n_total, paths_list_cache):
    notes_file = os.path.join(MANIFESTS, f"notes_{exp['dataset']}.txt")
    output_log_file = os.path.join(
        LOG_DIR, f"log_{exp['dataset']}_{exp['model']}"
                 f"_{experiment_label(exp).replace('*', 'c')}_{vs_tag(exp['vs_ext'])}.txt")

    manifest_path, clashed = write_manifest(exp)

    header_sep = "=" * 78
    header_lines = [header_sep, f"[{timestamp_now()}] Experiment {e}/{n_total}"] + [
        f"{k}={v}" for k, v in exp.items()] + [
        f"decoder_effective={effective_decoder_config(exp)}",
        f"label={experiment_label(exp)}",
        f"alignments={experiment_dir(exp['dataset'], exp['vs_ext'])}",
        f"manifest={manifest_path}",
        f"notes_file={notes_file}",
        f"output_log_file={output_log_file}",
        header_sep,
    ]
    log_block(notes_file, header_lines)
    log_block(output_log_file, header_lines)
    print("\n\n" + "\n".join(header_lines) + "\n\n")

    if exp["dataset"] not in paths_list_cache:
        paths_list_cache[exp["dataset"]] = get_paths_list(exp["dataset"])
    full_paths_list = paths_list_cache[exp["dataset"]]

    # Stale files are cleared against the FULL list, not the subset. A previous run of this
    # extension may have covered files this one will not touch, and the scorer walks the whole
    # paths list - so anything left behind outside the subset would be scored as if it belonged
    # to this configuration.
    if clashed:
        clean_up_extension(exp["dataset"], exp["vs_ext"], log_file=output_log_file)

    # Deterministic random subset for quick turnaround. Seeded, so align and score agree, and
    # `subset` is in the label because a 100-file row is not comparable to a full one.
    paths_list = full_paths_list
    if int(exp.get("subset", -1) or -1) > 0:
        import random
        rng = random.Random(42)          # local RNG: does not disturb any global random state
        paths_list = rng.sample(full_paths_list, min(exp["subset"], len(full_paths_list)))
        log_message(f"Using a deterministic random subset of {len(paths_list)} of "
                    f"{len(full_paths_list)} files.", output_log_file)

    if RUN_CLEANUP:
        clean_up(exp["dataset"], keep=(exp["vs_ext"],))
    if RUN_VOCAB_ANALYSIS:
        analyze_phoneme_vocab_mappings(paths_list, log_file=output_log_file)

    stats = task_bfa_align(paths_list, exp, log_file=output_log_file)
    make_bfa_phoneme_dict(paths_list, exp["dataset"], vs_ext=exp["vs_ext"],
                          log_file=output_log_file)

    # re-write the manifest now that the run stats exist, so it describes the OUTCOME too.
    # `files_selected` is what the scorer should limit itself to - see compute_metrics.py.
    write_manifest(exp, extra={"stats": stats,
                               "processes": total_processes,
                               "files_available": len(full_paths_list),
                               "files_selected": len(paths_list)})

    for f in (notes_file, output_log_file):
        log_message(f"[{timestamp_now()}] Finished experiment {e}/{n_total} "
                    f"({experiment_label(exp)})", f)


if __name__ == "__main__":
    all_experiments = build_experiments()
    experiments = all_experiments[process_id::total_processes]

    print(f"{len(experiments)} experiment(s) to run"
          + (f" (process {process_id} of {total_processes}; the grid has "
             f"{len(all_experiments)})" if total_processes > 1 else "") + ":")
    for e, exp in enumerate(experiments, start=1):
        print(f"  {e}. {exp['vs_ext']:<12} {experiment_label(exp):<34} {exp}")
    line()

    paths_list_cache = {}
    for e, exp in enumerate(experiments, start=1):
        run_experiment(exp, e, len(experiments), paths_list_cache)

    line()
    print(f"Alignments:  {OUTPUT_DIR}/<dataset>/<bfN>/   (delete a directory to drop a run)")
    print(f"Manifests:   {MANIFEST_DIR}")
    print("Score these with compute_metrics.py; the CSV `notes` and `cfg_*` columns are filled "
          "from those manifests.")
