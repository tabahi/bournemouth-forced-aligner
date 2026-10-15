"""
Run every experiment of `evaluation/evaluate.yaml`: BFA models and MFA alike, once each.

    python evaluation/process_alignments.py                  # everything in the yaml
    python evaluation/process_alignments.py --list           # what would run, and what is cached
    python evaluation/process_alignments.py --only 'timit__*'
    python evaluation/process_alignments.py 10 3             # one of 10 parallel shells

Each experiment is named by what it is - `<dataset>__<model>__<key settings>` - and writes one
`<clip>.json` per audio file into `tmp/alignments/<that name>/`, with a `_manifest.json`
describing exactly what produced them. The name is the cache key: an experiment that is
already there is skipped (`run.existing` in the yaml), so the yaml can keep growing and only
the new rows cost anything. `compute_metrics.py` scores whatever is on disk.

Nothing is configured in this file. See `evaluate.yaml`, and `README.md` beside it.
"""

import argparse
import fnmatch
import json
import os
import shutil
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

from evaluation.config import ALIGNER_KWARGS, BFA_SETTINGS, DECODER_AXES, load_config
from evaluation.layout import (OUTPUT_DIR, PATHS_LISTS, alignment_path, assert_unique_stems,
                               experiment_dir, log_path, manifest_path, resolve_srt)

# `evaluate.yaml`, parsed. Set by `configure`; everything below reads it.
CFG = {}


def configure(config_path=None):
    """Load the yaml. Call once before anything else here."""
    CFG.clear()
    CFG.update(load_config(config_path))
    return CFG


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
# Models
# =====================================================================================

def is_mfa(model_key):
    return CFG["models"][model_key].get("method") == "mfa"


def resolve_model(model_key):
    """
    Turn a `models:` key into the exact `BFA` kwargs, and a label.

    Returns `(kwargs, label)` where kwargs has exactly ONE of `model_name` / `ckpt_path` set,
    because `load_model` gives `ckpt_path` priority and silently ignores the other.
    """
    models = CFG["models"]
    if model_key not in models:
        raise KeyError(f"unknown model {model_key!r}; `models:` has {sorted(models)}")
    entry = models[model_key]
    if 'ckpt_path' in entry and not os.path.exists(entry['ckpt_path']):
        raise FileNotFoundError(f"models.{model_key}: checkpoint not found: "
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
    Which model family and phoneme set a `models:` entry is, read off the model: `family`,
    `phoneme_set` ('ph66', or 'std/<lang_group>'), `lang_group`, for a released p4mbfa model
    the `gold_fingerprint` of the token list it was trained on, and for any released model
    the training run it is (`source_experiment`, `source_checkpoint`).
    """
    if model_key not in _CKPT_IDENTITY_CACHE:
        _read_checkpoint(model_key)
    return _CKPT_IDENTITY_CACHE[model_key]


def model_path(model_key):
    """Where a `models:` entry is on disk; a published `model_name` is downloaded if needed."""
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


# =====================================================================================
# Experiments: the yaml's lists -> fully-resolved runs, each with its own name
# =====================================================================================

def _validate_experiment(exp):
    """Reject the values that fail loudly later, or quietly never."""
    if exp["dataset"] not in CFG["paths"]["audio_roots"]:
        raise KeyError(f"unknown dataset {exp['dataset']!r}; paths.audio_roots has "
                       f"{sorted(CFG['paths']['audio_roots'])}")
    if exp["model"] not in CFG["models"]:
        raise KeyError(f"unknown model {exp['model']!r}; `models:` has {sorted(CFG['models'])}")
    if exp["method"] == "MFA":
        return exp
    resolve_model(exp["model"])          # raises on a missing checkpoint

    # A p4mbfa model belongs to one language group and BFA raises on a language outside it -
    # per file, so without this a mismatched model "runs" and reports every file as failed.
    group = checkpoint_identity(exp["model"])["lang_group"]
    if group is not None:
        from bournemouth_aligner import presets
        corpus_group = presets.get_std_preset(exp["lang"])[2]
        if group != corpus_group:
            raise ValueError(
                f"model {exp['model']!r} is a '{group}' model, but lang is {exp['lang']} "
                f"('{corpus_group}'): it cannot align it. Take it out of that experiment.")

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


# Settings that are in an experiment's name only when they are NOT at these values, so the
# name stays short for the usual run and still tells two unusual ones apart. The values are a
# naming convention and nothing else - what runs is whatever the yaml says.
ID_EXTRAS = (
    # setting,                  tag,     left out of the name at
    ("duration_max",             "dur",   30),
    ("batch_size",               "bs",    16),
    ("coverage_min",             "cov",   0.6),
    ("lang",                     "lang",  "en-us"),
    ("bad_confidence_threshold", "bct",   0.6),
    ("low_confidence_level",     "lcl",   None),
    ("chunk_audio",              "chunk", True),
    ("segment_max_seconds",      "seg",   120),
)


def _tag_value(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def experiment_id(exp):
    """
    The experiment's name, which is also its directory under `tmp/alignments/`.

        timit__CUPE-3-en__b30_d6_s10_grid            dataset, model, then the decoder as it RAN
        buckeye__mbfa-latin__b0_d6_s10_grid_n10      ... on a 10-clip subset
        timit__mfa-english_us_arpa                   MFA has no settings of ours

    The decoder values are the resolved ones: a `null` in the yaml and the model's own value
    written out are the same run, so they get the same name and are aligned once.
    """
    parts = []
    if exp["method"] == "BFA":
        eff = effective_decoder_config(exp)
        parts += [f"b{eff['boundary_bonus_weight']:g}", f"d{eff['min_phone_frames']}",
                  f"s{eff['soft_phone_frames']}",
                  "subframe" if eff["sub_frame_onsets"] else "grid"]
        parts += [f"{tag}{_tag_value(exp[key])}" for key, tag, usual in ID_EXTRAS
                  if exp[key] != usual]
    if int(exp["subset"]) > 0:
        parts.append(f"n{int(exp['subset'])}")
    name = "__".join([exp["dataset"], exp["model"]] + (["_".join(parts)] if parts else []))
    if os.sep in name:
        raise ValueError(f"experiment name {name!r} is not a directory name")
    return name


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def build_experiments():
    """
    `experiments:` -> the list of runs.

    Each entry is a grid: every key may be one value or a list, and the entry stands for the
    cartesian product. A setting an entry does not mention comes from `bfa:`. The entries are
    simply concatenated, and a run that two of them describe is kept once.
    """
    experiments, seen = [], set()
    for n, entry in enumerate(CFG["experiments"]):
        if not entry.get("enabled", True):
            continue
        group = entry.get("name", f"experiments[{n}]")
        settings = {key: _as_list(entry.get(key, CFG["bfa"][key])) for key in BFA_SETTINGS}
        for dataset, subset, model in product(_as_list(entry["dataset"]),
                                              _as_list(entry.get("subset", -1)),
                                              _as_list(entry["model"])):
            base = {"dataset": dataset, "subset": int(subset), "model": model}
            if model in CFG["models"] and is_mfa(model):
                # None of the BFA settings reach MFA, so the whole grid is one run of it.
                grid = [{"method": "MFA",
                         "acoustic_model": CFG["models"][model]["acoustic_model"],
                         "dictionary": CFG["models"][model]["dictionary"]}]
            else:
                grid = [{"method": "BFA", **dict(zip(settings, values))}
                        for values in product(*settings.values())]
            for point in grid:
                exp = _validate_experiment({**base, **point})
                exp["id"] = experiment_id(exp)
                exp["group"] = group
                if exp["id"] not in seen:
                    seen.add(exp["id"])
                    experiments.append(exp)
    return experiments


def _fmt(value, resolved):
    """`30.0` -> `30`, and a `null` axis as `30*` so the label says it came from the checkpoint."""
    text = f"{resolved:g}" if isinstance(resolved, (int, float)) and not isinstance(resolved, bool) \
        else str(resolved)
    return text + ("*" if value is None else "")


def experiment_label(exp):
    """
    The short name that lands in the CSV's `notes` column. Unlike the experiment's id it
    marks with `*` the decoder values that were left to the checkpoint.
    """
    if exp["method"] == "MFA":
        label = f"{exp['model']}"
    else:
        eff = effective_decoder_config(exp)
        label = (f"{exp['model']}"
                 f"-b{_fmt(exp['boundary_bonus_weight'], eff['boundary_bonus_weight'])}"
                 f"-d{_fmt(exp['min_phone_frames'], eff['min_phone_frames'])}"
                 f"+{_fmt(exp['soft_phone_frames'], eff['soft_phone_frames'])}")
        if not eff["sub_frame_onsets"]:
            label += "-gridonset"
    if exp["subset"] > 0:
        label += f"-n{exp['subset']}"
    return label


def write_manifest(exp, status, extra=None):
    """
    Record the full configuration that produced this experiment, inside its directory.

    Written twice: as `running` before the first file, and as `done` with the run's statistics
    after the last. `compute_metrics.py` only scores a `done` one, and fills the CSV's `notes`
    and `cfg_*` columns from it, so a row is labelled by the code that ran it.
    """
    path = manifest_path(exp["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    manifest = {
        "experiment": exp["id"],
        "method": exp["method"],
        "dataset": exp["dataset"],
        "status": status,
        "label": experiment_label(exp),
        "timestamp": timestamp_now(),
        "config": exp,
    }
    if exp["method"] == "BFA":
        manifest.update({
            # What the `null` axes actually resolved to. Without this the manifest records the
            # request rather than the run, and a checkpoint change silently reinterprets every
            # historical row.
            "decoder_effective": effective_decoder_config(exp),
            "model": resolve_model(exp["model"])[0],
            # Family, phoneme set and language group, from the checkpoint itself.
            **checkpoint_identity(exp["model"]),
        })
    else:
        manifest["model"] = {"acoustic_model": exp["acoustic_model"],
                             "dictionary": exp["dictionary"]}
    manifest.update(extra or {})

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def load_manifest(experiment):
    """One experiment's manifest, or None when there is none."""
    path = manifest_path(experiment)
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
    root = CFG["paths"]["audio_roots"].get(dataset)
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



def select_clips(full_paths_list, subset):
    """
    The clips of one run: all of them, or a seeded random `subset`. Seeded so every model is
    given the same clips, and the subset size is in the experiment's name because a 100-file
    score is not comparable to a full one.
    """
    if subset <= 0:
        return full_paths_list
    import random
    rng = random.Random(42)          # local RNG: does not disturb any global random state
    return rng.sample(full_paths_list, min(subset, len(full_paths_list)))


# =====================================================================================
# Task: BFA alignment
# =====================================================================================

def build_aligner(exp, log_file=None):
    """
    Construct the aligner for one experiment. Every setting comes from `exp` - there are no
    literals here, which is what makes the manifest a complete description of the run.
    """
    from bournemouth_aligner import BFA

    model_kwargs, label = resolve_model(exp["model"])
    eff = effective_decoder_config(exp)
    source = (f"ckpt {os.path.basename(model_kwargs['ckpt_path'])}"
              if model_kwargs['ckpt_path'] else f"published {model_kwargs['model_name']}")
    identity = checkpoint_identity(exp["model"])
    log_message(
        f"[{timestamp_now()}] BFA aligner: model={label} ({source}, "
        f"{identity.get('family', '?')} / {identity.get('phoneme_set', '?')}), "
        + ", ".join(f"{axis}={_fmt(exp[axis], eff[axis])}" for axis in DECODER_AXES)
        + "   (* = the checkpoint's trained value)",
        log_file,
    )

    aligner = BFA(
        **model_kwargs,
        duration_max=exp["duration_max"],
        # A `None` axis is passed through as None, which is BFA's own "use the checkpoint's
        # trained value" - so the sweep never has to hard-code a default that could drift away
        # from what the model was actually trained with.
        **{axis: exp[axis] for axis in DECODER_AXES},
        **{key: exp[key] for key in ALIGNER_KWARGS},
    )
    aligner.warn_level = CFG["run"]["warn_level"]
    return aligner


def task_bfa_align(paths_list, exp, log_file=None):
    """
    Align every file in `paths_list` and write one json per audio file. A file that is
    already there is left alone, which is what resuming an interrupted run means.

    Segments whose `coverage_ratio` falls below `coverage_min` are dropped from the output; a file
    with no surviving segments is not written at all. Both are counted and reported, because
    they change the denominator of every score computed downstream.
    """
    aligner = build_aligner(exp, log_file=log_file)
    latencies = []
    aligned_files = 0
    total_segments = 0
    failed_segments = 0
    failed_files = 0
    bad_alignment_total = 0

    for i in tqdm(range(len(paths_list))):
        audio_path = resolve_audio_path(exp["dataset"], paths_list[i]['audio'])
        srt_path = resolve_srt(paths_list[i]['srt'])
        vs_path = alignment_path(exp["id"], exp["dataset"], srt_path)

        if os.path.exists(vs_path):
            continue

        try:
            start_time = time.time()

            vs2_data = aligner.process_srt_file(
                srt_path, audio_path,
                ts_out_path=None, extract_embeddings=False, vspt_path=None,
                do_groups=False, batch_size=exp["batch_size"], debug=False,
            )
            if vs2_data is None:
                log_message(f"Error: BFA alignment failed for {srt_path}, skipping.", log_file)
                failed_files += 1
                continue

            # --- drop low-coverage segments -------------------------------------------
            pop_s = []
            for seg_i, seg in enumerate(vs2_data['segments']):
                if seg['coverage_analysis']['coverage_ratio'] < exp["coverage_min"]:
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
    log_message(f"[{timestamp_now()}] Finished BFA alignment: {exp['id']}", log_file)

    return {
        "files_aligned_now": aligned_files,
        "segments_aligned_now": total_segments,
        "failed_segments": failed_segments,
        "failed_files": failed_files,
        "bad_alignment_kept": bad_alignment_total,
        "phonemes_aligned": aligner.total_phonemes_aligned,
        "phonemes_target": aligner.total_phonemes_target,
        "phonemes_missed": aligner.total_phonemes_missed,
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else None,
    }


def collect_run_stats(paths_list, exp):
    """
    What the alignment files of a run say about it, read back from disk.

    From disk and not counted while aligning, so a run that was interrupted and resumed is
    described whole rather than by its last sitting.
    """
    files = segments = bad = 0
    confidence_sum, confidence_count, durations = 0.0, 0, []
    for entry in paths_list:
        path = alignment_path(exp["id"], exp["dataset"], entry["srt"])
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        files += 1
        for seg in data["segments"]:
            segments += 1
            durations.append(float(seg["end"]) - float(seg["start"]))
            bad += bool(seg.get("coverage_analysis", {}).get("bad_alignment"))
            for phone in seg.get("phoneme_ts", []):
                if phone.get("confidence") is not None:
                    confidence_sum += float(phone["confidence"])
                    confidence_count += 1
    is_bfa = exp["method"] == "BFA"
    return {
        "files_written": files,
        "total_segments": segments,
        # MFA reports no confidence, so these stay empty for it.
        "mean_confidence": (confidence_sum / confidence_count) if confidence_count else None,
        "bad_confidence_segments": bad if is_bfa else None,
        "bad_confidence_segment_fraction": (bad / segments) if is_bfa and segments else None,
        "segment_duration_avg_s": float(np.mean(durations)) if durations else None,
        "segment_duration_min_s": float(np.min(durations)) if durations else None,
        "segment_duration_max_s": float(np.max(durations)) if durations else None,
    }


# =====================================================================================
# Task: phoneme dictionary over the alignments just written
# =====================================================================================

def make_bfa_phoneme_dict(paths_list, exp, log_file=None):
    not_found = 0
    missing_ts = 0
    total_ts = 0
    total_duration = 0.0
    phoneme_dict = {}

    for i in tqdm(range(len(paths_list))):
        vs_path = alignment_path(exp["id"], exp["dataset"], paths_list[i]['srt'])
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
    from bournemouth_aligner import BFA

    # phonemizer only - no model needed. This analysis is about espeak's vocabulary and how
    # ph66 reduces it, so it is pinned to that set whatever the default is.
    bfa = BFA(preset=None, model_name=None, ckpt_path=None,
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


# =====================================================================================
# Experiment runner
# =====================================================================================

def experiment_state(exp):
    """`done`, `partial` (a directory, not finished) or `new`, and the files it holds."""
    run_dir = experiment_dir(exp["id"])
    if not os.path.isdir(run_dir):
        return "new", 0
    files = sum(1 for name in os.listdir(run_dir)
                if name.endswith(".json") and not name.startswith("_"))
    manifest = load_manifest(exp["id"])
    return ("done" if manifest and manifest.get("status") == "done" else "partial"), files


def run_experiment(exp, e, n_total, paths_list_cache):
    """Align one experiment. Returns `skipped` or `done`."""
    run_dir = experiment_dir(exp["id"])
    state, files_present = experiment_state(exp)
    if CFG["run"]["existing"] == "skip" and state == "done":
        print(f"[{e}/{n_total}] {exp['id']}: already aligned ({files_present} files), skipping.")
        return "skipped"
    if CFG["run"]["existing"] == "redo" and state != "new":
        print(f"[{e}/{n_total}] {exp['id']}: run.existing is 'redo', deleting {run_dir} "
              f"({files_present} files).")
        shutil.rmtree(run_dir)
        files_present = 0

    output_log_file = log_path(exp["id"])
    write_manifest(exp, "running")

    header_sep = "=" * 78
    header_lines = [header_sep, f"[{timestamp_now()}] Experiment {e}/{n_total}"] + [
        f"{k}={v}" for k, v in exp.items()] + [
        f"label={experiment_label(exp)}",
        f"alignments={run_dir}",
        header_sep,
    ]
    log_block(output_log_file, header_lines)
    print("\n\n" + "\n".join(header_lines) + "\n\n")
    if files_present:
        log_message(f"Resuming: {files_present} file(s) are already there and are kept.",
                    output_log_file)

    if exp["dataset"] not in paths_list_cache:
        paths_list_cache[exp["dataset"]] = get_paths_list(exp["dataset"])
    full_paths_list = paths_list_cache[exp["dataset"]]
    paths_list = select_clips(full_paths_list, exp["subset"])
    if len(paths_list) < len(full_paths_list):
        log_message(f"Using a deterministic random subset of {len(paths_list)} of "
                    f"{len(full_paths_list)} files.", output_log_file)

    if exp["method"] == "MFA":
        from evaluation.align_with_mfa import run_mfa
        jobs = [(resolve_audio_path(exp["dataset"], entry["audio"]), resolve_srt(entry["srt"]),
                 alignment_path(exp["id"], exp["dataset"], entry["srt"]))
                for entry in paths_list]
        stats = run_mfa(exp, jobs, CFG["mfa"],
                        log=lambda message: log_message(message, output_log_file))
    else:
        if CFG["run"]["vocab_analysis"]:
            analyze_phoneme_vocab_mappings(paths_list, log_file=output_log_file)
        stats = task_bfa_align(paths_list, exp, log_file=output_log_file)
        make_bfa_phoneme_dict(paths_list, exp, log_file=output_log_file)

    stats.update(collect_run_stats(paths_list, exp))
    # Re-written now that the run is over, so the manifest describes the OUTCOME too.
    write_manifest(exp, "done", extra={"stats": stats,
                                       "resumed_from_files": files_present,
                                       "processes": total_processes,
                                       "files_available": len(full_paths_list),
                                       "files_selected": len(paths_list)})
    log_message(f"[{timestamp_now()}] Finished experiment {e}/{n_total} ({exp['id']}): "
                f"{stats['files_written']} of {len(paths_list)} files", output_log_file)
    return "done"


process_id = 0
total_processes = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiments of evaluate.yaml.")
    parser.add_argument("shard", nargs="*", type=int, metavar="N K",
                        help="run every N-th experiment starting at the K-th (0-based), so N "
                             "shells with K = 0 .. N-1 cover the list between them")
    parser.add_argument("--config", default=None, help="default: evaluation/evaluate.yaml")
    parser.add_argument("--only", default=None, metavar="GLOB",
                        help="only the experiments whose name matches, e.g. 'timit__*'")
    parser.add_argument("--list", action="store_true",
                        help="print the experiments and what is cached, then stop")
    args = parser.parse_args()

    if args.shard:
        if len(args.shard) != 2 or not 0 <= args.shard[1] < args.shard[0]:
            parser.error("the two numbers are N K with 0 <= K < N")
        total_processes, process_id = args.shard
    configure(args.config)

    threads = CFG["run"]["torch_threads"]
    if threads is None and total_processes > 1:
        # Share the cores: parallel sweeps that each start a full-size torch thread pool only
        # contend (the Viterbi is a single python thread per process either way).
        threads = max(1, (os.cpu_count() or 1) // (2 * total_processes))
    if threads is not None:
        torch.set_num_threads(int(threads))

    all_experiments = build_experiments()
    if args.only:
        all_experiments = [exp for exp in all_experiments
                           if fnmatch.fnmatch(exp["id"], args.only)]
    # The split is over the whole list, so an experiment belongs to the same shell whichever
    # of them is asked.
    experiments = all_experiments[process_id::total_processes]

    print(f"{len(experiments)} experiment(s) from {CFG['config_path']}"
          + (f" (process {process_id} of {total_processes}; the list has "
             f"{len(all_experiments)})" if total_processes > 1 else "") + ":")
    for e, exp in enumerate(experiments, start=1):
        state, files = experiment_state(exp)
        print(f"  {e:>3}. {exp['id']:<58} {state:<8} {files:>6} files   [{exp['group']}]")
    line()
    if args.list:
        raise SystemExit(0)

    paths_list_cache = {}
    outcomes = {}
    for e, exp in enumerate(experiments, start=1):
        try:
            outcomes[exp["id"]] = run_experiment(exp, e, len(experiments), paths_list_cache)
        except Exception as ex:
            # One experiment failing - MFA not installed, a checkpoint gone - must not cost
            # the rest of the list. It stays `running` on disk, so the next start resumes it.
            traceback.print_exc()
            log_message(f"[{timestamp_now()}] FAILED: {exp['id']}: {ex}", log_path(exp["id"]))
            outcomes[exp["id"]] = "FAILED"

    line()
    for name, outcome in outcomes.items():
        print(f"  {outcome:<8} {name}")
    print(f"Alignments:  {OUTPUT_DIR}/<experiment>/   (delete a directory to drop a run)")
    print("Score them with:  python evaluation/compute_metrics.py")
    if "FAILED" in outcomes.values():
        raise SystemExit(1)
