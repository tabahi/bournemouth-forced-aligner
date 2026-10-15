"""
The Montreal Forced Aligner as one more model of the sweep.

Not run directly: `process_alignments.py` calls `run_mfa` for every experiment whose model is
an `method: mfa` entry of `evaluate.yaml`, and the result is written in the same place and
shape as a BFA run, so `compute_metrics.py` scores both alike.

MFA lives in its own conda environment (`mfa.conda_env` in the yaml, `mfa` here) and is only
ever reached through its command line, `conda run -n <env> mfa ...`. Everything else - cutting
the segment audio, parsing the TextGrids - runs in the environment the sweep runs in, so the
MFA environment needs nothing but MFA:

    conda create -n mfa -c conda-forge montreal-forced-aligner

Every segment of the corpus becomes one utterance (`<clip>__seg<N>.wav` + `.txt`) of a
scratch corpus under `tmp/mfa/`, MFA aligns that corpus in ONE call, and the TextGrids are
read back into one `<clip>.json` per clip. Timestamps are therefore relative to the segment,
not the clip; the manifest's `method: MFA` is what tells the scorer.
"""

import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict

import soundfile as sf
from tqdm import tqdm

from evaluation.layout import MFA_WORK_DIR


def mfa_command(conda_env):
    """The prefix that runs `mfa` inside its conda environment."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")
    if not conda:
        raise RuntimeError("neither conda nor mamba is on PATH, so the MFA environment "
                           f"{conda_env!r} cannot be reached")
    return [conda, "run", "--no-capture-output", "-n", conda_env, "mfa"]


def _run(command, log):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        log("MFA command failed: " + " ".join(command))
        log("STDOUT: " + (result.stdout or "")[-2000:])
        log("STDERR: " + (result.stderr or "")[-2000:])
        raise RuntimeError(f"`{' '.join(command[-6:])}` exited with {result.returncode}")
    return result.stdout


def ensure_mfa_model(mfa, kind, name, log):
    """Download a pretrained `acoustic` model or `dictionary` unless MFA already has it."""
    if os.path.exists(name):          # a local model file or dictionary
        return
    # MFA prints the list as `['english_us_arpa', ...]`
    if name in re.findall(r"[\w.-]+", _run(mfa + ["model", "list", kind], log)):
        return
    log(f"Downloading MFA {kind} model {name!r}...")
    _run(mfa + ["model", "download", kind, name], log)


def clean_text(text):
    """
    A segment's text as MFA is given it: punctuation replaced by spaces, apostrophes kept.

    The apostrophe is part of the word in MFA's dictionaries (`don't`, `it's`); without it
    the word is looked up as `dont`, is not found, and is aligned as an unknown word.
    """
    text = text.replace("\u2019", "'")
    return " ".join(re.sub(r"[^\w\s']", " ", text).split())


def build_corpus(jobs, corpus_dir):
    """
    Write one wav + txt per segment of every job that has no output yet.

    `jobs` is a list of `(audio_path, srt_path, out_path)`. Returns
    `{utterance name: {out_path, seg_idx, start, end, text}}`.
    """
    utterances = {}
    for audio_path, srt_path, out_path in tqdm(jobs, desc="Building MFA corpus"):
        if os.path.exists(out_path):
            continue
        if not os.path.exists(srt_path) or not os.path.exists(audio_path):
            continue
        with open(srt_path, "r") as f:
            segments = json.load(f)["segments"]

        # Read once per clip: a Buckeye interview holds over a hundred segments.
        audio, sample_rate = sf.read(audio_path, dtype="int16", always_2d=True)
        audio = audio[:, 0]
        stem = os.path.basename(out_path)[:-len(".json")]

        for seg_idx, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            cleaned = clean_text(text)
            if not cleaned:
                continue
            piece = audio[int(segment["start"] * sample_rate):int(segment["end"] * sample_rate)]
            if len(piece) == 0:
                continue
            name = f"{stem}__seg{seg_idx:04d}"
            sf.write(os.path.join(corpus_dir, f"{name}.wav"), piece, sample_rate,
                     subtype="PCM_16")
            with open(os.path.join(corpus_dir, f"{name}.txt"), "w") as f:
                f.write(cleaned)
            utterances[name] = {"out_path": out_path, "seg_idx": seg_idx,
                                "start": segment["start"], "end": segment["end"],
                                "text": text}
    return utterances


def parse_textgrid(textgrid_path):
    """
    The intervals of a TextGrid's `phones` tier, as `{start_ms, end_ms, phone}`. Empty
    intervals (MFA's silences) are left out.
    """
    phonemes = []
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_phones_tier = False
    current = {}
    for line in lines:
        line = line.strip()
        if 'name = "phones"' in line:
            in_phones_tier = True
            continue
        if in_phones_tier and line.startswith('name = '):
            break                                      # the next tier
        if not in_phones_tier:
            continue
        if line.startswith('intervals ['):
            current = {}
        elif line.startswith('xmin ='):
            # round, not int: 0.29 s is 289.99999999999994 ms as a float
            current['start_ms'] = int(round(float(line.split('=')[1]) * 1000))
        elif line.startswith('xmax ='):
            current['end_ms'] = int(round(float(line.split('=')[1]) * 1000))
        elif line.startswith('text ='):
            phone = line.split('=', 1)[1].strip().strip('"')
            if phone:
                phonemes.append({**current, 'phone': phone})
    return phonemes


def run_mfa(exp, jobs, mfa_cfg, log):
    """
    Align every job's segments with MFA and write one `<clip>.json` each.

    Args:
        exp: the experiment; `id`, `acoustic_model` and `dictionary` are read
        jobs: `(audio_path, srt_path, out_path)` per clip. A clip whose `out_path` exists
            is left alone, which is what resuming an interrupted run means here.
        mfa_cfg: the `mfa:` section of evaluate.yaml
        log: called with each line worth keeping
    Returns the run's statistics for the manifest.
    """
    mfa = mfa_command(mfa_cfg["conda_env"])
    ensure_mfa_model(mfa, "acoustic", exp["acoustic_model"], log)
    ensure_mfa_model(mfa, "dictionary", exp["dictionary"], log)

    # MFA keeps its working database under the corpus directory's NAME, so two runs must not
    # share one.
    corpus_dir = os.path.join(MFA_WORK_DIR, f"{exp['id']}_corpus")
    textgrid_dir = os.path.join(MFA_WORK_DIR, f"{exp['id']}_textgrids")
    for d in (corpus_dir, textgrid_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    utterances = build_corpus(jobs, corpus_dir)
    stats = {"total_segments": len(utterances), "failed_segments": 0, "failed_files": 0,
             "mfa_seconds": None}
    if not utterances:
        log("Nothing left to align.")
        return stats

    log(f"Running MFA on {len(utterances)} segments (one call)...")
    t0 = time.time()
    _run(mfa + ["align", corpus_dir, exp["dictionary"], exp["acoustic_model"], textgrid_dir,
                "--clean", "--num_jobs", str(mfa_cfg["num_jobs"])]
         + [str(a) for a in mfa_cfg["extra_args"] or []], log)
    stats["mfa_seconds"] = time.time() - t0
    log(f"MFA finished in {stats['mfa_seconds']:.1f} s "
        f"({stats['mfa_seconds'] / len(utterances) * 1000:.0f} ms per segment)")

    by_clip = defaultdict(list)
    for name, info in utterances.items():
        by_clip[info["out_path"]].append((name, info))

    for out_path, clip_utterances in tqdm(by_clip.items(), desc="Parsing TextGrids"):
        segments = []
        for name, info in sorted(clip_utterances, key=lambda item: item[1]["seg_idx"]):
            # MFA mirrors the corpus tree, and puts a flat corpus's files at the top.
            textgrid_path = os.path.join(textgrid_dir, f"{name}.TextGrid")
            phonemes = parse_textgrid(textgrid_path) if os.path.exists(textgrid_path) else []
            if not phonemes:
                stats["failed_segments"] += 1
                continue
            segments.append({"start": info["start"], "end": info["end"],
                             "text": info["text"], "phoneme_ts": phonemes})
        if not segments:
            stats["failed_files"] += 1
            continue
        with open(out_path, 'w') as f:
            json.dump({"segments": segments}, f)

    if stats["failed_segments"]:
        log(f"{stats['failed_segments']} segment(s) came back from MFA without phones.")
    if not mfa_cfg["keep_workdir"]:
        shutil.rmtree(corpus_dir)
        shutil.rmtree(textgrid_dir)
    return stats
