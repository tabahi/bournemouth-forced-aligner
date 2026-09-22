
"""

Usage (conda env: mfa):
    python3 modules/mfa/align_with_mfa.py

Setup:
    conda activate aligner
    conda install -c conda-forge mamba
    pip install torchaudio
    mamba run -n aligner aligner
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from tqdm import tqdm

try:
    # Only the MFA alignment helpers need audio I/O; scoring must stay usable even
    # when the torch/torchaudio install in the active env is broken.
    import torchaudio
except Exception as _torchaudio_error:      # ImportError, or OSError from a torch ABI mismatch
    torchaudio = None
    _TORCHAUDIO_ERROR = _torchaudio_error
else:
    _TORCHAUDIO_ERROR = None



def mfa_align_segment(audio_path, segment_start, segment_end, segment_text, temp_dir=None):
    """
    Align a single audio segment with MFA.

    Args:
        audio_path: Path to the full audio file
        segment_start: Segment start in seconds
        segment_end: Segment end in seconds
        segment_text: Transcript for the segment
        temp_dir: Temporary directory (created automatically if None)

    Returns:
        List of phoneme dicts, or None on failure
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    try:
        segment_audio_path = os.path.join(temp_dir, "segment.wav")
        extract_audio_segment(audio_path, segment_start, segment_end, segment_audio_path)

        textgrid_path = mfa_align_single_file(
            segment_audio_path,
            segment_text,
            output_dir=os.path.join(temp_dir, "mfa_output"),
            corpus_dir=os.path.join(temp_dir, "mfa_corpus"),
        )

        if textgrid_path and os.path.exists(textgrid_path):
            return parse_textgrid_to_json(textgrid_path)
        return None

    except Exception as e:
        print(f"Error in MFA alignment: {e}")
        return None
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_segments_with_mfa(audio_path, srt_path, output_path, temp_dir):
    """
    Align every segment in an SRT JSON file with MFA (one segment at a time).

    Returns:
        Dict with aligned segments
    """
    with open(srt_path, 'r') as f:
        srt_data = json.load(f)

    mfa_results = {"segments": []}

    for seg_idx, segment in tqdm(
        enumerate(srt_data["segments"]),
        total=len(srt_data["segments"]),
        desc="Processing segments with MFA",
        leave=False,
    ):
        text = segment["text"]
        if not text.strip():
            continue
        cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
        if not cleaned_text:
            continue

        phonemes = mfa_align_segment(
            audio_path, segment["start"], segment["end"], cleaned_text, temp_dir
        )

        if phonemes:
            mfa_results["segments"].append({
                "start": segment["start"],
                "end":   segment["end"],
                "text":  text,
                "phoneme_ts": phonemes,
            })
        else:
            print(f"Failed to align segment {seg_idx}: {text[:50]}...")

    with open(output_path, 'w') as f:
        json.dump(mfa_results, f, indent=2)

    return mfa_results


def align_with_mfa(paths_list_path, vs_ext=".mfa.json"):
    """
    Batch MFA alignment: prepare the full corpus once, run MFA once, parse all TextGrids.

    Args:
        paths_list_path: JSON file listing {audio, srt, vs} path triplets
        vs_ext: Extension for the output alignment JSON files
    """
    # clean up old MFA outputs
    for d in ("tmp/data/mfa_corpus/", "tmp/data/collect/alignment/"):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


    print("Checking MFA models...")
    subprocess.run(
        ["mamba", "run", "-n", "aligner", "mfa", "model", "download", "dictionary", "english_us_arpa"],
        capture_output=True, text=True,
    )
    subprocess.run(
        ["mamba", "run", "-n", "aligner", "mfa", "model", "download", "acoustic", "english_us_arpa"],
        capture_output=True, text=True,
    )

    with open(paths_list_path, "r") as f:
        paths_list = json.load(f)


    corpus_dir   = os.path.join("tmp/data/mfa_corpus/", "mfa_corpus_batch")
    textgrid_dir = os.path.join("tmp/data/mfa_corpus/", "mfa_textgrids_batch")

    for d in (corpus_dir, textgrid_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    corpus_file_map = {}

    print("Preparing corpus for MFA batch alignment...")
    for i in tqdm(range(len(paths_list)), desc="Building corpus"):
        audio_path = paths_list[i]['audio']
        srt_path   = paths_list[i]['srt']
        vs_path    = paths_list[i]['vs'].replace('.vs.*', vs_ext)

        if srt_path.startswith('$/'): srt_path = os.path.join(COMMON_DIR, srt_path[2:])
        if vs_path.startswith('$/'):  vs_path  = os.path.join(COMMON_DIR, vs_path[2:])

        if os.path.exists(vs_path):
            os.remove(vs_path)
        if not os.path.exists(srt_path) or not os.path.exists(audio_path):
            continue

        with open(srt_path, "r") as f:
            srt_data = json.load(f)

        for seg_idx, segment in enumerate(srt_data["segments"]):
            text = segment.get("text", "").strip()
            if not text:
                continue
            cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
            if not cleaned_text:
                continue

            seg_name = f"file{i:05d}_seg{seg_idx:04d}"
            extract_audio_segment(
                audio_path,
                segment["start"],
                segment["end"],
                os.path.join(corpus_dir, f"{seg_name}.wav"),
            )
            with open(os.path.join(corpus_dir, f"{seg_name}.txt"), "w") as f:
                f.write(cleaned_text)

            corpus_file_map[seg_name] = {
                "file_idx": i,
                "seg_idx":  seg_idx,
                "vs_path":  vs_path,
                "srt_path": srt_path,
                "start":    segment["start"],
                "end":      segment["end"],
                "text":     text,
            }

    if not corpus_file_map:
        print("No files to align.")
        return

    print(f"Corpus prepared: {len(corpus_file_map)} segments.")

    print("Running MFA alignment on entire corpus (single call)...")
    t0 = time.time()
    result = subprocess.run([
        "mamba", "run", "-n", "aligner",
        "mfa", "align",
        corpus_dir,
        "english_us_arpa",
        "english_us_arpa",
        textgrid_dir,
        "--clean",
        "--num_jobs", "1",
    ], capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("MFA Error:")
        print("STDOUT:", result.stdout[-2000:] if result.stdout else "")
        print("STDERR:", result.stderr[-2000:] if result.stderr else "")
        return

    print(
        f"MFA completed in {elapsed:.1f}s for {len(corpus_file_map)} segments "
        f"({elapsed / len(corpus_file_map) * 1000:.0f}ms per segment)"
    )

    vs_path_groups = defaultdict(list)
    for seg_name, info in corpus_file_map.items():
        vs_path_groups[info["vs_path"]].append((seg_name, info))

    saved = 0
    for vs_path, segments in tqdm(vs_path_groups.items(), desc="Parsing TextGrids"):
        mfa_results = {"segments": []}
        segments.sort(key=lambda x: x[1]["seg_idx"])

        for seg_name, info in segments:
            textgrid_path = os.path.join(textgrid_dir, f"{seg_name}.TextGrid")
            if os.path.exists(textgrid_path):
                phonemes = parse_textgrid_to_json(textgrid_path)
                if phonemes:
                    mfa_results["segments"].append({
                        "start": info["start"],
                        "end":   info["end"],
                        "text":  info["text"],
                        "phoneme_ts": phonemes,
                    })
                else:
                    print(f"Empty TextGrid for {seg_name}")
            else:
                print(f"TextGrid not found for {seg_name}")

        if mfa_results["segments"]:
            os.makedirs(os.path.dirname(vs_path), exist_ok=True)
            with open(vs_path, 'w') as f:
                json.dump(mfa_results, f, indent=2)
            saved += 1

    print(f"Saved {saved} MFA alignment files.")
    exit()



def parse_textgrid_to_json(textgrid_path):
    """
    Parse a TextGrid file and return phoneme intervals from the phones tier.

    Returns:
        List of dicts with keys: start_ms, end_ms, phone
    """
    phonemes = []
    if not os.path.exists(textgrid_path):
        return phonemes

    with open(textgrid_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_phones_tier = False
    current_interval = {}

    for line in lines:
        line = line.strip()
        if 'name = "phones"' in line:
            in_phones_tier = True
            continue
        elif 'name = "words"' in line and in_phones_tier:
            break

        if in_phones_tier:
            if line.startswith('intervals ['):
                current_interval = {}
            elif line.startswith('xmin ='):
                current_interval['start_ms'] = int(float(line.split('=')[1].strip()) * 1000)
            elif line.startswith('xmax ='):
                current_interval['end_ms'] = int(float(line.split('=')[1].strip()) * 1000)
            elif line.startswith('text ='):
                phone = line.split('=')[1].strip().strip('"')
                if phone:
                    current_interval['phone'] = phone
                    phonemes.append(current_interval.copy())

    return phonemes


def extract_audio_segment(audio_path, start_time, end_time, output_path):
    """
    Extract [start_time, end_time] (seconds) from audio_path and save to output_path.
    """
    if torchaudio is None:
        raise RuntimeError(
            "torchaudio is required to extract audio segments for MFA, but it failed "
            f"to import: {_TORCHAUDIO_ERROR}"
        )
    waveform, sample_rate = torchaudio.load(audio_path)
    start_sample = int(start_time * sample_rate)
    end_sample   = int(end_time   * sample_rate)
    torchaudio.save(output_path, waveform[:, start_sample:end_sample], sample_rate)


def mfa_align_single_file(audio_path, transcription, output_dir="mfa_output", corpus_dir="mfa_corpus"):
    """
    Run MFA alignment on a single audio file.

    Args:
        audio_path: Path to audio file
        transcription: Text content of the audio
        output_dir: Directory for MFA outputs
        corpus_dir: Temporary corpus directory

    Returns:
        Path to the output TextGrid, or None on failure
    """
    textgrid_dir = f"{output_dir}/mfa_textgrids"

    if os.path.exists(corpus_dir):
        shutil.rmtree(corpus_dir)
    if os.path.exists(textgrid_dir):
        shutil.rmtree(textgrid_dir)
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(output_dir,  exist_ok=True)

    audio_name = Path(audio_path).stem
    shutil.copy(audio_path, f"{corpus_dir}/{audio_name}.wav")
    with open(f"{corpus_dir}/{audio_name}.txt", "w") as f:
        f.write(transcription)

    result = subprocess.run([
        "mamba", "run", "-n", "aligner",
        "mfa", "align",
        corpus_dir,
        "english_us_arpa",
        "english_us_arpa",
        textgrid_dir,
        "--clean",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("MFA Error:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    textgrid_path = f"{textgrid_dir}/{audio_name}.TextGrid"
    if os.path.exists(textgrid_path):
        return textgrid_path

    print("TextGrid not found")
    return None