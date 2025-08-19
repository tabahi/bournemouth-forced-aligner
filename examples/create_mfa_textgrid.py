'''
This code creates a TextGrid file using the Montreal Forced Aligner (MFA) for a given audio file and its transcription. This code is independent of the rest of the repository, and depends on MFA installation.
'''

import os
import shutil
import subprocess
from pathlib import Path

def align_single_file(audio_path, transcription, output_dir="mfa_output", corpus_dir = "mfa_corpus"):
    """
    Use Montreal Forced Aligner to generate phoneme-level timestamps
    
    Args:
        audio_path: Path to audio file
        transcription: Text content of the audio
        output_dir: Directory for outputs
    """
    
    # Setup directories
    
    textgrid_dir = f"{output_dir}/mfa_textgrids"
    
    # Clean and create directories
    if os.path.exists(corpus_dir):
        shutil.rmtree(corpus_dir)
    if os.path.exists(textgrid_dir):
        shutil.rmtree(textgrid_dir)
        
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    audio_name = Path(audio_path).stem
    
    # Copy audio file to corpus directory
    corpus_audio = f"{corpus_dir}/{audio_name}.wav"
    shutil.copy(audio_path, corpus_audio)
    
    # Create transcript file
    corpus_transcript = f"{corpus_dir}/{audio_name}.txt"
    with open(corpus_transcript, "w") as f:
        f.write(transcription)
    
    print(f"Created corpus files:")
    print(f"  Audio: {corpus_audio}")
    print(f"  Transcript: {corpus_transcript}")
    
    # Run MFA alignment
    print("Running MFA alignment...")
    result = subprocess.run([
        "mfa", "align",
        corpus_dir,
        "english_us_arpa",      # dictionary
        "english_us_arpa",      # acoustic model
        textgrid_dir,
        "--clean"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("MFA Error:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    
    # Find output TextGrid
    textgrid_path = f"{textgrid_dir}/{audio_name}.TextGrid"
    
    if os.path.exists(textgrid_path):
        print(f"TextGrid created: {textgrid_path}")
        return textgrid_path
    else:
        print("TextGrid not found")
        return None

# Example usage
if __name__ == "__main__":
    transcription = "in being comparatively modern."
    audio_path = "samples/LJSpeech/LJ001-0002.wav"
    
    # Make sure models are downloaded first
    print("Checking MFA models...")
    subprocess.run(["mfa", "model", "download", "dictionary", "english_us_arpa"])
    subprocess.run(["mfa", "model", "download", "acoustic", "english_us_arpa"])
    
    # Run alignment
    textgrid_path = align_single_file(audio_path, transcription, output_dir="samples/outputs/"+os.path.basename(audio_path))
    