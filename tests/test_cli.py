#!/usr/bin/env python3
"""
Test script for Bournemouth Forced Aligner CLI
"""

import subprocess
import sys
import tempfile
import torch
import torchaudio
import json
from pathlib import Path

def create_dummy_audio(duration=2.0, sample_rate=16000):
    """Create a dummy audio file for testing."""
    # Generate 2 seconds of dummy audio
    samples = int(duration * sample_rate)
    audio = torch.randn(1, samples) * 0.1  # Low volume random noise
    return audio

def test_cli_help():
    """Test CLI help command."""
    print("🧪 Testing CLI help...")
    try:
        result = subprocess.run(['balign', '--help'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ CLI help works!")
            print("📋 Help output preview:")
            print(result.stdout[:300] + "..." if len(result.stdout) > 300 else result.stdout)
            assert(True)  # Ensure help command runs without error
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            assert(False)  # Fail the test if help command fails
    except Exception as e:
        print(f"❌ CLI help error: {e}")
        assert(False)  # Fail the test if an exception occurs

def test_cli_version():
    """Test CLI version command."""
    print("\n🧪 Testing CLI version...")
    try:
        result = subprocess.run(['balign', '--version'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ CLI version works: {result.stdout.strip()}")
            assert(True)
        else:
            print(f"❌ CLI version failed: {result.stderr}")
            assert(False) 
    except Exception as e:
        print(f"❌ CLI version error: {e}")
        assert(False)  # Fail the test if an exception occurs

def test_cli_processing():
    """Test actual CLI processing with dummy data."""
    print("\n🧪 Testing CLI processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dummy audio file
        audio_path = temp_path / "test_audio.wav"
        dummy_audio = create_dummy_audio()
        torchaudio.save(str(audio_path), dummy_audio, 16000)
        
        # Create SRT file
        srt_path = temp_path / "test.srt"
        srt_data = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world"
                }
            ]
        }
        with open(srt_path, 'w') as f:
            json.dump(srt_data, f)
        
        # Output path
        output_path = temp_path / "output.json"
        
        try:
            print(f"📁 Temp files created in: {temp_dir}")
            print(f"🎵 Audio: {audio_path}")
            print(f"📄 SRT: {srt_path}")
            print(f"💾 Output: {output_path}")
            
            # Run CLI command
            cmd = [
                'balign',
                str(audio_path),
                str(srt_path), 
                str(output_path),
                '--debug'
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ CLI processing successful!")
                print("📤 Output:")
                print(result.stdout[-500:])  # Last 500 chars
                
                # Check if output file was created
                if output_path.exists():
                    print(f"✅ Output file created: {output_path}")
                    with open(output_path) as f:
                        data = json.load(f)
                        print(f"📊 Output contains {len(data.get('segments', []))} segments")
                    assert(True)
                else:
                    print("❌ Output file not created")
                    assert(False) 
            else:
                print(f"❌ CLI processing failed (exit code: {result.returncode})")
                print("📤 STDOUT:", result.stdout[-300:])
                print("📤 STDERR:", result.stderr[-300:])
                assert(False) 
                
        except subprocess.TimeoutExpired:
            print("❌ CLI processing timed out")
            assert(False) 
        except Exception as e:
            print(f"❌ CLI processing error: {e}")
            assert(False) 

def main():
    test_cli_help()
    test_cli_version()
    test_cli_processing()
if __name__ == "__main__":
    sys.exit(main())