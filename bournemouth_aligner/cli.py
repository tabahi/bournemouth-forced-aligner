import click
import os
from .core import PhonemeTimestampAligner

__version__ = "1.0.0"  # Set your version here

@click.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.argument('srt_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--model', default="en_libri1000_uj01d_e199_val_GER=0.2307.ckpt", 
              help='CUPE model name from HuggingFace')
@click.option('--lang', default='en-us', 
              help='Language code for phonemization (e.g., en-us, es, fr)')
@click.option('--device', default='cpu', 
              help='Device for inference (cpu/cuda)')
@click.option('--embeddings', type=click.Path(), 
              help='Path to save phoneme embeddings (.pt file)')
@click.option('--duration-max', default=10.0, type=float,
              help='Maximum segment duration in seconds')
@click.option('--debug/--no-debug', default=False, 
              help='Enable detailed debug output')
@click.option('--boost-targets/--no-boost-targets', default=True,
              help='Enable target phoneme boosting for better alignment')
@click.version_option(__version__, '--version', '-v', message='%(version)s')
def main(audio_path, srt_path, output_path, model, lang, device, embeddings, 
         duration_max, debug, boost_targets):
    """
    Bournemouth Forced Aligner - Extract phoneme timestamps from audio.
    
    AUDIO_PATH: Path to audio file (.wav, .mp3, etc.)
    SRT_PATH: Path to SRT file in JSON format
    OUTPUT_PATH: Path for output timestamps (.json)
    
    Example:
        balign audio.wav transcription.srt output.json --embeddings embeddings.pt
    """
    
    # Validate inputs
    if not os.path.exists(audio_path):
        click.echo(f"✗ Audio file not found: {audio_path}", err=True)
        raise click.Abort()
        
    if not os.path.exists(srt_path):
        click.echo(f"✗ SRT file not found: {srt_path}", err=True)
        raise click.Abort()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if debug:
        click.echo("🚀 Bournemouth Forced Aligner")
        click.echo(f"📁 Audio: {audio_path}")
        click.echo(f"📄 SRT: {srt_path}")
        click.echo(f"💾 Output: {output_path}")
        click.echo(f"🏷️  Language: {lang}")
        click.echo(f"🖥️  Device: {device}")
        click.echo(f"🎯 Model: {model}")
        if embeddings:
            click.echo(f"🧮 Embeddings: {embeddings}")
        click.echo("-" * 50)
    
    try:
        # Initialize aligner
        if debug:
            click.echo("🔧 Initializing aligner...")
            
        aligner = PhonemeTimestampAligner(
            model_name=model,
            lang=lang,
            duration_max=duration_max,
            device=device,
            boost_targets=boost_targets
        )
        
        if debug:
            click.echo("✅ Aligner initialized successfully")
            click.echo("🎵 Processing audio...")
        
        # Process audio
        result = aligner.process_srt_file(
            srt_path=srt_path,
            audio_path=audio_path,
            ts_out_path=output_path,
            vspt_path=embeddings,
            extract_embeddings=bool(embeddings),
            debug=debug
        )
        
        if result:
            click.echo(f"✅ Timestamps extracted to {output_path}")
            
            # Show summary
            total_segments = len(result.get('segments', []))
            total_phonemes = sum(len(seg.get('phoneme_ts', [])) for seg in result.get('segments', []))
            
            click.echo(f"📊 Processed {total_segments} segments with {total_phonemes} phonemes")
            
            if embeddings:
                click.echo(f"🧮 Embeddings saved to {embeddings}")
                
        else:
            click.echo("✗ Processing failed - no output generated", err=True)
            raise click.Abort()
            
    except KeyboardInterrupt:
        click.echo("\n⏹️  Processing interrupted by user", err=True)
        raise click.Abort()
        
    except Exception as e:
        click.echo(f"✗ Error during processing: {e}", err=True)
        if debug:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()
    
    if debug:
        click.echo("🎉 Processing completed successfully!")

if __name__ == '__main__':
    main()