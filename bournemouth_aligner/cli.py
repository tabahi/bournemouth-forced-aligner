import click
import json
import os
from .core import PhonemeTimestampAligner
from .presets import get_preset

__version__ = "1.1.4"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_lang(preset, model, lang):
    """Return (model_name, language) after applying preset then any explicit overrides."""
    if preset:
        preset_model, preset_lang = get_preset(preset, lang=lang or 'en-us')
    else:
        preset_model = "en_libri1000_ua01c_e4_val_GER=0.2186.ckpt"
        preset_lang = 'en-us'

    resolved_model = model or preset_model
    resolved_lang = lang or preset_lang
    return resolved_model, resolved_lang



# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command()
@click.argument('audio_path', required=False, default=None)
@click.argument('text_or_srt', required=False, default=None)
@click.argument('output_path', required=False, default=None)
# ---- preset / model / language ----
@click.option('--preset', default=None, metavar='LANG',
              help='Language preset, e.g. en-us, de, fr, hi. '
                   'Automatically picks the right model and language code. '
                   'Run `balign --list-presets` to see all options.')
@click.option('--model', default=None,
              help='Override the CUPE model name (advanced). '
                   'Default is chosen automatically by --preset.')
@click.option('--lang', default=None,
              help='espeak-ng language code, e.g. en-us, de, fr, hi. '
                   'Overrides the language part of --preset.')
# ---- output options ----
@click.option('--embeddings', type=click.Path(), default=None,
              help='Save per-phoneme embeddings to this .pt file.')
@click.option('--mel-path', type=click.Path(), default=None,
              help='Save mel spectrogram to this file. '
                   'Use .png for an image or .pt for a raw tensor.')
@click.option('--textgrid', type=click.Path(), default=None,
              help='Save a Praat TextGrid to this file.')
# ---- tuning ----
@click.option('--device', default='auto', show_default=True,
              help='Inference device: cpu, cuda, or auto.')
@click.option('--duration-max', default=60.0, type=float, show_default=True,
              help='Maximum segment duration in seconds (for windowed processing).')
@click.option('--boost-targets/--no-boost-targets', default=True,
              help='Boost expected phoneme probabilities before Viterbi alignment.')
@click.option('--silence-anchors', default=10, type=int, show_default=True,
              help='Number of silent frames to anchor at silences (punctuations) to help break the long segments. ')
# ---- misc ----
@click.option('--debug/--no-debug', default=False,
              help='Print detailed progress information.')
@click.option('--list-presets', is_flag=True, default=False,
              help='Print all supported language presets and exit.')
@click.version_option(__version__, '--version', '-v', message='%(version)s')
def main(audio_path, text_or_srt, output_path,
         preset, model, lang,
         embeddings, mel_path, textgrid,
         device, duration_max, boost_targets, silence_anchors,
         debug, list_presets):
    """Bournemouth Forced Aligner — extract phoneme timestamps from audio.

    \b
    TEXT mode (default):
        balign butterfly.wav "butterfly" --preset=en-us
        balign speech.wav "hello world" output.json --preset=en-us --mel-path=mel.png

    \b
    SRT/transcript file mode (auto-detected when TEXT_OR_SRT is an existing file):
        balign audio.wav transcript.srt.json output.json --preset=en-us

    \b
    When OUTPUT_PATH is omitted the result is saved next to the audio file,
    e.g. butterfly.wav → butterfly.vs.json
    """

    if list_presets:
        click.echo(_PRESET_HELP)
        return

    # ------------------------------------------------------------------
    # Validate required positional arguments
    # ------------------------------------------------------------------
    if not audio_path or not text_or_srt:
        raise click.UsageError(
            "Missing arguments: AUDIO_PATH and TEXT_OR_SRT are required.\n"
            "Usage: balign AUDIO_PATH TEXT_OR_SRT [OUTPUT_PATH] [OPTIONS]\n"
            "  Run 'balign --help' for full usage.\n"
            "  Run 'balign --list-presets' for supported language codes."
        )
    if not os.path.exists(audio_path):
        raise click.BadParameter(f"File not found: {audio_path}", param_hint="AUDIO_PATH")

    # ------------------------------------------------------------------
    # Resolve output path
    # ------------------------------------------------------------------
    if output_path is None:
        base = os.path.splitext(os.path.abspath(audio_path))[0]
        output_path = base + ".vs.json"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ------------------------------------------------------------------
    # Decide: text mode vs SRT mode
    # ------------------------------------------------------------------
    srt_mode = os.path.isfile(text_or_srt)
    if srt_mode:
        srt_path = text_or_srt
        input_text = None
    else:
        srt_path = None
        input_text = text_or_srt

    # ------------------------------------------------------------------
    # Resolve model + language
    # ------------------------------------------------------------------
    resolved_model, resolved_lang = _resolve_model_lang(
        preset or (None if lang else 'en-us'),
        model,
        lang
    )
    # If the user passed --preset AND --lang, honour --lang as override
    if lang:
        resolved_lang = lang
    if model:
        resolved_model = model

    # ------------------------------------------------------------------
    # Debug header
    # ------------------------------------------------------------------
    if debug:
        click.echo("Bournemouth Forced Aligner")
        click.echo(f"  Audio    : {audio_path}")
        click.echo(f"  Input    : {'[SRT] ' + srt_path if srt_mode else '[text] ' + repr(input_text)}")
        click.echo(f"  Output   : {output_path}")
        click.echo(f"  Model    : {resolved_model}")
        click.echo(f"  Language : {resolved_lang}")
        click.echo(f"  Device   : {device}")
        if mel_path:
            click.echo(f"  Mel path : {mel_path}")
        if textgrid:
            click.echo(f"  TextGrid : {textgrid}")
        if embeddings:
            click.echo(f"  Embeddings: {embeddings}")
        click.echo("-" * 52)

    # ------------------------------------------------------------------
    # Initialise aligner
    # ------------------------------------------------------------------
    try:
        if debug:
            click.echo("Initialising aligner...")

        aligner = PhonemeTimestampAligner(
            model_name=resolved_model,
            lang=resolved_lang,
            duration_max=duration_max,
            device=device,
            boost_targets=boost_targets,
            silence_anchors=silence_anchors
        )

        if debug:
            click.echo("Aligner ready. Processing audio...")

        # ------------------------------------------------------------------
        # Run alignment
        # ------------------------------------------------------------------
        if srt_mode:
            result = aligner.process_srt_file(
                srt_path=srt_path,
                audio_path=audio_path,
                ts_out_path=output_path,
                vspt_path=embeddings,
                extract_embeddings=bool(embeddings),
                batch_size=1,
                debug=debug
            )
        else:
            audio_wav = aligner.load_audio(audio_path)
            result_raw = aligner.process_sentence(
                text=input_text,
                audio_wav=audio_wav,
                extract_embeddings=bool(embeddings),
                debug=debug
            )

            if embeddings:
                result, p_emb, _g_emb = result_raw
                import torch
                os.makedirs(os.path.dirname(os.path.abspath(embeddings)), exist_ok=True)
                torch.save(p_emb, embeddings)
            else:
                result = result_raw

            # Wrap in SRT-style envelope and save
            output_data = {"segments": result.get("segments", [])}
            with open(output_path, "w", encoding="utf-8") as fout:
                json.dump(output_data, fout, indent=2, ensure_ascii=False)

        if not result:
            click.echo("Error: processing returned no output.", err=True)
            raise click.Abort()

        # ------------------------------------------------------------------
        # Optional outputs
        # ------------------------------------------------------------------
        if mel_path:
            if srt_mode:
                audio_wav = aligner.load_audio(audio_path)
            saved = aligner.plot_mel_alignment(result, audio_wav, mel_path)
            click.echo(f"Mel alignment plot saved to {saved}")

        if textgrid:
            os.makedirs(os.path.dirname(os.path.abspath(textgrid)), exist_ok=True)
            aligner.convert_to_textgrid(result, output_file=textgrid)
            click.echo(f"TextGrid saved to {textgrid}")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        segments = result.get("segments", [])
        total_phonemes = sum(len(s.get("phoneme_ts", [])) for s in segments)
        click.echo(f"Done. {len(segments)} segment(s), {total_phonemes} phoneme(s) → {output_path}")
        if embeddings:
            click.echo(f"Embeddings saved to {embeddings}")

    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        raise click.Abort()

    except click.Abort:
        raise

    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        if debug:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()

    if debug:
        click.echo("Completed successfully.")


# ---------------------------------------------------------------------------
# Preset help text
# ---------------------------------------------------------------------------
_PRESET_HELP = """\
Supported --preset values
─────────────────────────
English       en-us  en  en-gb  en-gb-x-rp  en-gb-scotland
European      de  fr  es  it  pt  pt-br  nl  pl  da  sv  nb  fi  cs  hu  ...
Slavic/E.Eu.  ru  uk  be  bg  ro  hr  sr  sk  sl  mk
Indic         hi  bn  ur  pa  gu  mr  ta  te  kn  ml  si
Middle East   ar  he  fa  ku  am  mt
Turkic        tr  az  kk  ky  uz  tt  tk
Other         ka  eu  id  ms  el  sq  la  eo  ...

For the full list see: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
"""

if __name__ == '__main__':
    main()
