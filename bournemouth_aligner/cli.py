import click
import json
import os
import textwrap
from .core import PhonemeTimestampAligner, DEFAULT_SEGMENT_MAX_SECONDS
from .presets import (DEFAULT_LANG, DEFAULT_MODEL, DEFAULT_PHONEME_SET, P4MBFA_MODELS,
                      PHONEME_SETS, get_preset, get_std_preset, std_languages)
from .chunking import (DEFAULT_CHUNK_SECONDS, DEFAULT_MAX_PHONEMES,
                       DEFAULT_PHONEMES_PER_SECOND)

__version__ = "2.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_lang(preset, model, lang, phoneme_set=None):
    """
    Return (model_name, language) after applying preset then any explicit overrides.

    `phoneme_set` None means the default set - unless `model` is given, in which case the
    checkpoint decides the set when it is loaded and the language is passed through as typed.
    """
    if phoneme_set == "ph66":
        if preset:
            preset_model, preset_lang = get_preset(preset, lang=lang or DEFAULT_LANG)
        else:
            preset_model, preset_lang = DEFAULT_MODEL, DEFAULT_LANG
        return model or preset_model, lang or preset_lang

    language = lang or preset or DEFAULT_LANG
    if model and phoneme_set is None:
        return model, language
    # Raises, with the way out in the message, for a language the std set cannot serve.
    preset_model, code, _group, _note = get_std_preset(language)
    return model or preset_model, code



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
@click.option('--phoneme-set', type=click.Choice(PHONEME_SETS), default=None,
              help=f'Phoneme set and, with it, the model family: "std" (standard_g2p '
                   f'phonemes, p4mbfa models) or "ph66" (espeak-ng, p3cupe models). '
                   f'Default: {DEFAULT_PHONEME_SET}, or whatever --model / --ckpt-path is.')
@click.option('--model', default=None,
              help='Published model to use instead of the preset\'s (advanced): mbfa-latin, '
                   'mbfa-latin-benchmark, mbfa-cyrillic, mbfa-cjk, mbfa-japanese, CUPE-3-en, '
                   'or a Hugging Face repository id. Downloaded on first use.')
@click.option('--ckpt-path', type=click.Path(), default=None,
              help='Path to a model on disk (advanced): a release directory (config.json + '
                   'model.safetensors) or a training checkpoint file, of either family. '
                   'Overrides --model and decides the phoneme set by itself.')
@click.option('--lang', default=None,
              help='Language code, e.g. en-us, de, fr, ru, zh (espeak-ng codes with '
                   '--phoneme-set=ph66). Overrides the language part of --preset.')
@click.option('--g2p-device', default=None,
              help='Device for the std set\'s G2P model. Default: same as --device.')
@click.option('--punctuation-sil/--no-punctuation-sil', default=False,
              help='std set only: write a silence at pause punctuation, as ph66 always '
                   'does. Off by default, matching how the p4mbfa models were trained.')
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
@click.option('--boundary-bonus-weight', default=None, type=float,
              help='Weight on the log-mel acoustic-change bonus added to every phone-onset '
                   "transition. Default: the checkpoint's own trained value (30 for the "
                   'shipped one). 0 turns it off; see the constructor docstring for what it '
                   'is measured to cost and buy.')
@click.option('--sub-frame-onsets/--frame-grid-onsets', default=True,
              help='Place onsets at the interpolated crossing of two posteriors (default) '
                   'rather than on the 5 ms frame grid.')
@click.option('--min-phone-frames', default=None, type=int,
              help='Shortest phone the aligner may produce, in 5 ms frames. Default: the '
                   "checkpoint's own training value.")
@click.option('--segment-max-seconds', default=DEFAULT_SEGMENT_MAX_SECONDS, type=float,
              show_default=True,
              help='Segments up to this long are decoded whole; longer ones are chunked to '
                   'bound time and memory. Whole decoding is more accurate, so raise it if '
                   'the hardware allows.')
@click.option('--chunk-audio/--no-chunk-audio', default=True,
              help='Align segments longer than --segment-max-seconds by walking overlapping '
                   'audio windows. Shorter segments are decoded whole and are unaffected.')
@click.option('--chunk-seconds', default=DEFAULT_CHUNK_SECONDS, type=float, show_default=True,
              help='Audio window length for chunked alignment of very long segments.')
@click.option('--chunk-max-phonemes', default=DEFAULT_MAX_PHONEMES, type=int, show_default=True,
              help='Cap on the phoneme budget for one chunk window.')
@click.option('--chunk-pps', default=DEFAULT_PHONEMES_PER_SECOND, type=float, show_default=True,
              help='Phonemes per second of SPEECH used to size each window budget.')
@click.option('--chunk-vad/--no-chunk-vad', default=True,
              help='Use an energy VAD to measure speech seconds per window (so pause-heavy '
                   'windows are not over-budgeted) and to land cuts in pauses.')
# ---- misc ----
@click.option('--debug/--no-debug', default=False,
              help='Print detailed progress information.')
@click.option('--list-presets', is_flag=True, default=False,
              help='Print all supported language presets and exit.')
@click.version_option(__version__, '--version', '-v', message='%(version)s')
def main(audio_path, text_or_srt, output_path,
         preset, phoneme_set, model, ckpt_path, lang, g2p_device, punctuation_sil,
         embeddings, mel_path, textgrid,
         device, duration_max, boundary_bonus_weight, sub_frame_onsets, min_phone_frames,
         segment_max_seconds, chunk_audio, chunk_seconds, chunk_max_phonemes, chunk_pps,
         chunk_vad, debug, list_presets):
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
        click.echo(_preset_help())
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
    if ckpt_path:
        # The path decides the model and the phoneme set; only the language is left to say.
        resolved_model, resolved_lang = None, lang or preset or DEFAULT_LANG
    else:
        try:
            resolved_model, resolved_lang = _resolve_model_lang(
                preset or (None if lang else DEFAULT_LANG), model, lang, phoneme_set)
        except ValueError as exc:
            raise click.UsageError(str(exc))

    # ------------------------------------------------------------------
    # Debug header
    # ------------------------------------------------------------------
    if debug:
        click.echo("Bournemouth Forced Aligner")
        click.echo(f"  Audio    : {audio_path}")
        click.echo(f"  Input    : {'[SRT] ' + srt_path if srt_mode else '[text] ' + repr(input_text)}")
        click.echo(f"  Output   : {output_path}")
        click.echo(f"  Model    : {ckpt_path or resolved_model}")
        shown_set = phoneme_set or ("from the model" if (ckpt_path or model)
                                    else DEFAULT_PHONEME_SET)
        click.echo(f"  Phonemes : {shown_set}")
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
            preset=None,
            model_name=resolved_model,
            ckpt_path=ckpt_path,
            lang=resolved_lang,
            phoneme_set=phoneme_set,
            g2p_device=g2p_device,
            punctuation_sil=punctuation_sil,
            duration_max=duration_max,
            device=device,
            boundary_bonus_weight=boundary_bonus_weight,
            sub_frame_onsets=sub_frame_onsets,
            min_phone_frames=min_phone_frames,
            segment_max_seconds=segment_max_seconds,
            chunk_audio=chunk_audio,
            chunk_seconds=chunk_seconds,
            chunk_max_phonemes=chunk_max_phonemes,
            chunk_phonemes_per_second=chunk_pps,
            chunk_vad=chunk_vad,
            # --debug turns on the informational output (frame geometry, decoder settings);
            # without it only real warnings are printed.
            warn_level=2 if debug else 1,
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
            output_data = dict(result)
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
--phoneme-set=ph66  (espeak-ng + p3cupe; one model, CUPE-3-en, trained on English)
──────────────────────────────────────────────────────────────────────────────
English       en-us  en  en-gb  en-gb-x-rp  en-gb-scotland
European      de  fr  es  it  pt  pt-br  nl  pl  da  sv  nb  fi  cs  hu  ...
Slavic/E.Eu.  ru  uk  be  bg  ro  hr  sr  sk  sl  mk
Indic         hi  bn  ur  pa  gu  mr  ta  te  kn  ml  si
Middle East   ar  he  fa  ku  am  mt
Turkic        tr  az  kk  ky  uz  tt  tk
Other         ka  eu  id  ms  el  sq  la  eo  ...

For the full list see: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
"""



def _preset_help():
    """Both preset tables: the std set's by language group, then ph66's."""
    lines = ["Supported --preset values", "",
             f"--phoneme-set=std  (standard_g2p + p4mbfa{', the default' if DEFAULT_PHONEME_SET == 'std' else ''})",
             "────────────────────────────────────────"]
    for group, codes in std_languages().items():
        lines.append(textwrap.fill("  ".join(codes), width=78, initial_indent=f"{group:<14}",
                                   subsequent_indent=" " * 14))
        lines.append(f"{'':<14}model: {P4MBFA_MODELS[group]}")
    lines += ["", "Any spelling of these works: en-us, en-GB, pt-br, cmn, zh-Hant, ...",
              "Languages outside this table need --phoneme-set=ph66.", "", _PRESET_HELP]
    return "\n".join(lines)


if __name__ == '__main__':
    main()
