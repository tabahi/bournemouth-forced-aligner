"""
Language preset -> checkpoint mapping.

## Model families

Two generations of checkpoint are supported and the aligner tells them apart from their own
weights, not from their filenames (`core.py::load_model`):

  model2i   the original contextless CUPE. No boundary head, so onset snapping reads a tap on
            the forward pass (`boundary_snap_source='auto'` gives it 'pre_transformer').
  model3a   model2i's base, byte for byte, plus a per-frame BoundaryHead. `'auto'` uses the
            head when it was actually trained - see the warning on `boundary_snap_source`.

The names below are the ONE place a checkpoint filename appears. `core.py` never special-cases
a name, so renaming a checkpoint on HuggingFace is an edit here and nowhere else.

## Which checkpoint covers which languages

The three model3a checkpoints were trained on three different corpora, so they inherit exactly
the coverage split the model2i generation had:

    MODEL3A_EN         libri_500h    English only
    MODEL3A_EU         MLS_7l_100h   7 European languages, English EXCLUDED
    MODEL3A_UNIVERSAL  mswc38_50h    38 languages, Indo-European and near families

## Legacy names

`MODEL2I_*` are the checkpoints BFA <= 1.1.5 shipped. They still load and still work; they are
kept here so `model_name=` can reach them and so the mapping below stays readable as a diff.
"""

# --- model3a (current) --------------------------------------------------------------------
# TODO: update these to the published filenames when the checkpoints are uploaded to
# huggingface.co/Tabahi/CUPE-2i/tree/main/ckpt. Until then `preset=` needs a local
# `cupe_ckpt_path=` instead, because `download_model` will 404 on these names.
MODEL3A_EN = "m_ue03c_epoch=81_step=39442_val_timit_lyr_fusion_f1_25ms=64.2429.ckpt"
MODEL3A_EU = "m_ue03f_epoch=19_step=12460_val_timit_lyr_fusion_f1_25ms=64.6520.ckpt"
MODEL3A_UNIVERSAL = "m_ue03d_epoch=4_step=13770_val_timit_lyr_fusion_f1_25ms=64.9033.ckpt"

# --- model2i (legacy, still supported) ------------------------------------------------------
MODEL2I_EN = "en_libri1000_ua01c_e4_val_GER=0.2186.ckpt"
MODEL2I_EU = "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"
MODEL2I_UNIVERSAL = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"




def get_preset(preset="en-us", lang='en-us'):
    
    # English - Use dedicated English model for best performance
    if preset in ["en-us", "en", "en-gb", "en-029", "en-gb-x-gbclan", "en-gb-x-rp", "en-gb-scotland", "en-gb-x-gbcwmd"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        #model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt" # OLDER version
        model_name = MODEL3A_EN

    # MLS8 European languages (trained on 8 European languages)
    elif preset in ["de", "fr", "fr-be", "fr-ch", "es", "es-419", "it", "pt", "pt-br", "pl", "nl"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_EU

    # Other European languages (similar to MLS8 training data)
    elif preset in ["da", "sv", "nb", "is", "cs", "sk", "sl", "hr", "bs", "sr", "mk", "bg", "ro", "hu", "et", "lv", "lt"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_EU

    # Romance languages (similar to trained Romance languages)
    elif preset in ["ca", "an", "pap", "ht"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_EU

    # Germanic languages (similar to trained Germanic languages)
    elif preset in ["af", "lb"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_EU

    # Celtic languages
    elif preset in ["ga", "gd", "cy"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_EU

    # Indo-European languages - Use MSWC38 universal model
    elif preset in ["ru", "ru-lv", "uk", "be"]:  # Slavic
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    elif preset in ["hi", "bn", "ur", "pa", "gu", "mr", "ne", "as", "or", "si", "kok", "bpy", "sd"]:  # Indic
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    elif preset in ["fa", "fa-latn", "ku"]:  # Iranian
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    elif preset in ["el", "grc"]:  # Greek
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    elif preset in ["hy", "hyw"]:  # Armenian
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    elif preset in ["sq"]:  # Albanian
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    elif preset in ["la"]:  # Latin
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    # Turkic languages
    elif preset in ["tr", "az", "kk", "ky", "uz", "tt", "tk", "ug", "ba", "cu", "nog"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    # Uralic languages (note: Finnish is in both MLS8 and here, prioritize MLS8 above)
    elif preset in ["fi", "et", "hu", "smj"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    # Semitic languages
    elif preset in ["ar", "he", "am", "mt"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL


    # Malayo-Polynesian languages (Indonesian/Malay only - closer to Indo-European contact)
    elif preset in ["id", "ms"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    # Dravidian languages
    elif preset in ["ta", "te", "kn", "ml"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL


    # South Caucasian languages
    elif preset in ["ka"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL


    # Language isolates (maintain only Basque and Quechua - European/Indo-European contact)
    elif preset in ["eu", "qu"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL

    # Constructed languages
    elif preset in ["eo", "ia", "io", "lfn", "jbo", "py", "qdb", "qya", "piqd", "sjn"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL


    # Default fallback for unrecognized presets
    elif preset:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = MODEL3A_UNIVERSAL  # Universal model as fallback
        print("Warning: Unrecognized preset '{}'. Defaulting to universal model with best effort language match.".format(preset))
    

    return model_name, lang