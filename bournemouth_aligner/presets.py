


def get_preset(preset="en-us", lang='en-us'):
    
    # English - Use dedicated English model for best performance
    if preset in ["en-us", "en", "en-gb", "en-029", "en-gb-x-gbclan", "en-gb-x-rp", "en-gb-scotland", "en-gb-x-gbcwmd"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        #model_name = "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt" # OLDER version
        model_name = "en_libri1000_ua01c_e4_val_GER=0.2186.ckpt"  # new after rhotics fix

    # MLS8 European languages (trained on 8 European languages)
    elif preset in ["de", "fr", "fr-be", "fr-ch", "es", "es-419", "it", "pt", "pt-br", "pl", "nl"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"

    # Other European languages (similar to MLS8 training data)
    elif preset in ["da", "sv", "nb", "is", "cs", "sk", "sl", "hr", "bs", "sr", "mk", "bg", "ro", "hu", "et", "lv", "lt"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"

    # Romance languages (similar to trained Romance languages)
    elif preset in ["ca", "an", "pap", "ht"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"

    # Germanic languages (similar to trained Germanic languages)
    elif preset in ["af", "lb"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"

    # Celtic languages
    elif preset in ["ga", "gd", "cy"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt"

    # Indo-European languages - Use MSWC38 universal model
    elif preset in ["ru", "ru-lv", "uk", "be"]:  # Slavic
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    elif preset in ["hi", "bn", "ur", "pa", "gu", "mr", "ne", "as", "or", "si", "kok", "bpy", "sd"]:  # Indic
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    elif preset in ["fa", "fa-latn", "ku"]:  # Iranian
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    elif preset in ["el", "grc"]:  # Greek
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    elif preset in ["hy", "hyw"]:  # Armenian
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    elif preset in ["sq"]:  # Albanian
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    elif preset in ["la"]:  # Latin
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    # Turkic languages
    elif preset in ["tr", "az", "kk", "ky", "uz", "tt", "tk", "ug", "ba", "cu", "nog"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    # Uralic languages (note: Finnish is in both MLS8 and here, prioritize MLS8 above)
    elif preset in ["fi", "et", "hu", "smj"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    # Semitic languages
    elif preset in ["ar", "he", "am", "mt"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"


    # Malayo-Polynesian languages (Indonesian/Malay only - closer to Indo-European contact)
    elif preset in ["id", "ms"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    # Dravidian languages
    elif preset in ["ta", "te", "kn", "ml"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"


    # South Caucasian languages
    elif preset in ["ka"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"


    # Language isolates (maintain only Basque and Quechua - European/Indo-European contact)
    elif preset in ["eu", "qu"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"

    # Constructed languages
    elif preset in ["eo", "ia", "io", "lfn", "jbo", "py", "qdb", "qya", "piqd", "sjn"]:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"


    # Default fallback for unrecognized presets
    elif preset:
        if lang == 'en-us':  # Only override if using default lang
            lang = preset
        model_name = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"  # Universal model as fallback

    return model_name, lang