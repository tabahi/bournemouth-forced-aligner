"""
Where BFA's weights come from.

Two kinds of download, both resolved here and nowhere else, and both left to huggingface_hub:
they land in its own cache (`~/.cache/huggingface/hub`, moved with `$HF_HOME`), an offline
machine falls back to what is already cached (`HF_HUB_OFFLINE=1` skips the check), and nothing
is ever written inside the installed package.

    aligner models   one Hub repository each - `Tabahi/mbfa-<group>` for the std phoneme set,
                     `Tabahi/CUPE-3-en` for ph66. Only `config.json` and `model.safetensors`
                     are fetched (~55 MB), not the training checkpoint the repository also
                     holds for fine-tuning.
    the G2P model    `charsiu/g2p_multilingual_byT5_small_100`, one ByT5 for every language
                     (~1.2 GB), plus the three small tokenizer files of `google/byt5-small`.
                     Needed only by the std set and only when text is actually phonemized.

## Overrides

    $BFA_MODEL_DIR   a directory of aligner models, searched before the Hub: `<dir>/<name>/`
                     holding config.json + model.safetensors
    $BFA_G2P_DIR     a directory already holding the G2P model (config.json +
                     pytorch_model.bin, and optionally `tokenizer/`) - e.g. a CharsiuG2P
                     checkout's `tmp/local_model_weights/...`
"""

import os

# --- aligner models -------------------------------------------------------------------------

# A model named without a namespace ('mbfa-latin') is this account's repository of that name.
HF_NAMESPACE = "Tabahi"
# What a released model is: these two files, in a Hub repository or a directory.
MODEL_FILES = ("config.json", "model.safetensors")

# --- the G2P model --------------------------------------------------------------------------

G2P_REPO = "charsiu/g2p_multilingual_byT5_small_100"
G2P_FILES = ("config.json", "pytorch_model.bin")
# The charsiu repository has no tokenizer of its own: the model is a fine-tuned ByT5 and uses
# that tokenizer unchanged (bytes in, no vocabulary file - these three files are ~6 KB).
G2P_TOKENIZER_REPO = "google/byt5-small"
G2P_TOKENIZER_FILES = ("config.json", "tokenizer_config.json", "special_tokens_map.json")


def repo_id_of(model_name):
    """'mbfa-latin' -> 'Tabahi/mbfa-latin'; a name that already has a namespace is kept."""
    return model_name if "/" in model_name else f"{HF_NAMESPACE}/{model_name}"


def is_model_dir(path):
    """True for a directory holding a released model (`MODEL_FILES`)."""
    return all(os.path.isfile(os.path.join(path, name)) for name in MODEL_FILES)


def model_search_dirs():
    """Directories searched for `<name>/` before the Hub: $BFA_MODEL_DIR, then `./models`."""
    return tuple(d for d in (os.environ.get("BFA_MODEL_DIR"),
                             os.path.join(os.getcwd(), "models")) if d)


def find_model(model_name):
    """A released model's directory on disk, or None. Never touches the network."""
    name = model_name.rstrip("/").split("/")[-1]
    for directory in model_search_dirs():
        candidate = os.path.join(directory, name)
        if is_model_dir(candidate):
            return candidate
    return None


def resolve_model(model_name, local_dir=None, revision=None):
    """
    A model name -> a path `load_extractor` can read, downloading it if it is not on disk.

    Args:
        model_name: a released model's name ('mbfa-latin'), a full Hub repository id
            ('Tabahi/mbfa-latin', or anyone's repository in the same layout), or a path that
            already exists - a release directory or a training checkpoint file.
        local_dir: download into this directory instead of the huggingface_hub cache.
        revision: a branch, tag or commit of the repository.
    Raises:
        FileNotFoundError: not on disk and not downloadable, with every place that was tried.
    """
    if os.path.exists(model_name):
        return model_name
    found = find_model(model_name)
    if found is not None:
        return found

    repo_id = repo_id_of(model_name)
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, revision=revision, local_dir=local_dir,
                                 allow_patterns=list(MODEL_FILES))
    except Exception as e:
        searched = ", ".join(os.path.join(d, model_name.split("/")[-1])
                             for d in model_search_dirs())
        raise FileNotFoundError(
            f"could not obtain the model '{model_name}': it is not on disk (looked in "
            f"{searched}; set $BFA_MODEL_DIR to add a directory) and downloading "
            f"https://huggingface.co/{repo_id} failed with {type(e).__name__}: {e}. Pass "
            f"ckpt_path=... to point at a release directory or a checkpoint file "
            f"directly.") from e
    if not is_model_dir(path):
        raise FileNotFoundError(
            f"https://huggingface.co/{repo_id} is not a BFA model: it has no "
            f"{' + '.join(MODEL_FILES)}")
    return path


# --- G2P ------------------------------------------------------------------------------------

def _has(directory, names):
    return all(os.path.isfile(os.path.join(directory, name)) for name in names)


def ensure_g2p_model(model_dir=None, verbose=1):
    """
    Make sure the G2P model is on disk and return `(model_dir, tokenizer_dir)`.

    With no directory named (the argument, else $BFA_G2P_DIR) both come from the Hub through
    huggingface_hub's cache: ~1.2 GB the first time, a no-op after it.

    A directory that IS named is used as it stands and must already hold the model; nothing is
    downloaded into it. Its tokenizer is `<dir>/tokenizer/` when that exists (the layout of a
    CharsiuG2P checkout), the directory itself when the tokenizer files sit next to the model,
    and the Hub's otherwise.
    """
    model_dir = model_dir or os.environ.get("BFA_G2P_DIR")
    if model_dir:
        if not _has(model_dir, G2P_FILES):
            raise FileNotFoundError(
                f"{model_dir} does not hold the G2P model ({' + '.join(G2P_FILES)}). Leave "
                f"g2p_model_dir / $BFA_G2P_DIR unset to have it downloaded from "
                f"https://huggingface.co/{G2P_REPO}.")
        for tokenizer_dir in (os.path.join(model_dir, "tokenizer"), model_dir):
            if _has(tokenizer_dir, ("tokenizer_config.json",)):
                return model_dir, tokenizer_dir
        return model_dir, _download(G2P_TOKENIZER_REPO, G2P_TOKENIZER_FILES)

    if verbose >= 1 and not _is_cached(G2P_REPO, G2P_FILES):
        print(f"Downloading the G2P model {G2P_REPO} (~1.2 GB, once) ...")
    return (_download(G2P_REPO, G2P_FILES),
            _download(G2P_TOKENIZER_REPO, G2P_TOKENIZER_FILES))


def _is_cached(repo_id, names):
    """Whether every one of `names` is already in the huggingface_hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(repo_id, name), str) for name in names)
    except Exception:           # an old hub without the helper: just skip the announcement
        return True


def _download(repo_id, names):
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id, allow_patterns=list(names))
    except Exception as e:
        raise RuntimeError(
            f"could not download {', '.join(names)} from https://huggingface.co/{repo_id}: "
            f"{type(e).__name__}: {e}. If the G2P model is already on this machine, point "
            f"$BFA_G2P_DIR (or g2p_model_dir=) at the directory holding it.") from e
    if not _has(path, names):
        raise RuntimeError(f"https://huggingface.co/{repo_id} did not provide "
                           f"{', '.join(n for n in names if not _has(path, (n,)))}")
    return path
