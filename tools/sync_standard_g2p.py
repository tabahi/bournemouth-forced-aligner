"""
Refresh the vendored copy of standard_g2p from a CharsiuG2P checkout.

    python tools/sync_standard_g2p.py [/path/to/CharsiuG2P]

`bournemouth_aligner/standard_g2p/` is a RUNTIME SNAPSHOT of
https://github.com/tabahi/CharsiuG2P's `standard_g2p/` package. That repository stays the source
of truth - it holds the dictionaries, the generators under `scripts/` and the measurements the
tables were built from - and nothing under the vendored directory is edited by hand. This
script is the only writer, so the two can always be diffed.

What is copied:

    the six runtime modules             verbatim
    mappings/lang_group_inventories.json  verbatim (the only mapping read at run time)
    LICENSE                             CharsiuG2P is MIT; the notice travels with the code

What is generated:

    dicts/zho-s.tsv, dicts/zho-t.tsv    KEYS ONLY. Mandarin segmentation is a longest match over
                                        the model's own dictionary and reads nothing but the
                                        first column, so the pronunciations (and the other 98
                                        dictionaries, 196 MB in all) stay behind.
    __init__.py                         points `word_segmentation.DICTS_DIR` at those files
    VENDORED.md                         the commit, whether the tree was dirty, the gold
                                        fingerprint

The two paths the upstream code resolves from its own repository root - the G2P weights and the
dictionaries - are redirected by assignment (`__init__.py` here, `phonemizers/std.py` for the
weights) rather than by patching the modules, which is what keeps the copy verbatim.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO_ROOT, "bournemouth_aligner", "standard_g2p")
DEFAULT_SRC = os.path.join(os.path.dirname(REPO_ROOT), "CharsiuG2P")

MODULES = ("gold_g2p.py", "lang_codes.py", "lang_group_inventory.py", "phoneme_features.py",
           "phoneme_inventory_gold.py", "word_segmentation.py")
MAPPINGS = ("lang_group_inventories.json",)
# CharsiuG2P tags whose dictionary KEYS the Mandarin segmenter matches against.
KEY_DICTS = ("zho-s", "zho-t")

INIT_PY = '''"""
standard_g2p - VENDORED from https://github.com/tabahi/CharsiuG2P (MIT, see LICENSE).

Do not edit anything in this directory by hand: `tools/sync_standard_g2p.py` is its only
writer, and VENDORED.md records the commit it was taken from. The modules are verbatim copies;
this file is the one BFA-owned piece, and all it does is tell the Mandarin segmenter where the
dictionary keys shipped with this package live (upstream resolves them from its repository
root, which does not exist in an installed package).
"""

import os as _os

from . import word_segmentation as _word_segmentation

_word_segmentation.DICTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "dicts")
'''


def _git(src, *args):
    try:
        return subprocess.run(["git", "-C", src, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _write_keys(src_tsv, dest_tsv):
    """First column of a CharsiuG2P dictionary, one `key<TAB>` per line, sorted."""
    with open(src_tsv, encoding="utf-8") as f:
        keys = sorted({line.split("\t", 1)[0] for line in f if line.strip()})
    # The trailing tab is deliberate: the upstream reader takes `line.split('\t', 1)[0]`, and
    # without a tab that would be the key WITH its newline.
    with open(dest_tsv, "w", encoding="utf-8") as f:
        f.writelines(f"{key}\t\n" for key in keys)
    return len(keys)


def sync(src):
    src_pkg = os.path.join(src, "standard_g2p")
    if not os.path.isdir(src_pkg):
        raise SystemExit(f"{src_pkg} not found; pass the CharsiuG2P checkout as an argument")

    os.makedirs(os.path.join(DEST, "mappings"), exist_ok=True)
    os.makedirs(os.path.join(DEST, "dicts"), exist_ok=True)

    for name in MODULES:
        shutil.copyfile(os.path.join(src_pkg, name), os.path.join(DEST, name))
    for name in MAPPINGS:
        shutil.copyfile(os.path.join(src_pkg, "mappings", name),
                        os.path.join(DEST, "mappings", name))
    shutil.copyfile(os.path.join(src, "LICENSE"), os.path.join(DEST, "LICENSE"))

    n_keys = {tag: _write_keys(os.path.join(src, "dicts", f"{tag}.tsv"),
                               os.path.join(DEST, "dicts", f"{tag}.tsv"))
              for tag in KEY_DICTS}

    with open(os.path.join(DEST, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(INIT_PY)

    # The fingerprint the vendored tables carry, computed the way lang_group_inventory does -
    # and by importing the copy just written, so a broken sync fails here and not at a user.
    sys.path.insert(0, REPO_ROOT)
    for name in [m for m in sys.modules if m.startswith("bournemouth_aligner.standard_g2p")]:
        del sys.modules[name]
    from bournemouth_aligner.standard_g2p import lang_group_inventory as LGI
    from bournemouth_aligner.standard_g2p import phoneme_inventory_gold as PI
    table = LGI.load()                       # raises StaleInventoryError on a mismatched pair
    fingerprint = LGI.gold_fingerprint()
    assert hashlib.sha1("\n".join(PI.TOKENS).encode("utf-8")).hexdigest()[:12] == fingerprint

    commit = _git(src, "rev-parse", "HEAD") or "unknown"
    dirty = [line[3:] for line in _git(src, "status", "--porcelain").splitlines()
             if line[3:].startswith("standard_g2p/") or line[3:].startswith("dicts/zho-")]
    groups = ", ".join(f"{g} {table['lang_groups'][g]['n_tokens']}" for g in table["lang_groups"])

    with open(os.path.join(DEST, "VENDORED.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Vendored standard_g2p\n\n"
            "Written by `tools/sync_standard_g2p.py`. Do not edit this directory by hand.\n\n"
            f"- source: https://github.com/tabahi/CharsiuG2P (`standard_g2p/`)\n"
            f"- commit: `{commit}`\n"
            f"- uncommitted changes in the copied files: "
            f"{', '.join(f'`{p}`' for p in dirty) if dirty else 'none'}\n"
            f"- synced: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
            f"- gold inventory: {PI.N_TOKENS} tokens, fingerprint `{fingerprint}`\n"
            f"- language groups (tokens, specials included): {groups}\n"
            f"- Mandarin dictionary keys: "
            f"{', '.join(f'{tag} {n}' for tag, n in n_keys.items())}\n\n"
            "A p4mbfa model is only valid against the token lists it was trained with. "
            "A released model (`Tabahi/mbfa-*`) carries its token list and this fingerprint "
            "in config.json, and BFA refuses to load one that disagrees with the tables "
            "here.\n")

    print(f"standard_g2p synced from {src} @ {commit[:12]}"
          f"{' (dirty: ' + ', '.join(dirty) + ')' if dirty else ''}")
    print(f"  gold fingerprint {fingerprint}; groups: {groups}")
    print(f"  Mandarin keys: {n_keys}")


if __name__ == "__main__":
    sync(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC)
