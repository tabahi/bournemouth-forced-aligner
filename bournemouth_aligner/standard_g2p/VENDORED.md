# Vendored standard_g2p

Written by `tools/sync_standard_g2p.py`. Do not edit this directory by hand.

- source: https://github.com/tabahi/CharsiuG2P (`standard_g2p/`)
- commit: `1a7ff18b1a855f83850de6272721ab2a9859d642`
- uncommitted changes in the copied files: `standard_g2p/gold_g2p.py`, `standard_g2p/lang_group_inventory.py`
- synced: 2026-10-07
- gold inventory: 270 tokens, fingerprint `9438371ed6dd`
- language groups (tokens, specials included): latin 208, vietnamese 45, cyrillic 127, other_alphabetic 104, abjad 90, brahmic 96, cjk 48, thai_khmer 38, japanese 28, burmese 4
- Mandarin dictionary keys: zho-s 44779, zho-t 48239

A p4mbfa model is only valid against the token lists it was trained with. A released model (`Tabahi/mbfa-*`) carries its token list and this fingerprint in config.json, and BFA refuses to load one that disagrees with the tables here.
