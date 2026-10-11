'''
Loading a p4mbfa model: a release directory (`Tabahi/mbfa-<group>`) or a training checkpoint.

The clip-level forward pass is shared with p3cupe (`segmental/extractor.py`). What is p4mbfa's
own is here: three heads instead of two, and a model that belongs to ONE language group.

## The inventory check

`ph_head` emits indices into a language group's token list, and that list lives in
standard_g2p, not in the weights. If the list is regenerated - a different coverage target,
a language moved in or out - every index can change meaning while `n_tokens` stays the same,
and nothing would raise: the aligner would simply align the wrong phonemes.

So a released model carries the token list it was trained on (config.json `labels.tokens`,
written by `export_hf.py` in the models repository), and `check_inventory` compares it against
the tables in this installation. A raw training checkpoint has no such record; it loads, with
only the group name and the token count to go on.
'''

from ..segmental.extractor import BaseExtractor
from .model import MbfaCNN


class MbfaExtractor(BaseExtractor):
    '''
    A loaded p4mbfa checkpoint. See `segmental.extractor.BaseExtractor`.

    Attributes (beyond the base's):
        lang_group: the language group `ph_head` was trained for.
        tones_dim: 22.
        stamp: the token list the model was trained on (`lang_group`, `tokens`,
            `gold_fingerprint`), from a release's config.json; None for a training checkpoint.
    '''

    family = 'p4mbfa'
    low_confidence_level = 0.1

    def _build_model(self, checkpoint, verbose):
        model = MbfaCNN()
        model.load_config_state_dict(checkpoint['model_config'])    # raises if not p4mbfa
        model.make_model()
        state = {k[len('model.'):]: v for k, v in checkpoint['state_dict'].items()
                 if k.startswith('model.')}
        # Strict: every parameter is trunk, phg_head or a fine head, and a checkpoint whose
        # heads have another shape is not this model.
        model.load_state_dict(state, strict=True)
        self.lang_group = model.lang_group
        self.classes_dim = model.n_tokens
        self.groups_dim = model.n_phg
        self.tones_dim = model.n_tones
        self.stamp = checkpoint.get('inventory')
        return model

    def _heads(self, emb):
        logits_ph, logits_phg, logits_tone = self.model.classify(emb)
        return {'ph': logits_ph, 'group': logits_phg, 'tone': logits_tone}

    def check_inventory(self, inventory):
        if inventory.lang_group != self.lang_group:
            raise ValueError(
                f"checkpoint was trained for the '{self.lang_group}' language group but the "
                f"phonemizer is producing '{inventory.name}' tokens; pick a language of the "
                f"'{self.lang_group}' group or a checkpoint for '{inventory.lang_group}'")
        super().check_inventory(inventory)
        if self.groups_dim != inventory.n_groups:
            raise ValueError(f"checkpoint emits {self.groups_dim} phoneme groups but the "
                             f"'{inventory.name}' inventory has {inventory.n_groups}")
        if self.stamp is None:
            return
        if list(self.stamp['tokens']) != inventory.tokens:
            changed = sum(a != b for a, b in zip(self.stamp['tokens'], inventory.tokens))
            raise ValueError(
                f"{self.ckpt_path} was trained on a different '{self.lang_group}' token list "
                f"than this installation's standard_g2p tables "
                f"(gold fingerprint {self.stamp.get('gold_fingerprint')} vs "
                f"{inventory.fingerprint}; {changed} of {inventory.n_classes} positions "
                f"differ). Its phoneme indices would be decoded as the wrong phonemes. "
                f"Use the BFA version this model was released for.")
