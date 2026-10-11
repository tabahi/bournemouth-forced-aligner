'''
Loading a p3cupe model: a release directory (`Tabahi/CUPE-3-*`) or a training checkpoint.

The clip-level forward pass - windows in, one stitched 5 ms frame grid out - is the same for
both model families and lives in `segmental/extractor.py`. What is p3cupe's own is here: how
its model is rebuilt from a checkpoint, and that it has two heads, 67 ph66 classes and 17
phoneme groups.
'''

from ..segmental.extractor import BaseExtractor
from .model import CupeCNN

# The state_dict prefixes a p3cupe checkpoint carries. Anything else in it (the loss module's
# buffers, optimiser state) belongs to training and is dropped.
_TRUNK_PREFIXES = ('feature_extractor.', 'freq_attention.', 'context.', 'blocks.', 'norm.',
                   'classifier.', 'group_classifier.')


class CupeExtractor(BaseExtractor):
    '''A loaded p3cupe checkpoint. See `segmental.extractor.BaseExtractor`.'''

    family = 'p3cupe'
    low_confidence_level = 0.2

    def _build_model(self, checkpoint, verbose):
        if 'phoneme_classes' not in checkpoint['model_config']:
            raise ValueError(f"{self.ckpt_path} is not a p3cupe checkpoint (its model_config "
                             f"has no 'phoneme_classes')")
        model = CupeCNN()
        model.load_config_state_dict(checkpoint['model_config'])
        model.make_model()
        state = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        state = {k: v for k, v in state.items() if k.startswith(_TRUNK_PREFIXES)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"checkpoint is missing weights for: {sorted(missing)[:8]}")
        if unexpected and verbose >= 1:
            print(f"WARN: ignoring unexpected checkpoint keys: {sorted(unexpected)[:8]}")
        self.classes_dim = model.classes_dim
        self.groups_dim = model.groups_dim
        return model

    def _heads(self, emb):
        logits_class, logits_group = self.model.classify(emb)
        return {'ph': logits_class, 'group': logits_group}

    def check_inventory(self, inventory):
        super().check_inventory(inventory)
        # A released model names its classes; a training checkpoint only counts them.
        labels = ((self.release or {}).get('labels') or {}).get('phonemes')
        if labels is not None and list(labels) != inventory.tokens:
            changed = [f"{i}: {a!r} vs {b!r}"
                       for i, (a, b) in enumerate(zip(labels, inventory.tokens)) if a != b]
            raise ValueError(
                f"{self.ckpt_path} was trained on a different phoneme list than this "
                f"installation's '{inventory.name}' set ({'; '.join(changed[:6])}). Its "
                f"phoneme indices would be decoded as the wrong phonemes.")
