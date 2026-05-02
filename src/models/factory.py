"""Model registry and scale configurations."""

from .transformer import TransformerModel
from .lstm import LSTMModel
from .mamba_custom import MambaModel
from .mamba_posenc import MambaPosEncModel, OfficialMamba1Model, OfficialMamba2Model

MODEL_REG = {
    "transformer": TransformerModel,
    "lstm": LSTMModel,
    "mamba": MambaModel,
    "mamba_posenc": MambaPosEncModel,
    "mamba1_official": OfficialMamba1Model,
    "mamba2_official": OfficialMamba2Model,
}

SCALE_CONFIGS = {
    "small":  {"d_model": 48,  "n_layers": 2, "n_heads": 4, "d_ff": 192},
    "medium": {"d_model": 96,  "n_layers": 4, "n_heads": 4, "d_ff": 384},
    "large":  {"d_model": 192, "n_layers": 6, "n_heads": 8, "d_ff": 768},
}

OFFICIAL_MAMBA_CONFIGS = {
    "small":  {"d_model": 64,  "n_layers": 2},
    "medium": {"d_model": 128, "n_layers": 4},
    "large":  {"d_model": 256, "n_layers": 8},
}


def build_model(arch, vocab_size, **kwargs):
    cls = MODEL_REG[arch]
    return cls(vocab_size=vocab_size, **kwargs)
