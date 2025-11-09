"""
Nano XYZ - A lightweight transformer language model.

Modules:
- model.py: Core model architecture (Transformer, LCR/GTR, RoPE/YaRN)
- attention_utils.py: Attention helpers (KV repetition, additive masks)
- configuration_nano.py / modeling_nano.py: Hugging Face compatible config and models
- train_hf.py: Trainer-based entrypoint (optional)
"""

__version__ = "1.0.0"

from .model import ModelArchitecture, ModelSettings, RopeWithYaRN, LCRBlock, GTRBlock
from .configuration_nano import NanoConfig
from .modeling_nano import NanoModel, NanoForCausalLM

__all__ = [
    "ModelArchitecture",
    "ModelSettings",
    "RopeWithYaRN",
    "LCRBlock",
    "GTRBlock",
    "NanoConfig",
    "NanoModel",
    "NanoForCausalLM",
]

