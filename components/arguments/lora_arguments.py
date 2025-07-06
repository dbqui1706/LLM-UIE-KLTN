from dataclasses import dataclass, field
from typing import List


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=256,
        metadata={
            "help": "The rank of decomposition matrix in LoRA."
        }
    )
    lora_alpha: int = field(
        default=512,
        metadata={
            "help": "The ratio of decomposition matrix in LoRA."
        }
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use RSLoRA or not."
        }
    )
    use_gradient_checkpointing: str = field(
        default="unsloth",
        metadata={
            "help": "Type of gradient checkpointing to use."
        }
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={
            "help": "The dropout rate during training."
        }
    )
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        metadata={
            "help": "The target module names use LoRA."
        }
    )
