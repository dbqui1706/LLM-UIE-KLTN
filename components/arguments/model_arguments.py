from dataclasses import dataclass, field
from typing import Optional

from utils.util import MODEL_CLASSES


@dataclass
class ModelArguments:
    model_type: str = field(
        default="qwen2.5_14B",
        metadata={
            "help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
        }
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        }
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the model under 4B or not."
        }
    )
