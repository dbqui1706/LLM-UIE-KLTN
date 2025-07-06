from dataclasses import dataclass, field


@dataclass
class DataArguments:
    system_prompt: str = field(
        default='You are an AI Model.',
        metadata={
            'help': 'The prompt for the system prompt.'
        },
    )
    train_file_path: str = field(
        default='data/train/data.json',
        metadata={
            "help": "The input data train path."
        }
    )
    valid_file_path: str = field(
        default='data/train/data.json',
        metadata={
            "help": "The input data train path."
        }
    )
