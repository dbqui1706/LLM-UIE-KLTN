import os
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
from pprint import pprint

from .prompt import PromptManager
from .parser import TaskParser
from .extract import UIEResult, NER, RE, EE

logger = logging.getLogger(__name__)


class TaskType(Enum):
    NER = "NER"
    RE = "RE"
    EE = "EE"
    ALL = "ALL"


@dataclass
class TaskConfig:
    """Configuration for each task type"""
    system_prompt: str
    schema_keys: List[str]
    parser_method: str


class TaskConfigManager:
    """Manages task configurations"""

    TASK_CONFIGS = {
        TaskType.NER: TaskConfig(
            system_prompt="You are an expert in Named Entity Recognition (NER) task.",
            schema_keys=['entity_types'],
            parser_method='parse_ner'
        ),
        TaskType.RE: TaskConfig(
            system_prompt="You are an expert in Relation Extraction (RE) task.",
            schema_keys=['relation_types'],
            parser_method='parse_re'
        ),
        TaskType.EE: TaskConfig(
            system_prompt="You are an expert in Event Extraction (EE) task.",
            schema_keys=['event_types', 'argument_types'],
            parser_method='parse_ee'
        )
    }

    @classmethod
    def get_config(cls, task_type: TaskType) -> TaskConfig:
        return cls.TASK_CONFIGS.get(task_type)

    @classmethod
    def get_system_prompt(cls, task_type: TaskType) -> Optional[str]:
        config = cls.get_config(task_type)
        return config.system_prompt if config else None

    @classmethod
    def create_task_schema(cls, task_type: TaskType, user_schema: Optional[Dict[str, List[str]]]) -> Optional[
        Dict[str, List[str]]]:
        if not user_schema:
            return None

        config = cls.get_config(task_type)
        if not config:
            return None

        task_schema = {}
        for key in config.schema_keys:
            if key in user_schema:
                task_schema[key] = user_schema[key]

        return task_schema if task_schema else None


@dataclass
class BaseModel(ABC):
    """Base class for all UIE models"""
    model_name: str
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.prompter = PromptManager()
        self.parser = TaskParser()
        self.task_config_manager = TaskConfigManager()

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def extract(self, text: str, task: str, **kwargs):
        """Extract information from the given text"""
        pass

    def _generate_response(self, prompt: str, max_new_tokens: int = 512,
                           temperature: float = 0.1, **kwargs) -> str:
        """Generate response using the loaded model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # ✅ Prepare generation parameters
            generation_params = {
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }

            # ✅ Set defaults only if not provided in kwargs
            if 'do_sample' not in kwargs:
                generation_params['do_sample'] = True
            if 'top_p' not in kwargs:
                generation_params['top_p'] = 0.8

            # ✅ Update with user-provided kwargs (overwrites defaults)
            generation_params.update(kwargs)

            with torch.no_grad():
                outputs = self.model.generate(**generation_params)

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

    def format_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompt using chat template"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class LLamaModel(BaseModel):
    """LLaMA model implementation for UIE tasks"""

    def __init__(self, model_name: str = 'quidangz/LLama-8B-Instruct-MultiTask'):
        super().__init__()
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        """Load LLaMA model and tokenizer"""
        try:
            logger.info(f"Loading model {self.model_name}...")

            if torch.cuda.is_available():
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    dtype=None,
                    max_seq_length=2048,
                )
                self.model = FastLanguageModel.for_inference(self.model)
                logger.info(f"Model loaded on GPU")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.model.eval()
                logger.info(f"Model loaded on CPU")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract(self, text: str, task: str,
                user_schema: Optional[Dict[str, List[str]]] = None,
                mode: str = "flexible", **kwargs) -> Union[List[NER], List[RE], List[EE], UIEResult]:
        """Extract information from text"""
        if not text.strip():
            return self._get_empty_result(task)

        try:
            task_type = TaskType(task.upper())

            if task_type == TaskType.ALL:
                return self._extract_all_tasks(text, user_schema, mode, **kwargs)
            else:
                return self._extract_single_task(text, task_type, user_schema, mode, **kwargs)

        except ValueError:
            logger.error(f"Unsupported task: {task}")
            return self._get_empty_result(task)
        except Exception as e:
            logger.error(f"Error in extraction for task {task}: {e}")
            return self._get_empty_result(task)

    def _get_empty_result(self, task: str):
        """Get empty result based on task type"""
        return UIEResult(text="") if task.upper() == 'ALL' else []

    def _extract_single_task(self, text: str, task_type: TaskType,
                             user_schema: Optional[Dict[str, List[str]]] = None,
                             mode: str = "flexible", **kwargs):
        """Extract information for a single task using configuration"""

        # Get task configuration
        system_prompt = self.task_config_manager.get_system_prompt(task_type)
        task_schema = self.task_config_manager.create_task_schema(task_type, user_schema)

        # Generate and format prompt
        prompt = self.prompter.create_prompt(task_type.value, text, task_schema, mode)
        # pprint(f"\n============== Prompt ==============  \n{prompt}\n")
        formatted_prompt = self.format_chat_template(system_prompt, prompt)

        # Generate response
        response = self._generate_response(formatted_prompt, **kwargs)

        # Parse response using dynamic method calling
        config = self.task_config_manager.get_config(task_type)
        parser_method = getattr(self.parser, config.parser_method)

        # if task_type == TaskType.NER:
        #     return parser_method(response, text)
        # else:
        #     return parser_method(response)
        return parser_method(response)

    def _extract_all_tasks(self, text: str, user_schema: Optional[Dict[str, List[str]]] = None,
                           mode: str = "flexible", **kwargs) -> UIEResult:
        """Extract information for all tasks"""
        results = {}
        task_mapping = {
            TaskType.NER: 'entities',
            TaskType.RE: 'relations',
            TaskType.EE: 'events'
        }

        # Extract for each task
        for task_type, result_key in task_mapping.items():
            results[result_key] = self._extract_single_task(text, task_type, user_schema, mode, **kwargs)

        return UIEResult(text=text, **results)