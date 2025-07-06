import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging
import re

from prompt import PromptManager
from parser import TaskParser
from extract import UIEResult, NER, RE, EE

logger = logging.getLogger(__name__)


@dataclass
class BaseModel(ABC):
    """
    Base class for all models in the application.
    This class defines the basic structure and methods that all models should implement.
    """

    model_name: str
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        """
        Post-initialization method to set up the model.
        This method should be overridden by subclasses to load the model and tokenizer.
        """
        self.tokenizer = None
        self.model = None
        self.prompter = PromptManager()
        self.parser = TaskParser

    @abstractmethod
    def load_model(self):
        """
        Load the model and tokenizer.
        This method should be implemented by subclasses to load their specific models.
        """
        pass

    def _load_model_gpu(self):
        if torch.cuda.is_available():
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                dtype=None,
                max_seq_length=1024,
            )
        else:
            raise RuntimeError("GPU is not available. Please check your setup.")

    @abstractmethod
    def extract(self, text: str, task: str, **kwargs):
        """Extract information from the given text."""
        pass

    @abstractmethod
    def extract_batch(self, texts: List[str], task: str, **kwargs):
        """Extract information from a batch of texts."""
        pass

    def _generate_response(self,
                           prompt: str,
                           max_new_tokens: int = 512,
                           temperature: float = 0.1,
                           do_sample: bool = True,
                           top_p: float = 0.9,
                           top_k: int = 50,
                           repetition_penalty: float = 1.1,
                           **kwargs) -> str:
        """
        Generate response using the loaded model

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Generated response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)

            # Generate with model
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode response (only new tokens)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

    def _generate_batch_response(self,
                                 prompts,
                                 max_new_tokens: int = 512,
                                 temperature: float = 0.1,
                                 do_sample: bool = True,
                                 top_p: float = 0.9,
                                 top_k: int = 50,
                                 repetition_penalty: float = 1.1,
                                 **kwargs) -> List[str]:

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length - max_new_tokens if hasattr(self.tokenizer,
                                                                                       'model_max_length') else 4096 - max_new_tokens,
                padding=True
            ).to(self.model.divece)

            # Generate with model
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode response (only new tokens)
            input_lengths = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_lengths:]

            # Batch decode
            decoded_outputs = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

            return [response.strip() for response in decoded_outputs]

        except Exception as e:
            logger.error(f"Error generating batch response: {e}")
            return []

    def format_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt


class LLamaModel(BaseModel):
    """
    Example implementation of a specific model using the BaseModel structure.
    This class should implement the load_model, extract, and extract_batch methods.
    """

    def __init__(self, model_name: str = 'quidangz/LLama-8B-Instruct-MultiTask-CE') -> None:
        super().__init__()
        self.model_name = model_name
        self.load_model()

        # Generation parameters
        self.generation_config = {
            'max_new_tokens': 512,
            'temperature': 0.1,
            'do_sample': True,
            'top_p': 0.9,
        }

    def load_model(self):
        """Load LLaMA model and tokenizer"""
        try:
            logger.info(f"Loading model {self.model_name}...")
            if not torch.cuda.is_available():
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                # Set pad token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

                if self.device.type != "cuda":
                    self.model = self.model.to(self.device)

                # Set to evaluation mode
                self.model.eval()

                logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
            else:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    dtype=None,
                    max_seq_length=1024,
                )
                logger.info(f"Model {self.model_name} loaded successfully on GPU")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def extract(self,
                text: str,
                task: str,
                user_schema: Optional[Dict[str, List[str]]] = None,
                mode: str = "flexible",
                **kwargs) -> Union[List[NER], List[RE], List[EE], UIEResult]:
        """
        Extract information from the given text for a specific task

        Args:
            text: Input text to analyze
            task: Task type ('NER', 'RE', 'EE', 'ALL')
            user_schema: Optional user-provided schema
            mode: Extraction mode ('strict', 'flexible', 'open')
            **kwargs: Additional generation parameters

        Returns:
            Extraction results based on task type
        """
        if not text.strip():
            logger.warning("Empty text provided")
            return [] if task != 'ALL' else UIEResult(text="")

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Update generation config with kwargs
            gen_config = {**self.generation_config, **kwargs}

            if task.upper() == 'ALL':
                return self._extract_all_tasks(text, user_schema, mode, **gen_config)
            else:
                return self._extract_single_task(text, task.upper(), user_schema, mode, **gen_config)

        except Exception as e:
            logger.error(f"Error in extraction for task {task}: {e}")
            return [] if task != 'ALL' else UIEResult(text=text)

    def _extract_single_task(self,
                             text: str,
                             task: str,
                             user_schema: Optional[Dict[str, List[str]]] = None,
                             mode: str = "flexible",
                             **gen_config) -> Union[List[NER], List[RE], List[EE]]:
        """Extract information for a single task"""

        # Create task-specific schema
        task_schema = None
        if user_schema:
            if task == 'NER' and 'entity_types' in user_schema:
                task_schema = {'entity_types': user_schema['entity_types']}
            elif task == 'RE' and 'relation_types' in user_schema:
                task_schema = {'relation_types': user_schema['relation_types']}
            elif task == 'EE':
                task_schema = {}
                if 'event_types' in user_schema:
                    task_schema['event_types'] = user_schema['event_types']
                if 'argument_types' in user_schema:
                    task_schema['argument_types'] = user_schema['argument_types']

        # Generate prompt
        prompt = self.prompter.create_prompt(task, text, task_schema, mode)
        logger.debug(f"Generated prompt for {task}: {prompt[:200]}...")

        # Format prompt for chat model
        formatted_prompt = self.format_chat_template(prompt, text)
        print(formatted_prompt)

        # Generate response
        response = self._generate_response(formatted_prompt, **gen_config)
        logger.debug(f"Model response for {task}: {response}")

        # Parse response based on task
        if task == 'NER':
            return self.parser.parse_ner(response, text)
        elif task == 'RE':
            # For RE, we might need entities first for better parsing
            entities = []
            return self.parser.parse_re(response, entities)
        elif task == 'EE':
            entities = []
            return self.parser.parse_ee(response, entities)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _extract_all_tasks(self,
                           text: str,
                           user_schema: Optional[Dict[str, List[str]]] = None,
                           mode: str = "flexible",
                           **gen_config) -> UIEResult:
        """Extract information for all tasks and return unified result"""

        logger.info(f"Extracting all tasks for text: {text[:100]}...")

        # Extract entities first
        entities = self._extract_single_task(text, 'NER', user_schema, mode, **gen_config)
        logger.debug(f"Extracted {len(entities)} entities")

        # Extract relations (with entity context)
        relations = self._extract_single_task(text, 'RE', user_schema, mode, **gen_config)
        logger.debug(f"Extracted {len(relations)} relations")

        # Extract events (with entity context)
        events = self._extract_single_task(text, 'EE', user_schema, mode, **gen_config)
        logger.debug(f"Extracted {len(events)} event groups")

        # Create unified result
        result = UIEResult(
            text=text,
            entities=entities,
            relations=relations,
            events=events
        )

        logger.info(f"Completed extraction: {len(entities)} entities, {len(relations)} relations, {len(events)} events")

        return result

    def extract_batch(self,
                      texts: List[str],
                      task: str,
                      user_schema: Optional[Dict[str, List[str]]] = None,
                      mode: str = "flexible",
                      batch_size: int = 8,
                      **gen_config) -> List[Union[List[NER], List[RE], List[EE]]]:
        """
        Extract information from a batch of texts

        Args:
            texts: List of input texts
            task: Task type ('NER', 'RE', 'EE')
            user_schema: Optional user-provided schema
            mode: Extraction mode
            batch_size: Batch size for processing
            **gen_config: Additional generation parameters

        Returns:
            List of extraction results
        """
        if task not in ['NER', 'RE', 'EE']:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are 'NER', 'RE', 'EE'.")

        if not texts:
            logger.warning("Empty text list provided")
            return []

        logger.info(f"Processing batch of {len(texts)} texts for task {task}")

        results = []

        prompt = self.prompter.create_prompt(task, '', user_schema, mode)

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            # Create prompts for batch
            formatted_prompts = [
                self.format_chat_template(prompt, text) for text in batch_texts
            ]
            responses = self._generate_batch_response(
                formatted_prompts,
                **gen_config
            )
            results.extend(responses)

        if task == 'NER':
            results = [self.parser.parse_ner(response, text) for response, text in zip(results, texts)]
        elif task == 'RE':
            # For RE, we might need entities first for better parsing
            entities = []
            results = [self.parser.parse_re(response, entities) for response in results]
        elif task == 'EE':
            entities = []
            results = [self.parser.parse_ee(response, entities) for response in results]

        logger.info(f"Completed batch processing: {len(results)} results")
        return results

    def set_generation_config(self, **config):
        """Update generation configuration"""
        self.generation_config.update(config)
        logger.info(f"Updated generation config: {config}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.model is not None,
            'generation_config': self.generation_config
        }

        if self.model is not None:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['model_dtype'] = next(self.model.parameters()).dtype

        return info


def clean_predictions(predictions):
    cleaned = []
    for pred in predictions:
        cleaned = pred.strip()
        if cleaned.lower().startswith('assistant'):
            cleaned = cleaned[9:].strip()  # Remove "assistant"

        while cleaned.lower().startswith('assistant'):
            cleaned = cleaned[9:].strip()
        cleaned = cleaned.lstrip('\n\r\t ')

        # Remove any remaining role markers
        role_markers = ['user:', 'assistant:', 'system:', '<|assistant|>', '<|user|>', '<|system|>']
        for marker in role_markers:
            if cleaned.lower().startswith(marker.lower()):
                cleaned = cleaned[len(marker):].strip()

        # Remove space and puncation in firt
        cleaned = re.sub(r'^[\s\W]+', '', cleaned)
        cleaned.append(cleaned)

    return cleaned
