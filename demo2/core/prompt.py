from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from enum import Enum
from ..utils import NER_SCHEMA, RE_SCHEMA, EET_SCHEMA, EEA_SCHEMA


class ExtractionMode(Enum):
    OPEN = "open"
    STRICT = "strict"
    FLEXIBLE = "flexible"


class PromptStrategy(ABC):
    """Abstract strategy for prompt generation"""

    @abstractmethod
    def create_prompt(self, task: str, text: str, labels: Optional[List[str]] = None) -> str:
        pass


class SchemaBasedPromptStrategy(PromptStrategy):
    """Strategy for schema-based prompts"""

    PROMPTS = {
        'NER': """Extract entities from the text **strictly using ONLY the provided Entity List** below and **MUST** strictly adhere to the output format.
Format output as '<entity tag>: <entity name>' and separated multiple entities by '|'. Return 'None' if no entities are identified.
Entity List: {labels}
Text: {text}""",

        'RE': """Extract relationships between entities in text **strictly using ONLY the provided Relationship List** below and **MUST** strictly adhere to the output format.
Format each relationship as '<relation_type>: <head_entity>, <tail_entity>' and separated multiple relationship by '|'. Return 'None' if no relationships are identified.
Relationship List: {labels}
Text: {text}""",

        'EE': """Extract events and their components from text **strictly using ONLY the provided Event List** below and **MUST** strictly adhere to the output format.
Format output as '<event_type>: <trigger_word> | <role1>: <argument1> | <role2>: <argument2>' and separate multiple events with '|'. Return 'None' if no events are identified.
Event List: {labels}
Text: {text}"""
    }

    def create_prompt(self, task: str, text: str, labels: Optional[List[str]] = None) -> str:
        template = self.PROMPTS.get(task, "")
        labels_text = ", ".join(labels) if labels else ""
        return template.format(labels=labels_text, text=text)


class OpenPromptStrategy(PromptStrategy):
    """Strategy for open extraction prompts"""

    PROMPTS = {
        'NER': """Extract all named entities from the text. Format output as '<entity tag>: <entity name>' and separated multiple entities by '|'. Return 'None' if no entities are identified.
Text: {text}""",

        'RE': """Extract all relationships between entities in text. Format each relationship as '<relation_type>: <head_entity>, <tail_entity>' and separated multiple relationship by '|'. Return 'None' if no relationships are identified.
Text: {text}""",

        'EE': """Extract all events and their components from text. Format output as '<event_type>: <trigger_word> | <role1>: <argument1> | <role2>: <argument2>' and separate multiple events with '|'. Return 'None' if no events are identified.
Text: {text}"""
    }

    def create_prompt(self, task: str, text: str, labels: Optional[List[str]] = None) -> str:
        template = self.PROMPTS.get(task, "")
        return template.format(text=text)


class SchemaManager:
    """Manages schema resolution strategies"""

    DEFAULT_SCHEMAS = {
        'NER': {'entity_types': NER_SCHEMA},
        'RE': {'relation_types': RE_SCHEMA},
        'EE': {'event_types': EET_SCHEMA, 'argument_types': EEA_SCHEMA}
    }

    @classmethod
    def resolve_schema(cls, task: str, user_schema: Optional[Dict[str, List[str]]], mode: ExtractionMode) -> Optional[
        List[str]]:
        """Resolve schema based on mode and inputs"""

        resolver_map = {
            ExtractionMode.STRICT: cls._resolve_strict,
            ExtractionMode.FLEXIBLE: cls._resolve_flexible,
            ExtractionMode.OPEN: lambda *args: None
        }

        resolver = resolver_map.get(mode, cls._resolve_flexible)
        return resolver(task, user_schema)

    @classmethod
    def _resolve_strict(cls, task: str, user_schema: Optional[Dict[str, List[str]]]) -> Optional[List[str]]:
        """Strict mode: only user schema"""
        if not user_schema:
            return None

        schema_key_map = {
            'NER': 'entity_types',
            'RE': 'relation_types',
            'EE': cls._get_ee_labels
        }

        resolver = schema_key_map.get(task)
        if callable(resolver):
            return resolver(user_schema)
        elif resolver and resolver in user_schema:
            return user_schema[resolver]

        return None

    @classmethod
    def _resolve_flexible(cls, task: str, user_schema: Optional[Dict[str, List[str]]]) -> Optional[List[str]]:
        """Flexible mode: user schema + default fallback"""
        user_labels = cls._resolve_strict(task, user_schema)
        if user_labels:
            return user_labels

        # Fallback to default
        default_schema = cls.DEFAULT_SCHEMAS.get(task)
        if task == 'EE':
            return cls._get_ee_labels(default_schema)
        elif default_schema:
            key = 'entity_types' if task == 'NER' else 'relation_types'
            return default_schema.get(key)

        return None

    @classmethod
    def _get_ee_labels(cls, schema: Dict[str, List[str]]) -> List[str]:
        """Get EE labels combining event types and argument types"""
        ee_labels = []
        if 'event_types' in schema:
            ee_labels.extend([f"Event: {et}" for et in schema['event_types']])
        if 'argument_types' in schema:
            ee_labels.extend([f"Argument: {at}" for at in schema['argument_types']])
        return ee_labels


class PromptManager:
    def __init__(self):
        self.strategies = {
            ExtractionMode.OPEN: OpenPromptStrategy(),
            ExtractionMode.STRICT: SchemaBasedPromptStrategy(),
            ExtractionMode.FLEXIBLE: SchemaBasedPromptStrategy()
        }
        self.schema_manager = SchemaManager()

    def create_prompt(self, task: str, text: str,
                      user_schema: Optional[Dict[str, List[str]]] = None,
                      mode: str = "flexible") -> str:
        """Create prompt for UIE task"""

        extraction_mode = ExtractionMode(mode.lower())
        strategy = self.strategies[extraction_mode]

        # Resolve schema
        labels = self.schema_manager.resolve_schema(task, user_schema, extraction_mode)

        return strategy.create_prompt(task, text, labels)

    def get_prompt_info(self, task: str, mode: str) -> Dict[str, str]:
        """Get information about prompt strategy"""

        info_map = {
            ExtractionMode.OPEN: {
                "strategy": "Open extraction - no predefined schema",
                "prompt_type": "open_extraction"
            },
            ExtractionMode.STRICT: {
                "strategy": "Strict - only user-provided schema",
                "prompt_type": "trained_with_schema"
            },
            ExtractionMode.FLEXIBLE: {
                "strategy": "Flexible - user schema + default fallback",
                "prompt_type": "trained_with_schema"
            }
        }

        extraction_mode = ExtractionMode(mode.lower())
        return info_map.get(extraction_mode, info_map[ExtractionMode.FLEXIBLE])