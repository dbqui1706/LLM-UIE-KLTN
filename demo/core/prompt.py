from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..utils.schema import NER_SCHEMA, RE_SCHEMA, EET_SCHEMA, EEA_SCHEMA
from utils import BASE_SYSTEM_PROMPTS, SCHEMA_SYSTEM_PROMPTS


@dataclass
class PromptConfig:
    include_schema: bool = True
    use_examples: bool = False
    strict_schema_only: bool = False  # Chỉ extract types trong schema
    allow_custom_types: bool = True  # Cho phép extract thêm types khác


class PromptGenerator:
    """Generate prompts for UIE tasks with flexible schema support"""
    @staticmethod
    def generate_prompt(task: str,
                        text: str,
                        schema: Optional[Dict[str, List[str]]] = None,
                        config: PromptConfig = None) -> str:
        """
        Generate complete prompt for UIE task

        Args:
            task: Task type ('NER', 'RE', 'EE')
            text: Input text
            schema: Optional schema with entity/relation/event types
            config: Prompt configuration

        Returns:
            Complete prompt string
        """
        if config is None:
            config = PromptConfig()

        # Choose base prompt
        if schema and config.include_schema:
            system_prompt = PromptGenerator.SCHEMA_SYSTEM_PROMPTS.get(task, "")
        else:
            system_prompt = PromptGenerator.BASE_SYSTEM_PROMPTS.get(task, "")

        # Build prompt parts
        prompt_parts = [system_prompt]

        # Add schema if provided and enabled
        if schema and config.include_schema:
            schema_text = PromptGenerator._format_schema(task, schema, config)
            if schema_text:
                prompt_parts.append(schema_text)

        # Add examples if enabled
        if config.use_examples:
            examples = PromptGenerator._get_examples(task)
            if examples:
                prompt_parts.append(examples)

        return "\n".join(prompt_parts)

    @staticmethod
    def _format_schema(task: str,
                       schema: Dict[str, List[str]],
                       config: PromptConfig) -> str:

        if task == "NER":
            entity_types = schema.get('entity_types', [])
            if entity_types:
                schema_text = f"\nEntity types to extract: {', '.join(entity_types)}"
                if not config.strict_schema_only:
                    schema_text += "\nYou may also extract other relevant entity types if found."
                return schema_text

        elif task == "RE":
            relation_types = schema.get('relation_types', [])
            if relation_types:
                schema_text = f"\nRelation types to extract: {', '.join(relation_types)}"
                if not config.strict_schema_only:
                    schema_text += "\nYou may also extract other relevant relation types if found."
                return schema_text

        elif task == "EE":
            event_types = schema.get('event_types', [])
            argument_types = schema.get('argument_types', [])

            schema_parts = []
            if event_types:
                schema_parts.append(f"Event types: {', '.join(event_types)}")
            if argument_types:
                schema_parts.append(f"Argument types: {', '.join(argument_types)}")

            if schema_parts:
                schema_text = f"\n{'. '.join(schema_parts)}"
                if not config.strict_schema_only:
                    schema_text += "\nYou may also extract other relevant event types and arguments if found."
                return schema_text

        return ""

    @staticmethod
    def _get_examples(task: str) -> str:
        examples = {
            "NER": """
Example:
Text: "John Smith works at Apple Inc in Cupertino, California."
Extracted information: PERSON: John Smith | ORGANIZATION: Apple Inc | LOCATION: Cupertino | LOCATION: California""",

            "RE": """
Example:
Text: "John Smith works at Apple Inc in Cupertino, California."
Extracted information: works_for: John Smith -> Apple Inc | located_in: Apple Inc -> Cupertino""",

            "EE": """
Example:
Text: "Apple Inc hired John Smith as a software engineer yesterday."
Extracted information: Personnel: hired | employee: John Smith | employer: Apple Inc | position: software engineer"""
        }
        return examples.get(task, "")


class PromptManager:
    def __init__(self):
        self.default_schemas = {
            'NER': {'entity_types': NER_SCHEMA},
            'RE': {'relation_types': RE_SCHEMA},
            'EE': {'event_types': EET_SCHEMA, 'argument_types': EEA_SCHEMA}
        }

    def create_prompt(self,
                      task: str,
                      text: str,
                      user_schema: Optional[Dict[str, List[str]]] = None,
                      mode: str = "flexible") -> str:

        # Debug log
        print(f"🔧 Creating prompt for task: {task}")
        print(f"🔧 User schema: {user_schema}")
        print(f"🔧 Mode: {mode}")

        config = PromptConfig()

        if mode == "strict":
            config.include_schema = True
            config.strict_schema_only = True
            config.allow_custom_types = False
        elif mode == "flexible":
            config.include_schema = True
            config.strict_schema_only = False
            config.allow_custom_types = True
        elif mode == "open":
            config.include_schema = False
            config.allow_custom_types = True

        # Use user schema or default
        schema = user_schema or self.default_schemas.get(task)

        print(f"🔧 Final schema used: {schema}")

        prompt = PromptGenerator.generate_prompt(task, text, schema, config)
        print(f"🔧 Generated prompt: {prompt[:300]}...")

        return prompt

    def create_batch_prompts(self,
                             task: str,
                             texts: List[str],
                             user_schema: Optional[Dict[str, List[str]]] = None,
                             mode: str = "flexible") -> List[str]:
        return [
            self.create_prompt(task, text, user_schema, mode)
            for text in texts
        ]
