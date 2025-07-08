from typing import List, Optional, Callable, Dict
import logging
from abc import ABC, abstractmethod
from extract import NER, RE, EEA, EET, EE

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Base parser interface"""

    @abstractmethod
    def parse(self, text: str, *args, **kwargs):
        pass

    def _is_empty_result(self, text: str) -> bool:
        """Check if text represents empty result"""
        return not text or text.strip().lower() in ['none', 'nan']


class NERParser(BaseParser):
    """Parser for NER results"""

    def parse(self, text: str, original_text: str = "") -> List[NER]:
        if self._is_empty_result(text):
            return []

        entities = []
        parts = text.split(' | ')

        for part in parts:
            entity = self._parse_entity_part(part.strip())
            if entity:
                entities.append(entity)

        return entities

    def _parse_entity_part(self, part: str) -> Optional[NER]:
        """Parse single entity part"""
        if ':' not in part:
            return None

        try:
            entity_type, entity_mention = part.split(':', 1)
            entity_type = entity_type.strip()
            entity_mention = entity_mention.strip()

            if entity_type and entity_mention:
                return NER(entity_type=entity_type, entity_mention=entity_mention)
        except ValueError:
            logger.warning(f"Failed to parse NER part: {part}")

        return None


class REParser(BaseParser):
    """Parser for RE results"""

    def parse(self, text: str, entities: Optional[List[NER]] = None) -> List[RE]:
        if self._is_empty_result(text):
            return []

        relations = []
        parts = text.split('|')

        for part in parts:
            relation = self._parse_relation_part(part.strip())
            if relation:
                relations.append(relation)

        return relations

    def _parse_relation_part(self, part: str) -> Optional[RE]:
        """Parse single relation part"""
        if ':' not in part or ', ' not in part:
            return None

        try:
            relation_type, relation_part = part.split(':', 1)
            relation_type = relation_type.strip()

            head_entity, tail_entity = relation_part.split(', ', 1)
            head_entity = head_entity.strip(" ,.!?;:")
            tail_entity = tail_entity.strip(" ,.!?;:")

            if relation_type and head_entity and tail_entity:
                return RE(
                    relation_type=relation_type,
                    head_entity=head_entity,
                    tail_entity=tail_entity
                )
        except ValueError:
            logger.warning(f"Failed to parse RE part: {part}")

        return None


class EEParser(BaseParser):
    """Parser for EE results"""

    def parse(self, text: str, entities: Optional[List[NER]] = None) -> List[EE]:
        if self._is_empty_result(text):
            return []

        events = []
        event_parts = text.split('||')

        for event_part in event_parts:
            event = self._parse_event_part(event_part.strip())
            if event:
                events.append(event)

        return events

    def _parse_event_part(self, event_part: str) -> Optional[EE]:
        """Parse single event part"""
        if not event_part:
            return None

        components = event_part.split('|')
        if len(components) < 1:
            return None

        # Parse event trigger
        trigger_info = self._parse_trigger(components[0].strip())
        if not trigger_info:
            return None

        event_type, trigger = trigger_info
        eet = EET(trigger=trigger, trigger_type=event_type)

        # Parse arguments
        for comp in components[1:]:
            argument = self._parse_argument(comp.strip())
            if argument:
                eet.arguments.append(argument)

        ee = EE()
        ee.events.append(eet)
        return ee

    def _parse_trigger(self, component: str) -> Optional[tuple]:
        """Parse trigger component"""
        if ':' not in component:
            return None

        try:
            event_type, trigger = component.split(':', 1)
            event_type = event_type.strip()
            trigger = trigger.strip()

            if event_type and trigger:
                return (event_type, trigger)
        except ValueError:
            logger.warning(f"Failed to parse trigger: {component}")

        return None

    def _parse_argument(self, component: str) -> Optional[EEA]:
        """Parse argument component"""
        if ':' not in component:
            return None

        try:
            role, argument = component.split(':', 1)
            role = role.strip()
            argument = argument.strip()

            if role and argument:
                return EEA(role=role, entity=argument)
        except ValueError:
            logger.warning(f"Failed to parse argument: {component}")

        return None


class ParserFactory:
    """Factory for creating parsers"""

    _parsers = {
        'NER': NERParser(),
        'RE': REParser(),
        'EE': EEParser()
    }

    @classmethod
    def get_parser(cls, task: str) -> BaseParser:
        """Get parser for task"""
        return cls._parsers.get(task.upper())

    @classmethod
    def parse(cls, task: str, text: str, *args, **kwargs):
        """Parse using appropriate parser"""
        parser = cls.get_parser(task)
        if not parser:
            raise ValueError(f"No parser found for task: {task}")

        return parser.parse(text, *args, **kwargs)


class TaskParser:
    """Unified parser interface"""

    def __init__(self):
        self.factory = ParserFactory()

    def parse_ner(self, text: str, original_text: str = "") -> List[NER]:
        return self.factory.parse('NER', text, original_text)

    def parse_re(self, text: str, entities: Optional[List[NER]] = None) -> List[RE]:
        return self.factory.parse('RE', text, entities)

    def parse_ee(self, text: str, entities: Optional[List[NER]] = None) -> List[EE]:
        return self.factory.parse('EE', text, entities)