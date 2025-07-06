"""
JSON-based parsers for NER, RE, and EE tasks
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from logging import getLogger
import re
import json
from extract import NER, RE, EEA, EET, EE, UIEResult

logger = getLogger(__name__)
logger.debug("parsers.py loaded")


class TaskParser:
    """JSON-based parser functions for different tasks"""

    @staticmethod
    def parse_ner(text: str, original_text: str = "") -> List[NER]:
        """
        Parse NER output format to JSON structure

        Args:
            text: NER prediction string
            original_text: Original input text for position finding

        Returns:
            List of entity dictionaries with JSON structure
        """
        if not text or text.strip().lower() == 'none' or text is None:
            return []

        entities = []
        parts = text.split(' | ')

        for i, part in enumerate(parts):
            part = part.strip()
            if ':' in part:
                try:
                    entity_type, entity_mention = part.split(':', 1)
                    entity_type = entity_type.strip()
                    entity_mention = entity_mention.strip()

                    if entity_type and entity_mention:
                        # Find positions in original text
                        # start_pos, end_pos = JSONTaskParsers._find_entity_positions(entity_mention)
                        #     entity_mention, original_text
                        # )

                        entity = NER(
                            entity_type=entity_type,
                            entity_mention=entity_mention,
                            confidence=0.0,
                            start=None,
                            end=None
                        )
                        entities.append(entity)

                except ValueError:
                    logger.warning(f"Failed to parse NER part: {part}")
                    continue

        return entities

    @staticmethod
    def parse_re(text: str, entities: Optional[List[NER]] = None) -> List[RE]:
        """
        Parse RE output format and return RE objects

        Args:
            text: RE prediction string
            entities: List of NER entities for validation

        Returns:
            List of RE objects
        """
        if not text or text.strip().lower() in ['none', 'nan'] or text is None:
            return []

        relations = []
        parts = text.split('|')

        # Create entity mapping for validation if provided
        entity_map = {}
        if entities:
            entity_map = {ent.entity_mention.lower(): ent for ent in entities}

        for i, part in enumerate(parts):
            part = part.strip()
            if ':' in part and ', ' in part:
                try:
                    relation_type, relation_part = part.split(':', 1)
                    relation_type = relation_type.strip()

                    if ', ' in relation_part:
                        head_entity, tail_entity = relation_part.split(', ', 1)
                        # Clean whitespace and punctuation
                        head_entity = head_entity.strip(" ,.!?;:")
                        tail_entity = tail_entity.strip(" ,.!?;:")

                        if relation_type and head_entity and tail_entity:
                            relation = RE(
                                relation_type=relation_type,
                                head_entity=head_entity,
                                tail_entity=tail_entity,
                                confidence=0.0
                            )
                            relations.append(relation)

                except ValueError:
                    logger.warning(f"Failed to parse RE part: {part}")
                    continue

        return relations

    @staticmethod
    def parse_ee(text: str, entities: Optional[List[NER]] = None) -> List[EE]:
        """
        Parse EE output format and return EE objects

        Args:
            text: EE prediction string
            entities: List of NER entities for validation

        Returns:
            List of EE objects
        """
        if not text or text.strip().lower() == 'none' or text is None:
            return []

        events = []
        event_parts = text.split('||')

        # Create entity mapping
        entity_map = {}
        if entities:
            entity_map = {ent.entity_mention.lower(): ent for ent in entities}

        for event_idx, event_part in enumerate(event_parts):
            event_part = event_part.strip()
            if not event_part:
                continue

            components = event_part.split('|')
            if len(components) < 1:
                continue

            # Parse event type and trigger
            first_component = components[0].strip()
            if ':' not in first_component:
                continue

            try:
                event_type, trigger = first_component.split(':', 1)
                event_type = event_type.strip()
                trigger = trigger.strip()

                # Create event trigger
                eet = EET(
                    trigger=trigger,
                    trigger_type=event_type,
                    confidence=0.0
                )

                # Parse arguments
                for comp in components[1:]:
                    comp = comp.strip()
                    if ':' in comp:
                        role, argument = comp.split(':', 1)
                        role = role.strip()
                        argument = argument.strip()

                        if role and argument:
                            # Create argument object
                            eea = EEA(role=role, entity=argument)
                            eet.arguments.append(eea)

                # Create event group
                if event_type and trigger:
                    ee = EE()
                    ee.events.append(eet)
                    events.append(ee)

            except ValueError:
                logger.warning(f"Failed to parse event part: {event_part}")
                continue

        return events

    @staticmethod
    def parse_all_tasks(ner_text: str,
                        re_text: str,
                        ee_text: str,
                        original_text: str = "") -> UIEResult:
        """
        Parse all tasks and return unified UIEResult object

        Args:
            ner_text: NER prediction string
            re_text: RE prediction string
            ee_text: EE prediction string
            original_text: Original input text

        Returns:
            UIEResult object with all extractions
        """
        # Parse entities first
        entities = TaskParser.parse_ner(ner_text, original_text)

        # Parse relations with entity context
        relations = TaskParser.parse_re(re_text, entities)

        # Parse events with entity context
        events = TaskParser.parse_ee(ee_text, entities)

        # Create unified result
        result = UIEResult(
            text=original_text,
            entities=entities,
            relations=relations,
            events=events
        )

        return result

    @staticmethod
    def _find_entity_positions(entity_text: str, full_text: str) -> Tuple[int, int]:
        """
        Find start and end positions of entity in text

        Args:
            entity_text: Entity text to find
            full_text: Full text to search in

        Returns:
            Tuple of (start, end) positions
        """
        if not full_text:
            return 0, len(entity_text)

        try:
            # Case-insensitive exact match
            start = full_text.lower().find(entity_text.lower())
            if start != -1:
                return start, start + len(entity_text)

            # Fuzzy matching for partial matches
            import difflib

            words = full_text.split()
            entity_words = entity_text.split()

            for i in range(len(words) - len(entity_words) + 1):
                candidate = ' '.join(words[i:i + len(entity_words)])
                similarity = difflib.SequenceMatcher(
                    None, entity_text.lower(), candidate.lower()
                ).ratio()

                if similarity > 0.8:  # 80% similarity threshold
                    start_pos = full_text.lower().find(candidate.lower())
                    if start_pos != -1:
                        return start_pos, start_pos + len(candidate)

            # Return default positions if not found
            return 0, len(entity_text)

        except Exception as e:
            logger.debug(f"Error finding entity positions: {e}")
            return 0, len(entity_text)


if __name__ == "__main__":
    # Sample outputs from your fine-tuned models
    ner_output = "PERSON: John Smith | ORGANIZATION: Apple Inc | LOCATION: New York"
    re_output = "works_for: John Smith -> Apple Inc | located_in: Apple Inc -> New York"
    ee_output = "Employment: hired | employee: John Smith | employer: Apple Inc"
    original_text = "John Smith was hired by Apple Inc in New York yesterday."

    # Parse with JSON format
    json_result = TaskParser.parse_all_tasks(
        ner_output, re_output, ee_output, original_text
    )
    ner = json_result.entities
    re = json_result.relations
    ee = json_result.events

    print(
        "\nJSON Format Result:"
    )
    for entity in ner:
        print(entity.to_dict())

    for relation in re:
        print(relation.to_dict())
    for event in ee:
        print(event.to_dict())
