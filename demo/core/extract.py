from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class NER:
    """Named Entity Recognition result"""
    entity_type: str
    entity_mention: str
    confidence: float = 0.9
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_type': self.entity_type,
            'entity_mention': self.entity_mention,
            'confidence': self.confidence,
            'start': self.start,
            'end': self.end
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return f"{self.entity_type}: {self.entity_mention}"


@dataclass
class RE:
    """Relation Extraction result"""
    relation_type: str
    head_entity: str
    tail_entity: str
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'relation_type': self.relation_type,
            'head_entity': self.head_entity,
            'tail_entity': self.tail_entity,
            'confidence': self.confidence
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return f"{self.relation_type}: {self.head_entity} -> {self.tail_entity}"


@dataclass
class EEA:
    """Event Argument"""
    role: str
    entity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'role': self.role,
            'entity': self.entity
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return f"{self.role}: {self.entity}"


@dataclass
class EET:
    """Event Trigger"""
    trigger: str
    trigger_type: str
    arguments: List[EEA] = field(default_factory=list)
    confidence: float = 0.9

    def add_argument(self, role: str, entity: str):
        """Add an argument to this event"""
        self.arguments.append(EEA(role, entity))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trigger': self.trigger,
            'trigger_type': self.trigger_type,
            'arguments': [arg.to_dict() for arg in self.arguments],
            'confidence': self.confidence
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        args_str = " | ".join(str(arg) for arg in self.arguments)
        return f"{self.trigger_type}: {self.trigger} | {args_str}"


@dataclass
class EE:
    """Event Extraction result"""
    events: List[EET] = field(default_factory=list)

    def add_event(self, trigger: str, trigger_type: str) -> EET:
        """Add a new event and return it"""
        event = EET(trigger, trigger_type)
        self.events.append(event)
        return event

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'events': [event.to_dict() for event in self.events]
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return " || ".join(str(event) for event in self.events)


@dataclass
class UIEResult:
    """Complete UIE extraction result"""
    text: str
    entities: List[NER] = field(default_factory=list)
    relations: List[RE] = field(default_factory=list)
    events: List[EE] = field(default_factory=list)

    def add_entity(self, entity_type: str, entity_mention: str, **kwargs) -> NER:
        """Add entity and return it"""
        entity = NER(entity_type, entity_mention, **kwargs)
        self.entities.append(entity)
        return entity

    def add_relation(self, relation_type: str, head_entity: str, tail_entity: str, **kwargs) -> RE:
        """Add relation and return it"""
        relation = RE(relation_type, head_entity, tail_entity, **kwargs)
        self.relations.append(relation)
        return relation

    def add_event_group(self) -> EE:
        """Add new event group and return it"""
        event_group = EE()
        self.events.append(event_group)
        return event_group

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'events': [event.to_dict() for event in self.events],
            'metadata': {
                'entity_count': len(self.entities),
                'relation_count': len(self.relations),
                'event_count': sum(len(ee.events) for ee in self.events),
                'entity_types': list(set(ent.entity_type for ent in self.entities)),
                'relation_types': list(set(rel.relation_type for rel in self.relations)),
                'event_types': list(set(eet.trigger_type for ee in self.events for eet in ee.events))
            }
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.to_dict()['metadata']