# app/utils/data_extractor.py
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ExtractionDataExtractor:
    """Smart extractor for getting entities, relations, events from various result formats"""
    
    @staticmethod
    def extract_uie_data(data: Any) -> Dict[str, List[Dict]]:
        """
        Extract entities, relations, events from various UIE result formats
        Returns normalized format: {'entities': [...], 'relations': [...], 'events': [...]}
        """
        
        if not data:
            logger.warning("No data provided for extraction")
            return {'entities': [], 'relations': [], 'events': []}
        
        logger.info(f"🔍 Extracting UIE data from: {type(data)} - {str(data)[:200]}...")
        
        try:
            # Case 1: Already normalized format
            if isinstance(data, dict) and all(key in data for key in ['entities', 'relations', 'events']):
                logger.info("📊 Found pre-normalized format")
                return ExtractionDataExtractor._normalize_direct_format(data)
            
            # Case 2: Document extraction results (with aggregated_results)
            if isinstance(data, dict) and 'aggregated_results' in data:
                logger.info("📄 Found document extraction format with aggregated_results")
                return ExtractionDataExtractor._extract_from_aggregated(data['aggregated_results'])
            
            # Case 3: Text extraction results (single task format)
            if isinstance(data, dict):
                logger.info("📝 Found text extraction format")
                return ExtractionDataExtractor._extract_from_text_format(data)
            
            # Case 4: List format (direct from model)
            if isinstance(data, list):
                logger.info("📋 Found list format")
                return ExtractionDataExtractor._extract_from_list_format(data)
            
            logger.warning(f"⚠️ Unknown data format: {type(data)}")
            return {'entities': [], 'relations': [], 'events': []}
            
        except Exception as e:
            logger.error(f"❌ Error extracting UIE data: {e}")
            return {'entities': [], 'relations': [], 'events': []}
    
    @staticmethod
    def _normalize_direct_format(data: Dict) -> Dict[str, List[Dict]]:
        """Normalize already correct format"""
        result = {
            'entities': ExtractionDataExtractor._normalize_entity_list(data.get('entities', [])),
            'relations': ExtractionDataExtractor._normalize_relation_list(data.get('relations', [])),
            'events': ExtractionDataExtractor._normalize_event_list(data.get('events', []))
        }
        
        logger.info(f"✅ Normalized direct format: {len(result['entities'])} entities, {len(result['relations'])} relations, {len(result['events'])} events")
        return result
    
    @staticmethod
    def _extract_from_aggregated(aggregated_data: Dict) -> Dict[str, List[Dict]]:
        """Extract from aggregated_results format"""
        if not isinstance(aggregated_data, dict):
            logger.warning("Aggregated data is not a dictionary")
            return {'entities': [], 'relations': [], 'events': []}
        
        result = {
            'entities': ExtractionDataExtractor._normalize_entity_list(aggregated_data.get('entities', [])),
            'relations': ExtractionDataExtractor._normalize_relation_list(aggregated_data.get('relations', [])),
            'events': ExtractionDataExtractor._normalize_event_list(aggregated_data.get('events', []))
        }
        
        logger.info(f"✅ Extracted from aggregated: {len(result['entities'])} entities, {len(result['relations'])} relations, {len(result['events'])} events")
        return result
    
    @staticmethod
    def _extract_from_text_format(data: Dict) -> Dict[str, List[Dict]]:
        """Extract from text extraction format (single task)"""
        result = {'entities': [], 'relations': [], 'events': []}
        
        # Check for each possible key
        if 'entities' in data:
            result['entities'] = ExtractionDataExtractor._normalize_entity_list(data['entities'])
        
        if 'relations' in data:
            result['relations'] = ExtractionDataExtractor._normalize_relation_list(data['relations'])
        
        if 'events' in data:
            result['events'] = ExtractionDataExtractor._normalize_event_list(data['events'])
        
        # Handle single task results (when task != ALL)
        for key, value in data.items():
            if key in ['entities', 'relations', 'events'] and isinstance(value, list):
                result[key] = ExtractionDataExtractor._normalize_list_by_type(value, key)
        
        logger.info(f"✅ Extracted from text format: {len(result['entities'])} entities, {len(result['relations'])} relations, {len(result['events'])} events")
        return result
    
    @staticmethod
    def _extract_from_list_format(data: List) -> Dict[str, List[Dict]]:
        """Extract from list format (direct model output)"""
        # This would be used if model returns raw list
        # Need to determine what type of list it is
        result = {'entities': [], 'relations': [], 'events': []}
        
        if data and isinstance(data[0], dict):
            # Try to determine type from first item
            first_item = data[0]
            if 'entity_mention' in first_item or 'entity_type' in first_item:
                result['entities'] = ExtractionDataExtractor._normalize_entity_list(data)
            elif 'relation_type' in first_item or 'head_entity' in first_item:
                result['relations'] = ExtractionDataExtractor._normalize_relation_list(data)
            elif 'trigger' in first_item or 'trigger_type' in first_item:
                result['events'] = ExtractionDataExtractor._normalize_event_list(data)
        
        logger.info(f"✅ Extracted from list format: {len(result['entities'])} entities, {len(result['relations'])} relations, {len(result['events'])} events")
        return result
    
    @staticmethod
    def _normalize_entity_list(entities: List) -> List[Dict]:
        """Normalize entity list to standard format"""
        if not entities:
            return []
        
        normalized = []
        for entity in entities:
            if isinstance(entity, dict):
                # Standard format check
                if 'entity_mention' in entity and 'entity_type' in entity:
                    normalized.append({
                        'entity_type': entity.get('entity_type', 'UNKNOWN'),
                        'entity_mention': entity.get('entity_mention', ''),
                        'confidence': entity.get('confidence', 1.0)
                    })
                else:
                    # Try alternative formats
                    mention = entity.get('mention', entity.get('text', entity.get('name', '')))
                    entity_type = entity.get('type', entity.get('label', entity.get('entity_type', 'UNKNOWN')))
                    
                    if mention:
                        normalized.append({
                            'entity_type': entity_type,
                            'entity_mention': mention,
                            'confidence': entity.get('confidence', 1.0)
                        })
        
        logger.debug(f"Normalized {len(normalized)} entities")
        return normalized
    
    @staticmethod
    def _normalize_relation_list(relations: List) -> List[Dict]:
        """Normalize relation list to standard format"""
        if not relations:
            return []
        
        normalized = []
        for relation in relations:
            if isinstance(relation, dict):
                # Standard format check
                if all(key in relation for key in ['relation_type', 'head_entity', 'tail_entity']):
                    normalized.append({
                        'relation_type': relation.get('relation_type', 'UNKNOWN'),
                        'head_entity': relation.get('head_entity', ''),
                        'tail_entity': relation.get('tail_entity', ''),
                        'confidence': relation.get('confidence', 1.0)
                    })
                else:
                    # Try alternative formats
                    rel_type = relation.get('type', relation.get('label', relation.get('relation_type', 'UNKNOWN')))
                    head = relation.get('head', relation.get('source', relation.get('head_entity', '')))
                    tail = relation.get('tail', relation.get('target', relation.get('tail_entity', '')))
                    
                    if head and tail:
                        normalized.append({
                            'relation_type': rel_type,
                            'head_entity': head,
                            'tail_entity': tail,
                            'confidence': relation.get('confidence', 1.0)
                        })
        
        logger.debug(f"Normalized {len(normalized)} relations")
        return normalized
    
    @staticmethod
    def _normalize_event_list(events: List) -> List[Dict]:
        """Normalize event list to standard format"""
        if not events:
            return []
        
        normalized = []
        for event_item in events:
            if isinstance(event_item, dict):
                # Check if it's an event group (contains 'events' key)
                if 'events' in event_item:
                    for event in event_item['events']:
                        normalized.append(ExtractionDataExtractor._normalize_single_event(event))
                else:
                    # Single event
                    normalized.append(ExtractionDataExtractor._normalize_single_event(event_item))
        
        logger.debug(f"Normalized {len(normalized)} events")
        return normalized
    
    @staticmethod
    def _normalize_single_event(event: Dict) -> Dict:
        """Normalize single event to standard format"""
        if not isinstance(event, dict):
            return {}
        
        # Standard format
        trigger = event.get('trigger', event.get('trigger_word', event.get('text', '')))
        trigger_type = event.get('trigger_type', event.get('event_type', event.get('type', 'UNKNOWN')))
        arguments = event.get('arguments', [])
        
        # Normalize arguments
        normalized_arguments = []
        for arg in arguments:
            if isinstance(arg, dict):
                normalized_arguments.append({
                    'role': arg.get('role', arg.get('type', 'UNKNOWN')),
                    'entity': arg.get('entity', arg.get('text', arg.get('mention', ''))),
                })
        
        return {
            'trigger': trigger,
            'trigger_type': trigger_type,
            'arguments': normalized_arguments,
            'confidence': event.get('confidence', 1.0)
        }
    
    @staticmethod
    def _normalize_list_by_type(data: List, data_type: str) -> List[Dict]:
        """Normalize list based on expected type"""
        if data_type == 'entities':
            return ExtractionDataExtractor._normalize_entity_list(data)
        elif data_type == 'relations':
            return ExtractionDataExtractor._normalize_relation_list(data)
        elif data_type == 'events':
            return ExtractionDataExtractor._normalize_event_list(data)
        else:
            return []


# Convenience function
def extract_uie_data(data: Any) -> Dict[str, List[Dict]]:
    """Convenience function to extract UIE data"""
    return ExtractionDataExtractor.extract_uie_data(data)