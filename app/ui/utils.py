# app/ui/utils.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def prepare_schema(entity_types: str, relation_types: str, event_types: str, argument_types: str) -> Dict[
    str, List[str]]:
    """Prepare user schema from input strings"""
    schema = {}

    for key, value in [
        ('entity_types', entity_types),
        ('relation_types', relation_types),
        ('event_types', event_types),
        ('argument_types', argument_types)
    ]:
        if value and value.strip():
            schema[key] = [t.strip() for t in value.split(',') if t.strip()]

    return schema if schema else None


def build_generation_params(**kwargs) -> Dict:
    """Build generation parameters - simplified"""
    # Only keep non-None values
    return {k: v for k, v in kwargs.items() if v is not None}


def aggregate_chunk_results(all_results: List[Dict], task: str, filter_duplicates_flag: bool) -> Dict[str, List]:
    """✅ FIXED: Aggregate results from chunks with proper event flattening"""
    result_key = {'NER': 'entities', 'RE': 'relations', 'EE': 'events'}.get(task.upper(), 'entities')
    aggregated = {'entities': [], 'relations': [], 'events': []}

    # Collect all results
    for chunk_result in all_results:
        if result_key in chunk_result:
            items = chunk_result[result_key]

            if result_key == 'events':
                # ✅ Fix: Flatten nested events structure
                for item in items:
                    if isinstance(item, dict) and 'events' in item:
                        # Nested format: {"events": [event1, event2]}
                        aggregated[result_key].extend(item['events'])
                    else:
                        # Flat format: direct event
                        aggregated[result_key].append(item)
            else:
                # For entities and relations - direct extend
                aggregated[result_key].extend(items)

    # Filter duplicates if requested
    if filter_duplicates_flag:
        aggregated[result_key] = _remove_duplicates(aggregated[result_key], task)

    return aggregated


def format_extraction_result(result, task: str, text: str, gen_params: Dict) -> Dict:
    """Format extraction result - simplified"""
    # Convert to dict
    if hasattr(result, 'to_dict'):
        formatted = result.to_dict()
    elif isinstance(result, list):
        task_key = {'NER': 'entities', 'RE': 'relations', 'EE': 'events'}.get(task.upper(), 'entities')
        formatted = {task_key: [item.to_dict() if hasattr(item, 'to_dict') else item for item in result]}
    elif isinstance(result, dict):
        formatted = result
    else:
        formatted = {'results': str(result)}

    # Add minimal metadata
    formatted['generation_info'] = {
        'parameters_used': gen_params,
        'text_length': len(text),
        'task': task
    }

    return formatted


def format_chunk_extraction_result(result, task: str, chunk_id: int) -> Dict:
    """✅ FIXED: Format chunk extraction result with event flattening"""
    formatted = {'chunk_id': chunk_id, 'entities': [], 'relations': [], 'events': []}
    task_key = {'NER': 'entities', 'RE': 'relations', 'EE': 'events'}.get(task.upper(), 'entities')

    try:
        if isinstance(result, list):
            items = [item.to_dict() if hasattr(item, 'to_dict') else item for item in result]

            if task_key == 'events':
                # ✅ Flatten events at chunk level
                for item in items:
                    if isinstance(item, dict) and 'events' in item:
                        # Nested EE format
                        formatted[task_key].extend(item['events'])
                    else:
                        # Single event
                        formatted[task_key].append(item)
            else:
                formatted[task_key] = items

    except Exception as e:
        logger.warning(f"Failed to format chunk result: {e}")

    return formatted


def _remove_duplicates(items: List[Dict], task: str) -> List[Dict]:
    """Remove duplicates based on task type"""
    seen = set()
    filtered = []

    for item in items:
        # Create unique key
        if task.upper() == 'NER':
            key = (item.get('entity_type', ''), item.get('entity_mention', ''))
        elif task.upper() == 'RE':
            key = (item.get('relation_type', ''), item.get('head_entity', ''), item.get('tail_entity', ''))
        elif task.upper() == 'EE':
            key = (item.get('trigger_type', ''), item.get('trigger', ''))
        else:
            key = str(item)

        if key not in seen:
            seen.add(key)
            filtered.append(item)

    return filtered


def validate_chunk_extraction_task(task: str) -> bool:
    """Validate if task is supported for chunk extraction"""
    return task.upper() in ['NER', 'RE', 'EE']