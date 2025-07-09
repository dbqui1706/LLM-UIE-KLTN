import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def prepare_schema(entity_types: str, relation_types: str, event_types: str, argument_types: str) -> Dict[
    str, List[str]]:
    """Prepare user schema from input strings"""
    schema = {}

    schema_mapping = {
        'entity_types': entity_types,
        'relation_types': relation_types,
        'event_types': event_types,
        'argument_types': argument_types
    }

    for key, value in schema_mapping.items():
        if value.strip():
            schema[key] = [t.strip() for t in value.split(',') if t.strip()]

    return schema if schema else None


def get_result_key(task: str) -> str:
    """Get result key for task"""
    mapping = {'NER': 'entities', 'RE': 'relations', 'EE': 'events'}
    return mapping.get(task.upper(), 'entities')


def filter_duplicates(items: List[Dict], task: str) -> List[Dict]:
    """Filter duplicate items based on task type"""

    def get_key_for_task(item: Dict, task_type: str) -> tuple:
        key_generators = {
            'NER': lambda x: (x.get('entity_type', ''), x.get('entity_mention', '')),
            'RE': lambda x: (x.get('relation_type', ''), x.get('head_entity', ''), x.get('tail_entity', '')),
            'EE': lambda x: (x.get('trigger_type', ''), x.get('trigger', ''))
        }
        generator = key_generators.get(task_type.upper(), str)
        return generator(item)

    seen = set()
    filtered = []

    for item in items:
        key = get_key_for_task(item, task)
        if key not in seen:
            seen.add(key)
            filtered.append(item)

    return filtered


def aggregate_chunk_results(all_results: List[Dict], task: str, filter_duplicates_flag: bool) -> Dict[str, List]:
    """Aggregate results from all chunks"""
    aggregated = {'entities': [], 'relations': [], 'events': []}
    result_key = get_result_key(task)

    # Collect all results
    for chunk_result in all_results:
        if result_key in chunk_result:
            aggregated[result_key].extend(chunk_result[result_key])

    # Filter duplicates if requested
    if filter_duplicates_flag:
        aggregated[result_key] = filter_duplicates(aggregated[result_key], task)

    return aggregated


def format_extraction_result(result, task: str, text: str, gen_params: Dict) -> Dict:
    """Format extraction result to dict"""
    try:
        # Try different formatting strategies
        formatters = [
            lambda r: r.to_dict() if hasattr(r, 'to_dict') else None,
            lambda r: r if isinstance(r, dict) else None,
            lambda r: format_list_result(r, task) if isinstance(r, list) else None,
            lambda r: {'results': str(r)}
        ]

        formatted = None
        for formatter in formatters:
            formatted = formatter(result)
            if formatted is not None:
                break

        # Add generation info
        formatted['generation_info'] = {
            'parameters_used': gen_params,
            'text_length': len(text),
            'task': task
        }

        return formatted

    except Exception as e:
        logger.error(f"Failed to format result: {e}")
        return {'error': f'Failed to format result: {e}'}


def format_list_result(result: list, task: str) -> Dict:
    """Format list results based on task"""
    task_keys = {
        'NER': 'entities',
        'RE': 'relations',
        'EE': 'events'
    }

    key = task_keys.get(task.upper(), 'results')
    formatted_items = [item.to_dict() if hasattr(item, 'to_dict') else item for item in result]

    return {key: formatted_items}


def format_chunk_extraction_result(result, task: str, chunk_id: int) -> Dict:
    """Format chunk extraction result"""
    formatted = {'chunk_id': chunk_id, 'entities': [], 'relations': [], 'events': []}

    try:
        if isinstance(result, list):
            task_key = get_result_key(task)
            formatted_items = [item.to_dict() if hasattr(item, 'to_dict') else item for item in result]
            formatted[task_key] = formatted_items
    except Exception as e:
        logger.warning(f"Failed to format chunk result: {e}")

    return formatted

def create_mock_extraction_result(text: str, task: str, gen_params: Dict) -> Dict:
    """Create mock extraction result for demo"""
    result = {
        'text': text[:100] + "..." if len(text) > 100 else text,
        'task': task,
        'entities': [],
        'relations': [],
        'events': [],
        'generation_info': {
            'parameters_used': gen_params,
            'text_length': len(text),
            'task': task,
            'note': 'Mock result - model not loaded'
        }
    }

    # Mock data generators
    mock_generators = {
        'NER': lambda: [
            {'entity_type': 'PERSON', 'entity_mention': 'Sample Person', 'confidence': 0.95},
            {'entity_type': 'ORG', 'entity_mention': 'Sample Organization', 'confidence': 0.90}
        ],
        'RE': lambda: [
            {'relation_type': 'WORKS_AT', 'head_entity': 'Sample Person',
             'tail_entity': 'Sample Organization', 'confidence': 0.88}
        ],
        'EE': lambda: [
            {'trigger': 'meeting', 'trigger_type': 'MEETING', 'confidence': 0.85}
        ]
    }

    # Add mock data based on task
    task_upper = task.upper()
    if task_upper in mock_generators:
        key = get_result_key(task)
        result[key] = mock_generators[task_upper]()
    elif task_upper == 'ALL':
        for task_type, generator in mock_generators.items():
            key = get_result_key(task_type)
            result[key] = generator()

    return result


def create_mock_chunk_result(chunk_id: int, task: str) -> Dict:
    """Create mock chunk result"""
    result = {'chunk_id': chunk_id, 'entities': [], 'relations': [], 'events': []}

    mock_generators = {
        'NER': lambda: [{'entity_type': 'PERSON', 'entity_mention': f'Person_{chunk_id}', 'confidence': 0.95}],
        'RE': lambda: [{'relation_type': 'WORKS_AT', 'head_entity': f'Person_{chunk_id}',
                        'tail_entity': f'Org_{chunk_id}', 'confidence': 0.88}],
        'EE': lambda: [{'trigger': f'meeting_{chunk_id}', 'trigger_type': 'MEETING', 'confidence': 0.85}]
    }

    generator = mock_generators.get(task.upper())
    if generator:
        key = get_result_key(task)
        result[key] = generator()

    return result


def validate_chunk_extraction_task(task: str) -> bool:
    """Validate if task is supported for chunk extraction"""
    return task.upper() in ['NER', 'RE', 'EE']


def build_generation_params(max_new_tokens=512, temperature=0.1, top_p=0.9, top_k=50,
                            do_sample=True, repetition_penalty=1.0, no_repeat_ngram_size=0,
                            num_beams=1, early_stopping=False) -> Dict:
    """Build generation parameters dict"""
    return {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'do_sample': do_sample,
        'repetition_penalty': repetition_penalty,
        'no_repeat_ngram_size': no_repeat_ngram_size,
        'num_beams': num_beams,
        'early_stopping': early_stopping
    }