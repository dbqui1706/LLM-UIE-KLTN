import logging
from typing import Dict, List, Any, Optional
logger = logging.getLogger(__name__)

def extract_uie_data(data: Any) -> Dict[str, List[Dict]]:
    result = {'entities': [], 'relations': [], 'events': []}

    if not data:
        return result

    if hasattr(data, 'to_dict'):
        data = data.to_dict()

    if isinstance(data, dict):
        result.update({k: v for k, v in data.items() if k in result})
        if 'aggregated_results' in data:
            return extract_uie_data(data['aggregated_results'])
        return result

    if isinstance(data, list):
        for item in data:
            item_data = item.to_dict() if hasattr(item, 'to_dict') else item
            type_name = type(item).__name__

            if type_name == 'NER':
                result['entities'].append(item_data)
            elif type_name == 'RE':
                result['relations'].append(item_data)
            elif type_name in ['EE', 'EET']:
                if 'events' in item_data:
                    for event in item_data['events']:
                        result['events'].append(event)
                else:
                    result['events'].append(item_data)

    return result


