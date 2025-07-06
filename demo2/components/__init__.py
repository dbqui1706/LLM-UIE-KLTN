from .generation_controls import create_generation_controls
from .text_extraction import text_extraction_tab
from .document_processing import document_processing_extraction_tab  # Updated import
from .handlers import setup_event_handlers, EventHandlers, TabEventHandlers
from .presets import GenerationPresets

__all__ = [
    'create_generation_controls',
    'text_extraction_tab',
    'document_processing_extraction_tab',  # Updated name
    'setup_event_handlers',
    'EventHandlers',
    'TabEventHandlers',
    'GenerationPresets'
]