# demo2/components/handlers/__init__.py
from .factory import EventHandlers, HandlerFactory
from .event_setup import setup_event_handlers, TabEventHandlers

__all__ = [
    'EventHandlers',
    'HandlerFactory', 
    'setup_event_handlers',
    'TabEventHandlers'
]