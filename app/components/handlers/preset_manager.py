from typing import Tuple, List, Dict, Any
from .base import BaseHandler
from ..presets import GenerationPresets

class PresetManager(BaseHandler):
    def apply_generation_preset(self, preset_name: str, *current_values) -> Tuple:
        """Apply generation parameter preset"""
        self._log_operation("Apply preset", preset=preset_name)
        
        try:
            return GenerationPresets.apply_preset(preset_name, current_values)
        except Exception as e:
            self.logger.error(f"Failed to apply preset {preset_name}: {e}")
            return current_values
    
    def get_available_presets(self) -> List[str]:
        """Get list of available presets"""
        return list(GenerationPresets.get_presets().keys())
    
    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get preset configuration info"""
        presets = GenerationPresets.get_presets()
        return presets.get(preset_name, {})