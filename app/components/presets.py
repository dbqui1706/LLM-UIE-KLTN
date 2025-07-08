from typing import Dict, Any

class GenerationPresets:
    
    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """Get all generation presets"""
        return {
            "conservative": {
                "temperature": 0.05,
                "top_p": 0.8,
                "top_k": 10,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2,
                "num_beams": 1,
                "early_stopping": False
            },
            "balanced": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "early_stopping": False
            },
            "creative": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 80,
                "do_sample": True,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "early_stopping": False
            },
            "precise": {
                "temperature": 0.01,
                "top_p": 0.7,
                "top_k": 5,
                "do_sample": False,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "num_beams": 3,
                "early_stopping": True
            }
        }
    
    @staticmethod
    def apply_preset(preset_name: str, current_values: tuple) -> tuple:
        """Apply generation parameter preset"""
        presets = GenerationPresets.get_presets()
        
        if preset_name not in presets:
            return current_values
        
        preset = presets[preset_name]
        
        # Return updated values keeping max_new_tokens unchanged
        return (
            current_values[0],  # max_new_tokens unchanged
            preset["temperature"],
            preset["top_p"], 
            preset["top_k"],
            preset["do_sample"],
            preset["repetition_penalty"],
            preset["no_repeat_ngram_size"],
            preset["num_beams"],
            preset["early_stopping"]
        )