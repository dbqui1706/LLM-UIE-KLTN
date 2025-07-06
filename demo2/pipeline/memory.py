import os
import gc
import psutil
import torch
import logging
from typing import Dict

class MemoryManager:
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }

    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}

        try:
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                memory_free = (torch.cuda.get_device_properties(i).total_memory
                               - torch.cuda.memory_reserved(i)) / 1024 / 1024

                gpu_memory[f'gpu_{i}'] = {
                    'allocated': memory_allocated,
                    'reserved': memory_reserved,
                    'free': memory_free
                }
            return gpu_memory
        except Exception as e:
            logger.warning(f"Error getting GPU memory info: {e}")
            return {}

    @staticmethod
    def cleanup_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()