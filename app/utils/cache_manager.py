import hashlib
from typing import Dict, Any, Optional, OrderedDict
from collections import OrderedDict
import time
import logging
import torch

logger = logging.getLogger(__name__)


class CachingManager:
    """Simple in-memory cache for document processing"""
    MAX_DOCUMENT_CACHE = 5
    MAX_CHUNKER_CACHE_SIZE = 5
    MAX_MODEL_CACHE_SIZE = 2

    def __init__(self):
        # ✅ Use OrderedDict để track insertion order (LRU)
        self._document_cache: OrderedDict = OrderedDict()
        self._chunker_cache: OrderedDict = OrderedDict()
        self._model_cache: OrderedDict = OrderedDict()
        self._last_file_hash = None

        # ✅ Track access time for LRU
        self._document_access_time: Dict[str, float] = {}
        self._chunker_access_time: Dict[str, float] = {}
        self._model_access_time: Dict[str, float] = {}

    def get_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            return str(hash(file_path))

    def cache_document(self, file_path: str, content: str, metadata: Dict[str, Any]):
        file_hash = self.get_file_hash(file_path)

        # ✅ Remove existing entry để move to end (LRU)
        if file_hash in self._document_cache:
            self._document_cache.pop(file_hash)

        self._document_cache[file_hash] = {
            'content': content,
            'metadata': metadata,
            'file_path': file_path
        }
        self._document_access_time[file_hash] = time.time()
        self._last_file_hash = file_hash

        logger.info(f"📦 Cached document: {file_path}")
        self.clear_cache_oversize()

    def get_cached_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached document with LRU update"""
        file_hash = self.get_file_hash(file_path)
        cached = self._document_cache.get(file_hash)

        if cached:
            self._document_cache.move_to_end(file_hash)
            self._document_access_time[file_hash] = time.time()
            logger.info(f"✅ Using cached document: {file_path}")

        return cached

    def get_chunker_key(self, config) -> str:
        return f"{config.strategy.value}_{config.chunk_size}_{config.chunk_overlap}"

    def cache_chunker(self, config, chunker_instance):
        key = self.get_chunker_key(config)
        if key in self._chunker_cache:
            self._chunker_cache.pop(key)

        self._chunker_cache[key] = chunker_instance
        self._chunker_access_time[key] = time.time()

        logger.info(f"📦 Cached chunker: {key}")
        self.clear_cache_oversize()

    def get_cached_chunker(self, config):
        key = self.get_chunker_key(config)
        chunker = self._chunker_cache.get(key)

        if chunker:
            self._chunker_cache.move_to_end(key)
            self._chunker_access_time[key] = time.time()
            logger.info(f"✅ Using cached chunker: {key}")

        return chunker

    # ========== Model Cache ==========
    def cache_model(self, model_key: str, model_instance):
        """Cache model with auto-cleanup"""
        if model_key in self._model_cache:
            self._model_cache.pop(model_key)

        self._model_cache[model_key] = model_instance
        self._model_access_time[model_key] = time.time()

        logger.info(f"📦 Cached model: {model_key}")
        self.clear_cache_oversize()

    def get_cached_model(self, model_key: str):
        """Get cached model with LRU update"""
        model = self._model_cache.get(model_key)

        if model:
            self._model_cache.move_to_end(model_key)
            self._model_access_time[model_key] = time.time()

        return model

    def has_cached_model(self, model_key: str) -> bool:
        """Check if model is cached"""
        return model_key in self._model_cache

    def clear_document_cache(self):
        self._document_cache.clear()
        self._last_file_hash = None
        logger.info("🧹 Document cache cleared")

    def clear_chunker_cache(self):
        self._chunker_cache.clear()
        logger.info("🧹 Chunker cache cleared")

    def clear_model_cache(self):
        self._model_cache.clear()
        logger.info("🧹 Model cache cleared")

    def clear_all_cache(self):
        self._document_cache.clear()
        self._chunker_cache.clear()
        self._model_cache.clear()  # ✅ Clear model cache
        self._last_file_hash = None
        logger.info("🧹 All caches cleared (documents, chunkers, models)")

    def clear_cache_oversize(self):
        current_time = time.time()
        if len(self._document_cache) > self.MAX_DOCUMENT_CACHE:
            excess_count = len(self._document_cache) - self.MAX_DOCUMENT_CACHE
            logger.info(
                f"🧹 Document cache oversized ({len(self._document_cache)}>{self.MAX_DOCUMENT_CACHE}), removing {excess_count} oldest items")

            for _ in range(excess_count):
                oldest_key = next(iter(self._document_cache))
                removed_item = self._document_cache.pop(oldest_key)
                self._document_access_time.pop(oldest_key, None)
                logger.debug(f"🗑️ Removed cached document: {removed_item.get('file_path', oldest_key)}")

        if len(self._chunker_cache) > self.MAX_CHUNKER_CACHE_SIZE:
            excess_count = len(self._chunker_cache) - self.MAX_CHUNKER_CACHE_SIZE
            logger.info(
                f"🧹 Chunker cache oversized ({len(self._chunker_cache)}>{self.MAX_CHUNKER_CACHE_SIZE}), removing {excess_count} oldest items")

            for _ in range(excess_count):
                oldest_key = next(iter(self._chunker_cache))
                self._chunker_cache.pop(oldest_key)
                self._chunker_access_time.pop(oldest_key, None)
                logger.debug(f"🗑️ Removed cached chunker: {oldest_key}")

        if len(self._model_cache) > self.MAX_MODEL_CACHE_SIZE:
            excess_count = len(self._model_cache) - self.MAX_MODEL_CACHE_SIZE
            logger.warning(
                f"🧹 Model cache oversized ({len(self._model_cache)}>{self.MAX_MODEL_CACHE_SIZE}), removing {excess_count} oldest models")

            for _ in range(excess_count):
                oldest_key = next(iter(self._model_cache))
                removed_model = self._model_cache.pop(oldest_key)
                self._model_access_time.pop(oldest_key, None)
                logger.info(f"🗑️ Removed cached model: {oldest_key}")

                # ✅ Cleanup model memory
                try:
                    del removed_model
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass
# Global cache instance
_cache_instance = None


def get_cache() -> CachingManager:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CachingManager()
    return _cache_instance
