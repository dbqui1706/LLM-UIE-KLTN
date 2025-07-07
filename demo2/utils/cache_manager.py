import hashlib
from typing import Dict, Any, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class CachingManager:
    """Simple in-memory cache for document processing"""
    
    def __init__(self):
        self._document_cache = {}
        self._model_cache = {}
        self._last_file_hash = None
        
    def get_file_hash(self, file_path: str) -> str:
        """Get file hash for cache key"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return str(hash(file_path))
    
    def cache_document(self, file_path: str, content: str, metadata: Dict[str, Any]):
        """Cache processed document"""
        file_hash = self.get_file_hash(file_path)
        self._document_cache[file_hash] = {
            'content': content,
            'metadata': metadata,
            'file_path': file_path
        }
        self._last_file_hash = file_hash
        logger.info(f"📦 Cached document: {file_path}")
    
    def get_cached_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached document if available"""
        file_hash = self.get_file_hash(file_path)
        cached = self._document_cache.get(file_hash)
        if cached:
            logger.info(f"✅ Using cached document: {file_path}")
        return cached
    
    def cache_model(self, model_key: str, model):
        """Cache model (like semantic model)"""
        self._model_cache[model_key] = model
        logger.info(f"📦 Cached model: {model_key}")
    
    def get_cached_model(self, model_key: str):
        """Get cached model"""
        model = self._model_cache.get(model_key)
        if model:
            logger.info(f"✅ Using cached model: {model_key}")
        return model
    
    def clear_document_cache(self):
        """Clear document cache"""
        self._document_cache.clear()
        self._last_file_hash = None
        logger.info("🧹 Document cache cleared")
    
    def clear_all_cache(self):
        """Clear all caches"""
        self._document_cache.clear()
        self._model_cache.clear()
        self._last_file_hash = None
        logger.info("🧹 All caches cleared")

# Global cache instance
_cache_instance = None

def get_cache() -> CachingManager:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CachingManager()
    return _cache_instance