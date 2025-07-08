import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter,
    MarkdownTextSplitter
)


# Sentence Transformers import with error handling
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# NLTK import with error handling
try:
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"


@dataclass
class ChunkingConfig:
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 300
    chunk_overlap: int = 50
    min_chunk_size: int = 10
    max_chunk_size: int = chunk_size * 2
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


class DocumentChunker:
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.splitter = None
        self.markdown_splitter = None

        # Import cache manager
        from ..utils.cache_manager import get_cache
        self.cache = get_cache()

        self._initialize_splitter()
        logger.info(f"✅ DocumentChunker ready: {self.config.strategy.value}")

    def _initialize_splitter(self):
        try:
            if self.config.strategy == ChunkingStrategy.SEMANTIC:
                self._init_semantic_splitter()
            elif self.config.strategy == ChunkingStrategy.SENTENCE and NLTK_AVAILABLE:
                self._init_sentence_splitter()
            elif self.config.strategy == ChunkingStrategy.MARKDOWN:  # ✅ Add markdown
                self._init_markdown_splitter()
            else:
                self._init_recursive_splitter()
        except Exception as e:
            logger.warning(f"❌ Failed to init {self.config.strategy.value}, using recursive: {e}")
            self._init_recursive_splitter()

    def _init_recursive_splitter(self):
        """Initialize recursive character text splitter"""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
        logger.info(f"📏 Recursive splitter: size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

    def _init_sentence_splitter(self):
        """Initialize sentence-based splitter"""
        self.splitter = NLTKTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        logger.info(f"📏 Sentence splitter: size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

    def _init_semantic_splitter(self):
        """Initialize semantic splitter with model caching"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available")

        # # Check cache for model
        # model_key = f"semantic_model_{self.config.model_name}"
        # cached_model = self.cache.get_cached_model(model_key)
        #
        # if not cached_model:
        #     logger.info(f"🔄 Loading semantic model: {self.config.model_name}")
        #     model = SentenceTransformer(self.config.model_name)
        #     self.cache.cache_model(model_key, model)

        self.splitter = SentenceTransformersTokenTextSplitter(
            model_name=self.config.model_name,
            chunk_overlap=self.config.chunk_overlap,
            tokens_per_chunk=self.config.chunk_size
        )
        logger.info(f"📏 Semantic splitter ready: tokens={self.config.chunk_size}")

    def _init_markdown_splitter(self):
        """✅ Initialize markdown splitter - simple version"""
        self.splitter = MarkdownTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "# ", "## ", "### ", "#### ", "##### ", "###### ", "* ", "- ", "> ", "  "]
        )

        logger.info(f"📏 Markdown splitter ready: size={self.config.chunk_size} overlap={self.config.chunk_overlap}")

    def chunk_document(self, content: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not content or not content.strip():
            return []

        logger.info(f"🔄 Chunking: {len(content)} chars with {self.config.strategy.value}")

        try:
            # Split text into chunks
            chunks = self.splitter.split_text(content)

            # Process chunks
            result = []
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.strip()

                # Skip too short chunks
                if len(chunk_text) < self.config.min_chunk_size:
                    continue

                # Create chunk with minimal metadata
                chunk_data = {
                    "content": chunk_text,
                    "metadata": {
                        "chunk_id": i,
                        "chunk_size": len(chunk_text),
                        "chunking_strategy": self.config.strategy.value,
                        **(metadata or {})
                    }
                }
                result.append(chunk_data)

            logger.info(f"✅ Created {len(result)} chunks")
            return result

        except Exception as e:
            logger.error(f"❌ Chunking failed: {e}")
            return []
