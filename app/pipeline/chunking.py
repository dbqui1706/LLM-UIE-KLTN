import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter,
    MarkdownTextSplitter
)

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


class SentenceTransformersTokenTextSplitterCustom(SentenceTransformersTokenTextSplitter):
    def __init__(
            self,
            chunk_overlap: int = 50,
            model_name: str = "sentence-transformers/all-mpnet-base-v2",
            model: Optional[Any] = None,
            tokens_per_chunk: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        if model is not None:
            from langchain_text_splitters.base import TextSplitter
            TextSplitter.__init__(self, **kwargs, chunk_overlap=chunk_overlap)

            logger.info(f"🎯 Using provided cached model")
            self.model_name = model_name
            self._model = model
            self.tokenizer = self._model.tokenizer
            self._initialize_chunk_configuration(tokens_per_chunk=tokens_per_chunk)

        else:
            logger.info(f"🔄 Loading model via parent class: {model_name}")
            super().__init__(
                chunk_overlap=chunk_overlap,
                model_name=model_name,
                tokens_per_chunk=tokens_per_chunk,
                **kwargs
            )

class DocumentChunker:
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.splitter = None

        # caching
        from ..utils.cache_manager import get_cache
        self.cache = get_cache()

        cached_chunker = self.cache.get_cached_chunker(self.config)
        if cached_chunker:
            self.splitter = cached_chunker.splitter
            logger.info(f"♻️ Reused cached splitter: {self.config.strategy.value}")
        else:
            self._initialize_splitter()
            # Caching the splitter instance
            self.cache.cache_chunker(self.config, self)
            logger.info(f"✅ DocumentChunker ready: {self.config.strategy.value}")
    def _initialize_splitter(self):
        strategy_initializers = {
            ChunkingStrategy.SEMANTIC: self._init_semantic_splitter,
            ChunkingStrategy.SENTENCE: lambda: self._init_sentence_splitter() if NLTK_AVAILABLE else self._init_recursive_splitter(),
            ChunkingStrategy.MARKDOWN: self._init_markdown_splitter,
            ChunkingStrategy.RECURSIVE: self._init_recursive_splitter
        }

        try:
            initializer = strategy_initializers.get(self.config.strategy, self._init_recursive_splitter)
            initializer()
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
        model_name = self.config.model_name
        cached_model = self.cache.get_cached_model(model_name)

        if cached_model:
            model = cached_model
            logger.info(f"♻️ Using cached model: {model_name}")
        else:
            logger.info(f"🔄 Loading semantic model: {model_name}")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)

            # Cache the model
            self.cache.cache_model(model_name, model)
            logger.info(f"✅ Model cached: {model_name}")

        self.splitter = SentenceTransformersTokenTextSplitterCustom(
            model=model,
            model_name=self.config.model_name,
            chunk_overlap=self.config.chunk_overlap,
            tokens_per_chunk=self.config.chunk_size
        )
        logger.info(f"📏 Semantic splitter ready: tokens={self.config.chunk_size}")

    def _init_markdown_splitter(self):
        self.splitter = MarkdownTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
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
