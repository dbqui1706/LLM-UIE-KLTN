import logging
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from ..utils.cache_manager import get_cache

# LangChain imports with error handling
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        SentenceTransformersTokenTextSplitter,
        SpacyTextSplitter,
        NLTKTextSplitter,
        MarkdownHeaderTextSplitter
    )
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain text splitters not available: {e}")
    LANGCHAIN_AVAILABLE = False

# Sentence Transformers import with error handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# NLTK import with error handling
try:
    import nltk
    # Download required data if not present
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
    TOKEN = "token"
    SEMANTIC = "semantic"
    SPACY = "spacy"
    HYBRID = "hybrid"
    MARKDOWN = "markdown"


@dataclass
class ChunkingConfig:
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE  # ✅ Changed default
    chunk_size: int = 300
    chunk_overlap: int = 50
    length_function: str = "len"
    separators: Optional[List[str]] = None
    keep_separator: bool = True

    # Semantic chunking
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    similarity_threshold: float = 0.7

    preserve_sentences: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 500

    def __post_init__(self):
        """Validate config after initialization"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")


class DocumentChunker:
    def __init__(self, config: ChunkingConfig = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain text splitters are required. Install with: pip install langchain-text-splitters")
        
        self.config = config or ChunkingConfig()
        self.splitter = None
        self.semantic_model = None
        self._strategy_cache = {}  # ✅ Cache cho splitters
        self.cache = get_cache()
        self._initialize_splitter()
        
        logger.info(f"✅ Initialized DocumentChunker with strategy: {self.config.strategy.value}")

    def _initialize_splitter(self):
        """Initialize splitter based on strategy với error handling"""
        
        strategy_map = {
            ChunkingStrategy.RECURSIVE: self._init_recursive_splitter,
            ChunkingStrategy.SENTENCE: self._init_sentence_splitter,
            ChunkingStrategy.TOKEN: self._init_token_splitter,
            ChunkingStrategy.SEMANTIC: self._init_semantic_splitter,
            ChunkingStrategy.SPACY: self._init_spacy_splitter,
            ChunkingStrategy.HYBRID: self._init_hybrid_splitter,
            ChunkingStrategy.MARKDOWN: self._init_markdown_splitter
        }

        # ✅ Check if strategy is cached
        cache_key = self._get_cache_key()
        if cache_key in self._strategy_cache:
            self.splitter = self._strategy_cache[cache_key]
            logger.info(f"🔄 Using cached splitter for {self.config.strategy.value}")
            return

        try:
            init_func = strategy_map.get(self.config.strategy)
            if init_func:
                init_func()
                # ✅ Cache the splitter
                self._strategy_cache[cache_key] = self.splitter
                logger.info(f"✅ Initialized {self.config.strategy.value} splitter")
            else:
                raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.config.strategy.value} splitter: {e}")
            # ✅ Fallback to recursive
            logger.info("🔄 Falling back to recursive splitter")
            self._init_recursive_splitter()

    def _get_cache_key(self) -> str:
        """Generate cache key for current config"""
        return (f"{self.config.strategy.value}_{self.config.chunk_size}_"
                f"{self.config.chunk_overlap}_{hash(str(self.config.separators))}")

    def _init_recursive_splitter(self):
        """✅ Fixed RecursiveCharacterTextSplitter"""
        separators = self.config.separators or [
            "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=separators,
            keep_separator=self.config.keep_separator,
            is_separator_regex=False,
            length_function=len
        )
        logger.info(f"📏 Recursive splitter: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

    def _init_sentence_splitter(self):
        """✅ Fixed sentence splitter với NLTK check"""
        if not NLTK_AVAILABLE:
            logger.warning("❌ NLTK not available, falling back to recursive splitter")
            self._init_recursive_splitter()
            return

        try:
            self.splitter = NLTKTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                language="english",
            )
            logger.info(f"📏 NLTK sentence splitter: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        except Exception as e:
            logger.warning(f"❌ NLTK splitter failed: {e}, falling back to recursive")
            self._init_recursive_splitter()

    def _init_token_splitter(self):
        """✅ Fixed token-based splitter"""
        try:
            self.splitter = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                encoding_name="cl100k_base"
            )
            logger.info(f"📏 Token splitter: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        except Exception as e:
            logger.warning(f"❌ Token splitter failed: {e}, falling back to recursive")
            self._init_recursive_splitter()

    def _init_semantic_splitter(self):
        """✅ Fixed semantic chunker"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("❌ SentenceTransformers not available, falling back to recursive")
            self._init_recursive_splitter()
            return

        try:
            # Load model if not cached
            model_key = f"semantic_model_{self.config.model_name}"
            cached_model = self.cache.get_cached_model(model_key)
            
            if cached_model:
                self.semantic_model = cached_model
            else:
                # Load và cache model
                logger.info(f"🔄 Loading semantic model: {self.config.model_name}")
                self.semantic_model = SentenceTransformer(self.config.model_name)
                self.cache.cache_model(model_key, self.semantic_model)

            self.splitter = SentenceTransformersTokenTextSplitter(
                model_name=self.config.model_name,
                chunk_overlap=self.config.chunk_overlap,
                tokens_per_chunk=self.config.chunk_size,
            )
            logger.info(f"📏 Semantic splitter ready: tokens_per_chunk={self.config.chunk_size}")
        except Exception as e:
            logger.warning(f"❌ Semantic splitter failed: {e}, falling back to recursive")
            self._init_recursive_splitter()

    def _init_spacy_splitter(self):
        """✅ Fixed spaCy splitter"""
        try:
            self.splitter = SpacyTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                pipeline="en_core_web_sm"
            )
            logger.info(f"📏 spaCy splitter: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
        except Exception as e:
            logger.warning(f"❌ spaCy splitter failed: {e}, falling back to recursive")
            self._init_recursive_splitter()

    def _init_hybrid_splitter(self):
        """✅ Fixed hybrid approach - không override splitter"""
        self._init_recursive_splitter()
        logger.info("📏 Hybrid splitter initialized (uses multiple strategies)")

    def _init_markdown_splitter(self):
        """✅ Fixed Markdown splitter"""
        try:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            
            # Combine với RecursiveCharacterTextSplitter
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            self.markdown_splitter = markdown_splitter
            logger.info(f"📏 Markdown splitter: chunk_size={self.config.chunk_size}")
        except Exception as e:
            logger.warning(f"❌ Markdown splitter failed: {e}, falling back to recursive")
            self._init_recursive_splitter()

    def chunk_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = "balanced"
    ) -> List[Dict[str, Any]]:
        """Fixed - respect user settings"""
        
        if not content or not content.strip():
            return []

        logger.info(f"🔄 Chunking with user settings: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

        try:
            # content = self._prepare_content(content)
            chunks = self._perform_chunking(content)
            processed_chunks = self._post_process_chunks(chunks, metadata or {})

            logger.info(f"✅ Created {len(processed_chunks)} chunks using {self.config.strategy.value}")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Chunking failed: {e}")
            return []


    def _perform_chunking(self, content: str) -> List[str]:
        """✅ Perform chunking với strategy-specific logic"""
        
        if self.config.strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunking(content)
        elif self.config.strategy == ChunkingStrategy.MARKDOWN and hasattr(self, 'markdown_splitter'):
            return self._markdown_chunking(content)
        else:
            return self._standard_chunking(content)

    # def _prepare_content(self, content: str) -> str:
    #     """✅ Prepare content với better cleaning"""
    #     # Normalize whitespace
    #     content = re.sub(r'\s+', ' ', content)
    #     content = re.sub(r'\n\s*\n', '\n\n', content)

    #     # Fix sentence boundaries
    #     content = re.sub(r'\.([A-Z])', r'. \1', content)
    #     content = re.sub(r'\!([A-Z])', r'! \1', content)
    #     content = re.sub(r'\?([A-Z])', r'? \1', content)

    #     return content.strip()

    def _standard_chunking(self, content: str) -> List[str]:
        """✅ Standard chunking với error handling"""
        try:
            if not self.splitter:
                raise ValueError("Splitter not initialized")
                
            chunks = self.splitter.split_text(content)
            result = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            logger.info(f"📊 Standard chunking: {len(result)} chunks, avg size: {sum(len(c) for c in result) / len(result) if result else 0:.0f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Standard chunking failed: {e}")
            return self._fallback_chunking(content)

    def _hybrid_chunking(self, content: str) -> List[str]:
        """✅ Fixed hybrid chunking - không override splitter"""
        logger.info("🔄 Starting hybrid chunking")
        
        try:
            # Step 1: Create sentence splitter for initial split
            if NLTK_AVAILABLE:
                sentence_splitter = NLTKTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
            else:
                # Fallback to recursive với sentence-friendly separators
                sentence_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    separators=[". ", "! ", "? ", "\n\n", "\n", " ", ""]
                )

            initial_chunks = sentence_splitter.split_text(content)
            logger.info(f"📊 Initial sentence split: {len(initial_chunks)} chunks")

            # Step 2: Apply recursive splitting cho chunks quá lớn
            final_chunks = []
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

            for chunk in initial_chunks:
                if len(chunk) > self.config.max_chunk_size:
                    # Split further
                    sub_chunks = recursive_splitter.split_text(chunk)
                    final_chunks.extend(sub_chunks)
                    logger.debug(f"📏 Split large chunk ({len(chunk)} chars) into {len(sub_chunks)} sub-chunks")
                else:
                    final_chunks.append(chunk)

            result = [chunk.strip() for chunk in final_chunks if chunk.strip()]
            logger.info(f"📊 Hybrid chunking completed: {len(result)} chunks")
            return result

        except Exception as e:
            logger.error(f"❌ Hybrid chunking failed: {e}")
            return self._fallback_chunking(content)

    def _markdown_chunking(self, content: str) -> List[str]:
        """✅ Markdown chunking implementation"""
        try:
            # First split by headers
            md_header_splits = self.markdown_splitter.split_text(content)
            
            # Then apply size-based splitting
            final_chunks = []
            for doc in md_header_splits:
                if len(doc.page_content) > self.config.chunk_size:
                    chunks = self.splitter.split_text(doc.page_content)
                    final_chunks.extend(chunks)
                else:
                    final_chunks.append(doc.page_content)
            
            return [chunk.strip() for chunk in final_chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"❌ Markdown chunking failed: {e}")
            return self._fallback_chunking(content)

    def _fallback_chunking(self, content: str) -> List[str]:
        """✅ Improved fallback chunking"""
        logger.warning("🔄 Using fallback chunking method")
        
        # Simple sentence-based splitting với better logic
        sentences = re.split(r'[.!?]+\s+', content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding sentence would exceed chunk size
            potential_chunk = current_chunk + sentence + ". "
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                if len(sentence) > self.config.chunk_size:
                    # Split long sentence
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + word + " ") <= self.config.chunk_size:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    current_chunk = temp_chunk
                else:
                    current_chunk = sentence + ". "

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.info(f"📊 Fallback chunking: {len(chunks)} chunks")
        return chunks

    def _post_process_chunks(self, chunks: List[str], base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """✅ Enhanced post-processing với detailed metadata"""
        
        processed_chunks = []
        total_chars = sum(len(chunk) for chunk in chunks)

        for i, chunk in enumerate(chunks):
            # Skip chunks quá ngắn
            if len(chunk.strip()) < self.config.min_chunk_size:
                logger.debug(f"⚠️ Skipping chunk {i}: too short ({len(chunk)} chars)")
                continue

            # Create detailed metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_tokens": self._count_tokens(chunk),
                "chunk_words": len(chunk.split()),
                "chunking_strategy": self.config.strategy.value,
                "chunk_overlap": self.config.chunk_overlap,
                "chunk_percentage": (len(chunk) / total_chars) * 100 if total_chars > 0 else 0,
                "config_used": {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "strategy": self.config.strategy.value
                }
            })

            processed_chunks.append({
                "content": chunk.strip(),
                "metadata": chunk_metadata
            })

        logger.info(f"📊 Post-processing: {len(processed_chunks)} valid chunks from {len(chunks)} original chunks")
        return processed_chunks

    def _count_tokens(self, text: str) -> int:
        """✅ Improved token counting"""
        # Simple approximation: ~4 chars per token for English
        return max(1, len(text) // 4)

    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """✅ Enhanced chunking statistics"""
        if not chunks:
            return {"error": "No chunks provided"}

        chunk_sizes = [len(chunk["content"]) for chunk in chunks]
        chunk_tokens = [chunk["metadata"]["chunk_tokens"] for chunk in chunks]
        chunk_words = [chunk["metadata"]["chunk_words"] for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "size_stats": {
                "avg_chars": sum(chunk_sizes) / len(chunk_sizes),
                "min_chars": min(chunk_sizes),
                "max_chars": max(chunk_sizes),
                "total_chars": sum(chunk_sizes)
            },
            "token_stats": {
                "avg_tokens": sum(chunk_tokens) / len(chunk_tokens),
                "total_tokens": sum(chunk_tokens)
            },
            "word_stats": {
                "avg_words": sum(chunk_words) / len(chunk_words),
                "total_words": sum(chunk_words)
            },
            "strategy_used": self.config.strategy.value,
            "config_used": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        }

        return stats


# ✅ Enhanced preset configurations
class ChunkingPresets:
    """Pre-configured chunking setups với realistic values"""

    @staticmethod
    def uie_optimized() -> ChunkingConfig:
        """Optimized cho UIE tasks"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,  # Most reliable
            chunk_size=300,
            chunk_overlap=50,
            preserve_sentences=True,
            min_chunk_size=50,
            max_chunk_size=400
        )

    @staticmethod
    def fast_processing() -> ChunkingConfig:
        """Fast processing với larger chunks"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=500,
            chunk_overlap=50,
            preserve_sentences=False
        )

    @staticmethod
    def high_quality() -> ChunkingConfig:
        """High quality với sentence preservation"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE if NLTK_AVAILABLE else ChunkingStrategy.RECURSIVE,
            chunk_size=250,
            chunk_overlap=30,
            preserve_sentences=True
        )

    @staticmethod
    def academic_papers() -> ChunkingConfig:
        """Optimized cho academic papers"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            chunk_size=400,
            chunk_overlap=80,  # Higher overlap for academic content
            preserve_sentences=True,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )

    @staticmethod
    def debug_mode() -> ChunkingConfig:
        """Tiny chunks for debugging"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=20,
            max_chunk_size=150
        )