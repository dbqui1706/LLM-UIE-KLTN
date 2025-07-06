import logging
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# LangChain imports - thư viện chunking phổ biến nhất
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter
)

# For semantic chunking
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Các strategy chunking khác nhau"""
    RECURSIVE = "recursive"  # Recursive character splitting
    SENTENCE = "sentence"  # Sentence-based splitting
    TOKEN = "token"  # Token-based splitting
    SEMANTIC = "semantic"  # Semantic similarity splitting
    SPACY = "spacy"  # spaCy sentence splitting
    HYBRID = "hybrid"  # Combination approach
    MARKDOWN = "markdown"


@dataclass
class ChunkingConfig:
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    chunk_size: int = 300  # Optimal cho UIE models
    chunk_overlap: int = 50  # Overlap để preserve context
    length_function: str = "len"  # "len" hoặc "tiktoken"
    separators: Optional[List[str]] = None
    keep_separator: bool = True

    # semantic chunking
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    similarity_threshold: float = 0.7

    preserve_sentences: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 500


class DocumentChunker:
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.splitter = None
        self.semantic_model = None

        # Initialize splitter based on strategy
        self._initialize_splitter()

        logger.info(f"Initialized DocumentChunker with strategy: {self.config.strategy.value}")

    def _initialize_splitter(self):

        strategy_map = {
            ChunkingStrategy.RECURSIVE: self._init_recursive_splitter,
            ChunkingStrategy.SENTENCE: self._init_sentence_splitter,
            ChunkingStrategy.TOKEN: self._init_token_splitter,
            ChunkingStrategy.SEMANTIC: self._init_semantic_splitter,
            ChunkingStrategy.SPACY: self._init_spacy_splitter,
            ChunkingStrategy.HYBRID: self._init_hybrid_splitter,
            ChunkingStrategy.MARKDOWN: self._init_markdown_splitter
        }

        strategy_map[self.config.strategy]()

    def _init_recursive_splitter(self):
        """Initialize RecursiveCharacterTextSplitter"""
        separators = self.config.separators or [
            "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=separators,
            keep_separator=self.config.keep_separator,
            is_separator_regex=False
        )

    def _init_sentence_splitter(self):
        # Sử dụng NLTK sentence splitter với custom config
        self.splitter = NLTKTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len
        )

    def _init_token_splitter(self):
        """token-based splitter"""
        self.splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            encoding_name="cl100k_base"  # GPT-4 encoding
        )

    def _init_semantic_splitter(self):
        """semantic chunker với SentenceTransformers"""
        # sentence transformer model
        self.semantic_model = SentenceTransformer(self.config.model_name)

        self.splitter = SentenceTransformersTokenTextSplitter(
            model_name=self.config.model_name,
            chunk_overlap=self.config.chunk_overlap,
            tokens_per_chunk=self.config.chunk_size,
        )

    def _init_spacy_splitter(self):
        """spaCy splitter"""
        try:
            self.splitter = SpacyTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                pipeline="en_core_web_sm"
            )
        except OSError:
            logger.warning("spaCy model not found, falling back to sentence splitter")
            self._init_sentence_splitter()

    def _init_hybrid_splitter(self):
        """Initialize hybrid approach"""
        # Start with sentence-based, then apply recursive if needed
        self._init_sentence_splitter()

    def _init_markdown_splitter(self):
        """Initialize Markdown header splitter"""
        ...

    def chunk_document(
            self,
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            task_type: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Chunk document content với metadata preservation

        Args:
            content: Text content từ Docling
            metadata: Document metadata
            task_type: "ner", "re", "ee", hoặc "balanced"

        Returns:
            List of chunk dictionaries với content và metadata
        """
        if not content or not content.strip():
            return []

        logger.info(f"Chunking document for task: {task_type}")

        # Adjust config based on task type
        self._adjust_config_for_task(task_type)

        # Clean content trước khi chunk
        cleaned_content = self._prepare_content(content)

        # Perform chunking
        if self.config.strategy == ChunkingStrategy.HYBRID:
            chunks = self._hybrid_chunking(cleaned_content)
        else:
            chunks = self._standard_chunking(cleaned_content)

        # Post-process chunks
        processed_chunks = self._post_process_chunks(chunks, metadata or {})

        logger.info(f"Created {len(processed_chunks)} chunks")
        return processed_chunks

    def _adjust_config_for_task(self, task_type: str):
        """Adjust chunking config dựa trên UIE task"""
        task_configs = {
            "ner": {
                "chunk_size": 250,
                "chunk_overlap": 30,
                "preserve_sentences": True
            },
            "re": {
                "chunk_size": 350,
                "chunk_overlap": 50,
                "preserve_sentences": True
            },
            "ee": {
                "chunk_size": 400,
                "chunk_overlap": 70,
                "preserve_sentences": True
            },
            "balanced": {
                "chunk_size": 300,
                "chunk_overlap": 50,
                "preserve_sentences": True
            }
        }

        task_config = task_configs.get(task_type, task_configs["balanced"])

        # Update config
        for key, value in task_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Re-initialize splitter với config mới
        self._initialize_splitter()

    def _prepare_content(self, content: str) -> str:
        """Prepare content trước khi chunking"""
        # Basic cleaning để improve chunking quality

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)

        # Fix sentence boundaries
        content = re.sub(r'\.([A-Z])', r'. \1', content)
        content = re.sub(r'\!([A-Z])', r'! \1', content)
        content = re.sub(r'\?([A-Z])', r'? \1', content)

        return content.strip()

    def _standard_chunking(self, content: str) -> List[str]:
        """Standard chunking using configured splitter"""
        try:
            chunks = self.splitter.split_text(content)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception as e:
            logger.error(f"Error in standard chunking: {e}")
            # Fallback to simple splitting
            return self._fallback_chunking(content)

    def _hybrid_chunking(self, content: str) -> List[str]:
        """Hybrid chunking approach"""
        # Step 1: Sentence-based splitting
        self._init_sentence_splitter()
        initial_chunks = self.splitter.split_text(content)

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
            else:
                final_chunks.append(chunk)

        return [chunk.strip() for chunk in final_chunks if chunk.strip()]

    def _fallback_chunking(self, content: str) -> List[str]:
        """Fallback chunking method"""
        # Simple sentence-based splitting
        sentences = re.split(r'[.!?]+\s+', content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < self.config.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _post_process_chunks(
            self,
            chunks: List[str],
            base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Post-process chunks với metadata"""

        processed_chunks = []

        for i, chunk in enumerate(chunks):
            # Skip chunks quá ngắn
            if len(chunk.strip()) < self.config.min_chunk_size:
                continue

            # Create chunk metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_tokens": self._count_tokens(chunk),
                "chunking_strategy": self.config.strategy.value,
                "chunk_overlap": self.config.chunk_overlap
            })

            processed_chunks.append({
                "content": chunk.strip(),
                "metadata": chunk_metadata
            })

        return processed_chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens trong text"""
        # Simple token counting (có thể improve với tiktoken)
        return len(text.split())

    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics về chunking results"""
        if not chunks:
            return {}

        chunk_sizes = [len(chunk["content"]) for chunk in chunks]
        chunk_tokens = [chunk["metadata"]["chunk_tokens"] for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_tokens_per_chunk": sum(chunk_tokens) / len(chunk_tokens),
            "total_characters": sum(chunk_sizes),
            "total_tokens": sum(chunk_tokens),
            "strategy_used": self.config.strategy.value
        }

        return stats


# Preset configurations cho different use cases
class ChunkingPresets:
    """Pre-configured chunking setups"""

    @staticmethod
    def uie_optimized() -> ChunkingConfig:
        """Optimized cho UIE tasks"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=300,
            chunk_overlap=50,
            preserve_sentences=True,
            min_chunk_size=50,
            max_chunk_size=400
        )

    @staticmethod
    def fast_processing() -> ChunkingConfig:
        """Fast processing với basic chunking"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=500,
            chunk_overlap=50,
            preserve_sentences=False
        )

    @staticmethod
    def high_quality() -> ChunkingConfig:
        """High quality semantic chunking"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=250,
            chunk_overlap=30,
            model_name="Qwen/Qwen3-Embedding-0.6B",
            similarity_threshold=0.8
        )

    @staticmethod
    def academic_papers() -> ChunkingConfig:
        """Optimized cho academic papers"""
        return ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=400,
            chunk_overlap=0,
            preserve_sentences=False,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
