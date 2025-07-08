from typing import Tuple, Dict
from .base import BaseHandler
import time
import os
from .utils import (
    create_chunks_file,
    create_chunks_preview,
    create_processing_summary,
    build_processing_result
)
from ...utils.cache_manager import get_cache
from ...pipeline.document_processor import DocumentProcessor
from ...pipeline.chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy
from ...pipeline.config import ProcessingMode


class DocumentCache:
    def __init__(self, content, metadata, processing_time):
        self.content = content
        self.metadata = metadata
        self.processing_time = processing_time
        self.success = True


class DocumentProcessingHandler(BaseHandler):
    """Handler chuyên biệt cho document processing"""

    def __init__(self, context):
        super().__init__(context)
        self.document_processor = None
        self.document_chunker = None
        self.cache = get_cache()

    def process_document_upload(self, file, processing_mode: str, use_gpu: bool,
                                enable_ocr: bool, enable_table_structure: bool,
                                enable_cleaning: bool, aggressive_clean: bool,
                                enable_chunking: bool, chunking_strategy: str,
                                chunk_size: int, chunk_overlap: int,
                                *generation_params) -> Tuple[str, str, str, str]:

        # Validation
        if file is None:
            return self._create_error_response("Please upload a file first", "Document Processing") + ("",)

        self._log_operation("Document processing", filename=file.name, mode=processing_mode)

        try:
            cached_doc = self.cache.get_cached_document(file.name)
            if cached_doc and self._is_processing_config_same(cached_doc, processing_mode, use_gpu, enable_ocr,
                                                              enable_table_structure, enable_cleaning,
                                                              aggressive_clean):
                result = DocumentCache(
                    content=cached_doc['content'],
                    metadata=cached_doc['metadata'],
                    processing_time=0.1
                )
            else:
                self.logger.info("🔄 Processing document fresh")

                # initialize processor
                self.document_processor = DocumentProcessor(
                    processing_mode=ProcessingMode(processing_mode),
                    use_gpu=use_gpu,
                    enable_ocr=enable_ocr,
                    enable_table_structure=enable_table_structure,
                    clean=enable_cleaning,
                    aggressive_clean=aggressive_clean,
                    max_workers=1
                )

                # process document
                result = self.document_processor.process_single_document(file.name)

                if not result.success:
                    return self._create_error_response(result.error_message, "Document Processing") + ("",)

                # cache processed document
                self.cache.cache_document(file.name, result.content, result.metadata)

                self.logger.info(f"✅ Document processed in {result.processing_time:.2f}s")

            # chunking
            chunks, chunks_preview = self._handle_chunking(
                enable_chunking, result, chunking_strategy, chunk_size,
                chunk_overlap
            )

            # If chunking is enabled, create download file
            download_file = None
            if chunks:
                download_file = create_chunks_file(chunks, chunking_strategy)

            processing_result = build_processing_result(
                file, result, processing_mode, use_gpu, enable_ocr,
                enable_table_structure, enable_cleaning, aggressive_clean,
                enable_chunking, chunking_strategy, chunk_size, chunk_overlap,
                chunks, generation_params
            )

            summary = create_processing_summary(processing_result)
            detailed_json = self._create_success_response(summary, processing_result)[1]

            return summary, detailed_json, chunks_preview, download_file

        except Exception as e:
            return self._create_error_response(str(e), "Document Processing") + ("",)

    def _handle_chunking(self, enable_chunking: bool, result, chunking_strategy: str,
                         chunk_size: int, chunk_overlap: int) -> Tuple[list, str]:

        if not enable_chunking or not result.content:
            return [], "Chunking disabled"

        self.logger.info(
            f"🔄 Chunking with current settings: {chunking_strategy}, size={chunk_size}, overlap={chunk_overlap}")

        # Create chunking config
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy(chunking_strategy),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=50,
            max_chunk_size=chunk_size * 2
        )

        # Initialize chunker
        self.document_chunker = DocumentChunker(chunking_config)

        # Perform chunking
        chunks = self.document_chunker.chunk_document(
            content=result.content,
            metadata=result.metadata
        )

        self.logger.info(f"✅ Created {len(chunks)} chunks")

        # Save chunks to context
        self.context.set_current_chunks(chunks)

        # Create preview
        chunks_preview = create_chunks_preview(chunks)

        return chunks, chunks_preview

    def _is_processing_config_same(self, cached_doc: dict, processing_mode: str, use_gpu: bool,
                                   enable_ocr: bool, enable_table_structure: bool,
                                   enable_cleaning: bool, aggressive_clean: bool) -> bool:
        """✅ Check if processing config matches cached version"""
        if not cached_doc or 'metadata' not in cached_doc:
            return False

        cached_metadata = cached_doc.get('metadata', {})
        current_mode = cached_metadata.get('processing_mode')
        return current_mode == processing_mode
