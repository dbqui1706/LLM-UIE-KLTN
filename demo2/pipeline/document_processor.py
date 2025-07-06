import logging
import torch
from pathlib import Path
from typing import List, Optional, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Add parent directory to path

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from memory import MemoryManager
from config import ProcessingMode, ProcessingStats, DocumentResult, SupportedFormat
from processing import clean_docling_output, quick_clean_metadata
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
    TableFormerMode,
    EasyOcrOptions,
)
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(
            self,
            processing_mode: ProcessingMode = ProcessingMode.BALANCED,
            use_gpu: bool = True,
            num_threads: int = 4,
            artifacts_path: Optional[str] = None,
            enable_ocr: bool = False,
            enable_table_structure: bool = False,
            max_workers: int = 2,
            # cleaning
            clean: bool = True,
            aggressive_clean: bool = True
    ):
        self.processing_mode = processing_mode
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.artifacts_path = artifacts_path
        self.enable_ocr = enable_ocr
        self.enable_table_structure = enable_table_structure
        self.max_workers = max_workers

        self.memory_manager = MemoryManager()
        self.stats = ProcessingStats()
        self.clean_text = clean
        self.aggressive_clean = aggressive_clean
        # Initialize converter
        self._setup_converter()

        logger.info(f"Initialized DocumentProcessor with mode: {processing_mode.value}")
        logger.info(f"GPU enabled: {self.use_gpu}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

    def _setup_converter(self):
        """Setup document converter with appropriate configurations"""

        # Configure accelerator
        if self.use_gpu and torch.cuda.is_available():
            device = AcceleratorDevice.AUTO  # Let Docling auto-detect best device
        else:
            device = AcceleratorDevice.CPU

        accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=device
        )

        # Configure pipeline based on processing mode
        pipeline_options = PdfPipelineOptions()

        if self.artifacts_path:
            pipeline_options.artifacts_path = self.artifacts_path

        # Configure accelerator
        pipeline_options.accelerator_options = accelerator_options

        # Configure OCR
        pipeline_options.do_ocr = self.enable_ocr
        if self.enable_ocr:
            ocr_options = EasyOcrOptions()
            ocr_options.use_gpu = self.use_gpu and torch.cuda.is_available()
            ocr_options.download_enabled = True
            pipeline_options.ocr_options = ocr_options

        # Configure table structure recognition
        pipeline_options.do_table_structure = self.enable_table_structure
        if self.enable_table_structure:
            if self.processing_mode == ProcessingMode.FAST:
                pipeline_options.table_structure_options.mode = TableFormerMode.FAST
            else:
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.table_structure_options.do_cell_matching = True

        # Choose backend based on processing mode
        if self.processing_mode == ProcessingMode.FAST:
            backend = PyPdfiumDocumentBackend  # Faster, less memory
        else:
            backend = DoclingParseV2DocumentBackend  # Better quality

        # Create converter
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=backend
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_options=pipeline_options,
                    backend=MsWordDocumentBackend
                )
            }
        )

        logger.info(f"Document converter configured for {self.processing_mode.value} mode")

    @staticmethod
    def _get_supported_files(file_path: Union[str, Path]) -> List[Path]:
        """Get list of supported files from path"""
        path = Path(file_path)
        supported_files = []

        # Mapping of supported extensions
        supported_extensions = {
            '.pdf': SupportedFormat.PDF,
            '.docx': SupportedFormat.DOCX,
            '.doc': SupportedFormat.DOC,
            '.md': SupportedFormat.MD,
            '.txt': SupportedFormat.TXT,
            '.html': SupportedFormat.HTML,
            '.htm': SupportedFormat.HTML,
        }

        if path.is_file():
            if path.suffix.lower() in supported_extensions:
                supported_files.append(path)
        elif path.is_dir():
            for ext in supported_extensions.keys():
                supported_files.extend(path.rglob(f"*{ext}"))

        return supported_files

    def process_single_document(self, file_path: Union[str, Path]) -> DocumentResult:
        """Process a single document"""
        file_path = Path(file_path)
        start_time = time.time()

        try:
            logger.info(f"Processing document: {file_path}")

            # Track memory before processing
            memory_before = self.memory_manager.get_memory_usage()

            # Convert document
            result = self.converter.convert(str(file_path))

            # Extract content and metadata
            content = result.document.export_to_markdown()

            # Get document metadata
            metadata = {
                'title': getattr(result.document, 'title', ''),
                'page_count': len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'format': file_path.suffix.lower(),
                'processing_mode': self.processing_mode.value
            }
            # Áp dụng cleaning đơn giản nếu enabled
            if self.clean_text and content:
                logger.info("Cleaning text...")
                original_length = len(content)
                content = clean_docling_output(content, self.aggressive_clean)
                metadata = quick_clean_metadata(metadata)
                logger.info(f"Text cleaned: {original_length} -> {len(content)} chars")
            processing_time = time.time() - start_time

            # Track memory after processing
            memory_after = self.memory_manager.get_memory_usage()
            memory_used = memory_after['rss'] - memory_before['rss']

            logger.info(f"Successfully processed {file_path} in {processing_time:.2f}s")
            logger.info(f"Memory used: {memory_used:.2f} MB")

            return DocumentResult(
                file_path=str(file_path),
                success=True,
                content=content,
                metadata=metadata,
                processing_time=processing_time,
                page_count=metadata['page_count']
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing {file_path}: {str(e)}")

            return DocumentResult(
                file_path=str(file_path),
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
        finally:
            # Clean up memory after each document
            self.memory_manager.cleanup_memory()

    def process_documents(
            self,
            input_path: Union[str, Path, List[Union[str, Path]]],
            output_dir: Optional[Union[str, Path]] = None,
            save_metadata: bool = True
    ) -> List[DocumentResult]:

        # Get list of files to process
        if isinstance(input_path, (str, Path)):
            files_to_process = self._get_supported_files(input_path)
        else:
            files_to_process = []
            for path in input_path:
                files_to_process.extend(self._get_supported_files(path))

        if not files_to_process:
            logger.warning("No supported files found to process")
            return []

        logger.info(f"Found {len(files_to_process)} files to process")

        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stats
        self.stats = ProcessingStats()
        self.stats.total_documents = len(files_to_process)

        results = []
        start_time = time.time()
        peak_memory = 0

        # Process files with threading for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_document, file_path): file_path
                for file_path in files_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Update stats
                    if result.success:
                        self.stats.successful_documents += 1
                        self.stats.total_pages += result.page_count
                    else:
                        self.stats.failed_documents += 1
                        self.stats.failed_files.append(result.file_path)

                    # Save output if requested
                    if output_dir and result.success:
                        self._save_result(result, output_dir, save_metadata)

                    # Track peak memory usage
                    current_memory = self.memory_manager.get_memory_usage()['rss']
                    peak_memory = max(peak_memory, current_memory)

                    # Log progress
                    completed = len(results)
                    logger.info(f"Progress: {completed}/{len(files_to_process)} documents processed")

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    self.stats.failed_documents += 1
                    self.stats.failed_files.append(str(file_path))

        # Finalize stats
        self.stats.total_processing_time = time.time() - start_time
        self.stats.peak_memory_usage = peak_memory

        if self.stats.total_pages > 0:
            self.stats.average_time_per_page = (
                    self.stats.total_processing_time / self.stats.total_pages
            )

        # Log final statistics
        self._log_final_stats()

        return results

    @staticmethod
    def _save_result(
            result: DocumentResult,
            output_dir: Path,
            save_metadata: bool
    ):
        """Save processing result to files"""
        file_path = Path(result.file_path)
        output_base = output_dir / file_path.stem

        # Save content as markdown
        if result.content:
            markdown_file = output_base.with_suffix('.md')
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(result.content)

        # Save metadata as JSON
        if save_metadata and result.metadata:
            import json
            metadata_file = output_base.with_suffix('.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result.metadata, f, indent=2, ensure_ascii=False)

    def _log_final_stats(self):
        """Log final processing statistics"""
        logger.info("=" * 60)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total documents: {self.stats.total_documents}")
        logger.info(f"Successful: {self.stats.successful_documents}")
        logger.info(f"Failed: {self.stats.failed_documents}")
        logger.info(f"Total pages: {self.stats.total_pages}")
        logger.info(f"Total processing time: {self.stats.total_processing_time:.2f}s")
        logger.info(f"Average time per page: {self.stats.average_time_per_page:.2f}s")
        logger.info(f"Peak memory usage: {self.stats.peak_memory_usage:.2f} MB")

        if self.stats.failed_files:
            logger.warning(f"Failed files: {self.stats.failed_files}")

        # GPU stats if available
        gpu_memory = self.memory_manager.get_gpu_memory_usage()
        if gpu_memory:
            logger.info("GPU Memory Usage:")
            for gpu_id, memory_info in gpu_memory.items():
                logger.info(f"  {gpu_id}: {memory_info}")

        logger.info("=" * 60)

    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics"""
        return self.stats
