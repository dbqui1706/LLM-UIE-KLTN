from typing import Tuple, Dict
from .base import BaseHandler
import time
import os
from ...utils.cache_manager import get_cache

# Import with error handling
try:
    from ...pipeline.document_processor import DocumentProcessor
    from ...pipeline.chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy
    from ...pipeline.config import ProcessingMode
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False

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
                           chunk_size: int, chunk_overlap: int, task_type: str,
                           *generation_params) -> Tuple[str, str, str, str]:
        """Process document upload và chunking"""
        
        # Validation
        if file is None:
            return self._create_error_response("Please upload a file first", "Document Processing") + ("",)
        
        if not PIPELINE_AVAILABLE:
            return self._create_error_response("Document processing pipeline not available", "Document Processing") + ("",)
        
        self._log_operation("Document processing", filename=file.name, mode=processing_mode)
        
        try:
            cached_doc = self.cache.get_cached_document(file.name)
            if cached_doc and self._is_processing_config_same(cached_doc, processing_mode, use_gpu, enable_ocr, enable_table_structure, enable_cleaning, aggressive_clean):
                result = DocumentCache(
                    content=cached_doc['content'],
                    metadata=cached_doc['metadata'],
                    processing_time= 0.1
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
                chunk_overlap, task_type
            )
            download_file = None
            if chunks:
                download_file = self._create_chunks_file(chunks)
           
            processing_result = self._build_processing_result(
                file, result, processing_mode, use_gpu, enable_ocr, 
                enable_table_structure, enable_cleaning, aggressive_clean,
                enable_chunking, chunking_strategy, chunk_size, chunk_overlap,
                task_type, chunks, generation_params
            )
            
            summary = self._create_processing_summary(processing_result)
            detailed_json = self._create_success_response(summary, processing_result)[1]
            
            return summary, detailed_json, chunks_preview, download_file
            
        except Exception as e:
            return self._create_error_response(str(e), "Document Processing") + ("",)
    
    def _handle_chunking(self, enable_chunking: bool, result, chunking_strategy: str,
                        chunk_size: int, chunk_overlap: int, task_type: str) -> Tuple[list, str]:
        """Handle document chunking"""
        
        if not enable_chunking or not result.content:
            return [], "Chunking disabled"
        
        self.logger.info(f"🔄 Chunking with current settings: {chunking_strategy}, size={chunk_size}, overlap={chunk_overlap}")
        
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
            metadata=result.metadata,
            task_type=task_type
        )
        
        self.logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Save chunks to context
        self.context.current_chunks = chunks
        
        # Create preview
        chunks_preview = self._create_chunks_preview(chunks)
        
        return chunks, chunks_preview
    
    def _build_processing_result(self, file, result, processing_mode, use_gpu, 
                                enable_ocr, enable_table_structure, enable_cleaning,
                                aggressive_clean, enable_chunking, chunking_strategy,
                                chunk_size, chunk_overlap, task_type, chunks, 
                                generation_params) -> Dict:
        """Build comprehensive processing result"""
        
        return {
            'file_info': {
                'name': file.name,
                'size': result.metadata.get('file_size', 0),
                'format': result.metadata.get('format', ''),
                'pages': result.metadata.get('page_count', 0)
            },
            'processing': {
                'mode': processing_mode,
                'time': result.processing_time,
                'success': result.success,
                'use_gpu': use_gpu,
                'ocr_enabled': enable_ocr,
                'table_structure': enable_table_structure
            },
            'content': {
                'original_length': len(result.content),
                'cleaned_length': len(result.content),
                'cleaning_enabled': enable_cleaning,
                'aggressive_clean': aggressive_clean
            },
            'chunking': {
                'enabled': enable_chunking,
                'strategy': chunking_strategy if enable_chunking else None,
                'chunk_size': chunk_size if enable_chunking else None,
                'chunk_overlap': chunk_overlap if enable_chunking else None,
                'total_chunks': len(chunks),
                'task_optimized_for': task_type if enable_chunking else None
            },
            'chunks_sample': chunks[:3],
            'generation_params': {
                'max_new_tokens': generation_params[0] if generation_params else None,
                'temperature': generation_params[1] if generation_params else None,
                'top_p': generation_params[2] if generation_params else None,
                'top_k': generation_params[3] if generation_params else None
            }
        }
    
    def _create_processing_summary(self, result: Dict) -> str:
        """Create processing summary"""
        file_info = result['file_info']
        processing = result['processing']
        chunking = result['chunking']
        
        return f"""
## 📄 Document Processing Results

### 📁 File: {os.path.basename(file_info['name'])} ({file_info['size'] / 1024 / 1024:.2f} MB)
### ⚡ Processing: {processing['mode']} mode - {processing['time']:.2f}s
### ✂️ Chunking: {chunking['total_chunks']} chunks created
### 🎛️ GPU: {'✅' if processing['use_gpu'] else '❌'} | OCR: {'✅' if processing['ocr_enabled'] else '❌'}
        """.strip()
    
    def _create_chunks_preview(self, chunks: list) -> str:
        """Create chunks preview"""
        if not chunks:
            return "No chunks available for preview."
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        preview = f"""
## 📊 Chunking Statistics
- **Total Chunks:** {len(chunks)}
- **Average Size:** {avg_size:.0f} chars
- **Size Range:** {min(chunk_sizes)} - {max(chunk_sizes)} chars

## 👀 Sample Chunks Preview
"""
        
        for i, chunk in enumerate(chunks[:3]):
            content_preview = chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
            preview += f"\n**Chunk {i+1}:** {content_preview}\n"
        
        if len(chunks) > 3:
            preview += f"\n... and {len(chunks) - 3} more chunks ready for processing."
        
        return preview
    
    def _create_chunks_file(self, chunks: list) -> str:
        import tempfile
        import json
        from datetime import datetime
        
        # Create temp file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=f'_chunks_{timestamp}.txt',
            delete=False,
            encoding='utf-8'
        )
        
        # Write chunks to file
        temp_file.write(f"Document Chunks Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        temp_file.write("=" * 80 + "\n\n")
        
        for i, chunk in enumerate(chunks):
            temp_file.write(f"CHUNK {i+1}/{len(chunks)}\n")
            temp_file.write("-" * 40 + "\n")
            temp_file.write(f"Size: {len(chunk['content'])} chars\n")
            temp_file.write(f"Strategy: {chunk['metadata'].get('chunking_strategy', 'unknown')}\n")
            temp_file.write(f"Content:\n{chunk['content']}\n\n")
        
        temp_file.close()
        return temp_file.name
    
    def _is_processing_config_same(self, cached_doc: dict, processing_mode: str, use_gpu: bool, 
                                 enable_ocr: bool, enable_table_structure: bool, 
                                 enable_cleaning: bool, aggressive_clean: bool) -> bool:
        """✅ Check if processing config matches cached version"""
        cached_metadata = cached_doc.get('metadata', {})
        
        current_mode = cached_metadata.get('processing_mode')
        return current_mode == processing_mode