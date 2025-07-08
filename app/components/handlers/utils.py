from typing import Dict, List, Any
import os

def build_processing_result(file, result, processing_mode, use_gpu,
                            enable_ocr, enable_table_structure, enable_cleaning,
                            aggressive_clean, enable_chunking, chunking_strategy,
                            chunk_size, chunk_overlap, chunks,
                            generation_params) -> Dict:
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
        },
        'chunks_sample': chunks[:3],
        'generation_params': {
            'max_new_tokens': generation_params[0] if generation_params else None,
            'temperature': generation_params[1] if generation_params else None,
            'top_p': generation_params[2] if generation_params else None,
            'top_k': generation_params[3] if generation_params else None
        }
    }


def create_processing_summary(result: Dict) -> str:
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


def create_chunks_preview(chunks: list) -> str:
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
        preview += f"\n**Chunk {i + 1}:** {content_preview}\n"

    if len(chunks) > 3:
        preview += f"\n... and {len(chunks) - 3} more chunks ready for processing."

    return preview


def create_chunks_file(chunks: list, strategy: str) -> str:
    import tempfile
    import json
    from datetime import datetime

    # Create temp file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix=f'_{strategy}_chunks_{timestamp}.txt',
        delete=False,
        encoding='utf-8'
    )

    # Write chunks to file
    temp_file.write(f"Document Chunks Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    temp_file.write("=" * 80 + "\n\n")

    for i, chunk in enumerate(chunks):
        temp_file.write(f"CHUNK {i + 1}/{len(chunks)}\n")
        temp_file.write("-" * 40 + "\n")
        temp_file.write(f"Size: {len(chunk['content'])} chars\n")
        temp_file.write(f"Strategy: {chunk['metadata'].get('chunking_strategy', 'unknown')}\n")
        temp_file.write(f"Content:\n{chunk['content']}\n\n")

    temp_file.close()
    return temp_file.name


def create_extraction_summary(result: Dict) -> str:
    parts = []

    if "entities" in result:
        parts.append(f"🏷️ Entities: {len(result['entities'])}")
    if "relations" in result:
        parts.append(f"🔗 Relations: {len(result['relations'])}")
    if "events" in result:
        parts.append(f"📅 Events: {len(result['events'])}")

    if "generation_info" in result:
        gen_info = result["generation_info"]["parameters_used"]
        parts.append(f"🎛️ Temp: {gen_info.get('temperature', 'N/A')}")
        parts.append(f"🎯 Tokens: {gen_info.get('max_new_tokens', 'N/A')}")

    return " | ".join(parts) if parts else "No results found"
