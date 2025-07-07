import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from .config import ProcessingMode
from .document_processor import DocumentProcessor
from .chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy, ChunkingPresets
import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('docling_pipeline.log')
        ]
    )


def main():
    setup_logging("INFO")
    processor = DocumentProcessor(
        processing_mode=ProcessingMode.FAST,
        use_gpu=False,
    )
    PATH = "D:\\python-project\\llm_multitask\\data\\docling.pdf"
    result = processor.process_single_document(PATH)
    content = result.content if result.content else "No content extracted"

    # chunking
    config = ChunkingConfig(strategy=ChunkingStrategy.SENTENCE, chunk_size=300, chunk_overlap=0)
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(result.content)

    if result.success:
        # print(f"Successfully processed: {result.file_path}")
        # print(f"Processing time: {result.processing_time:.2f}s")
        # print(f"Content length: {len(result.content)}")
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: {chunk['content']}")
        print
    else:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    main()
