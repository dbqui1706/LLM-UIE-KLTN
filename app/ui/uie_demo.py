# app/ui/uie_demo.py
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import logging
import time
from tqdm.auto import tqdm
from typing import Dict, List
from utils import *
from app.core.base import LLamaModel

logger = logging.getLogger(__name__)


class UIEUi:
    MODEL_NAME = 'quidangz/LLama-8B-Instruct-MultiTask'

    def __init__(self):
        try:
            self.current_chunks = None
            self.model = LLamaModel(model_name=self.MODEL_NAME)
            self.last_text_results = None
            self.last_document_results = None

            self.load_status = "success" if (
                        self.model and hasattr(self.model, 'model') and self.model.model) else "failed"

            logger.info(f"✅ Model loaded successfully!" if self.load_status == "success" else "⚠️ Model not loaded")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.load_status = "failed"

    def extract_information(self, text, task, entity_types="", relation_types="", event_types="", argument_types="",
                            mode="flexible", **kwargs):
        if not text or not text.strip():
            return {"error": "Please provide text to analyze"}

        logger.info(f"🔍 Extracting {task} from text (length: {len(text)})")

        try:
            user_schema = prepare_schema(entity_types, relation_types, event_types, argument_types)
            gen_params = build_generation_params(**kwargs)

            result = self.model.extract(text=text, task=task, user_schema=user_schema, mode=mode, **gen_params)

            # Save for visualization
            self.last_text_results = result

            return format_extraction_result(result, task, text, gen_params)

        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return {"error": f"Extraction failed: {str(e)}"}

    def extract_from_chunks(self, task, entity_types="", relation_types="", event_types="", argument_types="",
                            mode="flexible", aggregate_results=True, filter_duplicates=True, **kwargs):
        """Extract from chunks with proper aggregation"""

        if not validate_chunk_extraction_task(task):
            return {"error": "Chunks extraction only supports NER, RE, or EE tasks"}

        if not self.current_chunks:
            return {"error": "No chunks available. Please process a document first."}

        logger.info(f"🔍 Extracting {task} from {len(self.current_chunks)} chunks")

        try:
            user_schema = prepare_schema(entity_types, relation_types, event_types, argument_types)
            gen_params = build_generation_params(**kwargs)

            # Process all chunks
            all_results = []
            start_time = time.time()

            for chunk in tqdm(self.current_chunks, desc="Processing chunks"):
                chunk_result = self._process_single_chunk(chunk, task, user_schema, mode, gen_params)
                all_results.append(chunk_result)

            processing_time = time.time() - start_time

            # ✅ Aggregate with fixed logic
            aggregated = aggregate_chunk_results(all_results, task, filter_duplicates) if aggregate_results else {
                'entities': [], 'relations': [], 'events': []}

            # Build final result
            result = {
                'task': task,
                'chunks_processed': len(self.current_chunks),
                'processing_time': processing_time,
                'aggregate_results': aggregate_results,
                'filter_duplicates': filter_duplicates,
                'aggregated_results': aggregated,
                'per_chunk_results': all_results,
                'generation_parameters': gen_params
            }

            # Get count for logging
            result_key = {'NER': 'entities', 'RE': 'relations', 'EE': 'events'}.get(task.upper(), 'entities')
            extracted_count = len(aggregated.get(result_key, []))
            logger.info(
                f"✅ Extraction completed: {extracted_count} {task} items from {len(self.current_chunks)} chunks")

            self.last_document_results = result
            return result

        except Exception as e:
            logger.error(f"❌ Chunk extraction failed: {e}")
            return {"error": f"Chunk extraction failed: {str(e)}"}

    def _process_single_chunk(self, chunk: Dict, task: str, user_schema: Dict, mode: str, gen_params: Dict) -> Dict:
        """Process single chunk"""
        chunk_text = chunk['content']
        chunk_id = chunk['metadata']['chunk_id']

        try:
            result = self.model.extract(text=chunk_text, task=task, user_schema=user_schema, mode=mode, **gen_params)
            chunk_result = format_chunk_extraction_result(result, task, chunk_id)
            chunk_result['chunk_metadata'] = chunk['metadata']
            return chunk_result

        except Exception as e:
            logger.warning(f"❌ Failed to process chunk {chunk_id}: {e}")
            return {
                'chunk_id': chunk_id,
                'entities': [],
                'relations': [],
                'events': [],
                'error': str(e),
                'chunk_metadata': chunk['metadata']
            }

    # Simple getters/setters
    def get_current_chunks(self):
        return self.current_chunks or []

    def set_current_chunks(self, chunks):
        self.current_chunks = chunks
        logger.info(f"📦 Set {len(chunks)} chunks for processing")

    def get_last_text_results(self):
        return self.last_text_results

    def get_last_document_results(self):
        return self.last_document_results