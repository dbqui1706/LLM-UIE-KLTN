import logging
import sys
from pathlib import Path
import time
from pprint import pprint
from tqdm.auto import tqdm
from typing import Dict, List
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)
from utils import *
from app.core.base import LLamaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UIEUi:
    MODEL_NAME = 'quidangz/LLama-8B-Instruct-MultiTask'

    def __init__(self):
        try:
            self.current_chunks = None
            self.model = LLamaModel(model_name=self.MODEL_NAME)  # Uncomment when model available
            # self.model = None

            self.load_status = "success" if (
                    self.model and hasattr(self.model, 'model') and self.model.model) else "failed"

            status_messages = {
                "success": f"✅ Model `{self.MODEL_NAME}` loaded successfully!",
                "failed": "⚠️ Model not loaded - using mock results"
            }
            logger.info(status_messages[self.load_status])

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.load_status = "failed"

    def extract_information(self, text, task, entity_types="", relation_types="",
                            event_types="", argument_types="", mode="flexible",
                            **kwargs):
        if not text or not text.strip():
            return {"error": "Please provide text to analyze"}

        logger.info(f"🔍 Extracting {task} from text (length: {len(text)})")

        try:
            user_schema = prepare_schema(entity_types, relation_types, event_types, argument_types)

            gen_params = build_generation_params(**kwargs)

            # Extract using model or mock
            result = self.model.extract(text=text, task=task, user_schema=user_schema, mode=mode, **gen_params)
            pprint(result)
            logger.info(f"✅ Extraction completed for task: {task}")
            return format_extraction_result(result, task, text, gen_params)

        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return {"error": f"Extraction failed: {str(e)}"}

    def extract_from_chunks(self, task, entity_types="", relation_types="",
                            event_types="", argument_types="", mode="flexible",
                            aggregate_results=True, filter_duplicates=True,
                            **kwargs):
        """Extract from chunks - sequential processing"""

        # Early validation
        validation_result = self._validate_chunk_extraction(task)
        if validation_result:
            return validation_result

        logger.info(f"🔍 Extracting {task} from {len(self.current_chunks)} chunks (sequential)")

        try:
            user_schema = prepare_schema(entity_types, relation_types, event_types, argument_types)
            gen_params = build_generation_params(**kwargs)

            # Process chunks
            all_results, processing_time = self._process_all_chunks(task, user_schema, mode, gen_params)

            # Aggregate
            aggregated = aggregate_chunk_results(all_results, task, filter_duplicates) if aggregate_results else {
                'entities': [], 'relations': [], 'events': []}

            # Build result
            result = self._build_chunk_result(task, all_results, aggregated, processing_time, aggregate_results,
                                              filter_duplicates, gen_params)

            extracted_count = len(aggregated.get(get_result_key(task), []))
            logger.info(f"✅ Sequential extraction completed: {extracted_count} {task} items extracted")
            return result

        except Exception as e:
            logger.error(f"❌ Chunk extraction failed: {e}")
            return {"error": f"Chunk extraction failed: {str(e)}"}

    def get_current_chunks(self):
        if self.current_chunks is None:
            logger.warning("No chunks available. Please process a document first.")
            return []
        return self.current_chunks

    def set_current_chunks(self, chunks):
        self.current_chunks = chunks
        logger.info(f"📦 Set {len(chunks)} chunks for processing")

    def _validate_chunk_extraction(self, task: str) -> Dict:
        if not validate_chunk_extraction_task(task):
            return {"error": "Chunks extraction only supports NER, RE, or EE tasks (not ALL)"}

        if not self.current_chunks:
            return {"error": "No chunks available. Please process a document first."}

        return None

    def _process_all_chunks(self, task: str, user_schema: Dict, mode: str, gen_params: Dict) -> tuple:
        all_results = []
        start_time = time.time()

        for i, chunk in tqdm(
                enumerate(self.current_chunks),
                total=len(self.current_chunks),
                desc="Processing chunks",
        ):
            chunk_result = self._process_single_chunk(chunk, task, user_schema, mode, gen_params)
            all_results.append(chunk_result)

        processing_time = time.time() - start_time
        return all_results, processing_time

    def _process_single_chunk(self, chunk: Dict, task: str, user_schema: Dict, mode: str, gen_params: Dict) -> Dict:
        chunk_text = chunk['content']
        chunk_id = chunk['metadata']['chunk_id']

        try:
            result = self.model.extract(text=chunk_text, task=task, user_schema=user_schema, mode=mode,
                                        **gen_params)
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

    def _build_chunk_result(self, task: str, all_results: List, aggregated: Dict, processing_time: float,
                            aggregate_results: bool, filter_duplicates: bool, gen_params: Dict) -> Dict:
        return {
            'task': task,
            'chunks_processed': len(self.current_chunks),
            'processing_time': processing_time,
            'aggregate_results': aggregate_results,
            'filter_duplicates': filter_duplicates,
            'aggregated_results': aggregated,
            'per_chunk_results': all_results,
            'generation_parameters': gen_params,
            'performance_metrics': {
                'avg_time_per_chunk': processing_time / len(self.current_chunks),
                'chunks_per_second': len(self.current_chunks) / processing_time if processing_time > 0 else 0
            }
        }
