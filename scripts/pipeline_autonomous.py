import argparse
import sys
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List

# Ensure src is in path for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from ai_eval_tools.core.llm_client import LLMClient
from ai_eval_tools.utils.io_utils import read_jsonl, write_jsonl
from ai_eval_tools.utils.embeddings import EmbeddingManager
from ai_eval_tools.utils.logger import logger
from ai_eval_tools.core.schema import AnnotationEntry

def auto_annotate(text: str, llm_client: LLMClient, language: str = "en") -> str:
    """Uses LLM to produce an initial annotation label."""
    prompt = f"""
    Task: Categorize the following text with a single concise label.
    Text ({language}): {text}
    Output: ONLY the label.
    """
    return llm_client.generate_text(prompt)

def pipeline(
    input_path: str, 
    output_path: str, 
    embeddings_path: str, 
    model: str = "gpt-4", 
    api_key: Optional[str] = None, 
    language: str = "en"
):
    try:
        logger.info(f"Initializing Enterprise Pipeline with model={model}, language={language}")
        
        llm = LLMClient(model=model, api_key=api_key)
        embedding_mgr = EmbeddingManager(api_key=api_key)
        
        raw_data = read_jsonl(input_path)
        processed_data: List[AnnotationEntry] = []
        
        logger.info(f"Processing {len(raw_data)} entries...")
        
        for item in raw_data:
            try:
                # Validation
                entry = AnnotationEntry(**item)
                
                # 1. Auto-Annotation
                logger.info(f"Annotating ID {entry.id}...")
                label = auto_annotate(entry.text, llm, language)
                entry.label = label
                
                # 2. AI Evaluation
                logger.debug(f"Scoring classification for ID {entry.id}...")
                score_res = llm.score_response(entry.text, label, language=language)
                
                entry.score = score_res.get("score")
                entry.feedback = score_res.get("feedback")
                
                processed_data.append(entry)
                
            except Exception as e:
                logger.error(f"Failed to process entry {item.get('id')}: {e}")
                continue

        # Export
        export_data = [e.model_dump() for e in processed_data]
        write_jsonl(export_data, output_path)
        
        # Embeddings
        logger.info("Generating RAG embeddings...")
        embedding_mgr.save_embeddings(export_data, embeddings_path)
        
        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Critical Pipeline Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise AI Data Pipeline")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--embeddings", required=True, help="Embeddings output path")
    parser.add_argument("--model", default="gpt-4", help="Model name (gpt-4, claude-3-opus)")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="API Key")
    parser.add_argument("--language", default="en", help="Language code (en, sw, etc.)")
    
    args = parser.parse_args()
    pipeline(args.input, args.output, args.embeddings, args.model, args.api_key, args.language)
