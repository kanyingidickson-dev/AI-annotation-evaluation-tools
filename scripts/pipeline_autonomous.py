import argparse
import sys
import os
from langchain.schema import HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import read_jsonl, write_jsonl
from utils.llm_client import LLMClient
from utils.embedding_utils import save_embeddings

def auto_annotate(text, llm_client, language="en"):
    """
    Uses LLM to produce an initial annotation.
    """
    prompt = f"""
You are a Swahili/English text annotator.
Text: {text}
Language: {language}
Task: Provide a concise label or category for this text.
Return only the label.
"""
    # Using the client directly to generate text (not json scored)
    response = llm_client.client([HumanMessage(content=prompt)])
    return response.content.strip()

def pipeline(input_path, output_path, embeddings_path, model="gpt-4", api_key=None, language="en"):
    llm = LLMClient(model=model, api_key=api_key)
    dataset = read_jsonl(input_path)
    
    annotated = []
    print(f"Starting pipeline for {len(dataset)} entries...")
    for entry in dataset:
        # Auto annotation
        print(f"Annotating ID: {entry.get('id')}...")
        label = auto_annotate(entry['text'], llm, language=language)
        entry['label'] = label
        
        # AI-powered evaluation (score + feedback)
        print(f"Scoring ID: {entry.get('id')}...")
        score_result = llm.score_response(entry['text'], label, language=language)
        entry.update(score_result)
        
        annotated.append(entry)
    
    # Save annotated & scored dataset
    write_jsonl(annotated, output_path)
    print(f"Annotated + scored data saved to {output_path}")

    # Create embeddings for RAG
    print("Generating embeddings...")
    save_embeddings(annotated, embeddings_path)
    print(f"Embeddings saved to {embeddings_path} for RAG pipeline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw JSONL dataset")
    parser.add_argument("--output", required=True, help="Path to save annotated+scored dataset")
    parser.add_argument("--embeddings", required=True, help="Path to save embeddings JSON")
    parser.add_argument("--model", default="gpt-4", help="LLM model name")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="API key for LLM")
    parser.add_argument("--language", default="en", help="Language (en/sw)")
    args = parser.parse_args()

    pipeline(args.input, args.output, args.embeddings, args.model, args.api_key, args.language)
