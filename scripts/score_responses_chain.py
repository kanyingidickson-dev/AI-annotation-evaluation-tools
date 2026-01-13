import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import read_jsonl, write_jsonl
from utils.llm_client import LLMClient

def main(input_path, output_path, model="gpt-4", api_key=None, language="en"):
    llm = LLMClient(model=model, api_key=api_key)
    responses = read_jsonl(input_path)
    scored = []

    for r in responses:
        result = llm.score_response(
            r['text'], r['response'], language=language,
            rubric="Score 0-1 for accuracy, relevance, clarity, and cultural correctness."
        )
        r.update(result)
        scored.append(r)

    write_jsonl(scored, output_path)
    print(f"LangChain-scored responses saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to AI responses JSONL")
    parser.add_argument("--output", required=True, help="Path to save scored responses")
    parser.add_argument("--model", default="gpt-4", help="LLM model name")
    parser.add_argument("--api_key", default=None, help="API key for the LLM")
    parser.add_argument("--language", default="en", help="Language of responses (en/sw)")
    args = parser.parse_args()

    main(args.input, args.output, model=args.model, api_key=args.api_key, language=args.language)
