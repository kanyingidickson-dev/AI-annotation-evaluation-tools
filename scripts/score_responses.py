import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import read_jsonl, write_jsonl

def score_response(response, rubric=None):
    # Example: simple accuracy scoring (can extend later)
    score = 1 if response.get('correct', False) else 0
    response['score'] = score
    return response

def main(input_path, output_path):
    responses = read_jsonl(input_path)
    scored = [score_response(r) for r in responses]
    write_jsonl(scored, output_path)
    print(f"Scored responses saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to AI responses JSONL")
    parser.add_argument("--output", required=True, help="Path to save scored responses")
    args = parser.parse_args()
    main(args.input, args.output)
