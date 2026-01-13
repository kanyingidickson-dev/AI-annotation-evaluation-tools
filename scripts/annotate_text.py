import argparse
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import read_jsonl, write_jsonl

def annotate_entry(entry):
    print(f"\nID: {entry['id']}")
    print(f"Text: {entry['text']}")
    label = input("Enter label: ")
    entry['label'] = label
    return entry

def main(input_path, output_path):
    dataset = read_jsonl(input_path)
    annotated = []
    for entry in dataset:
        annotated.append(annotate_entry(entry))
    write_jsonl(annotated, output_path)
    print(f"\nAnnotated data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output", required=True, help="Path to save annotated dataset")
    args = parser.parse_args()
    main(args.input, args.output)
