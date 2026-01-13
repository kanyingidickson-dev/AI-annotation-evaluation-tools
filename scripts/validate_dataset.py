import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import read_jsonl

def check_missing_fields(dataset, required_fields):
    missing = []
    for entry in dataset:
        for field in required_fields:
            if field not in entry:
                missing.append(entry['id'])
                break
    return missing

def check_duplicates(dataset):
    seen = set()
    duplicates = []
    for entry in dataset:
        text = entry['text']
        if text in seen:
            duplicates.append(entry['id'])
        else:
            seen.add(text)
    return duplicates

def main(input_path):
    dataset = read_jsonl(input_path)
    required_fields = ['id', 'text']
    missing = check_missing_fields(dataset, required_fields)
    duplicates = check_duplicates(dataset)
    
    print("=== Dataset Validation Report ===")
    print(f"Total entries: {len(dataset)}")
    print(f"Entries missing required fields: {missing}")
    print(f"Duplicate entries: {duplicates}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    args = parser.parse_args()
    main(args.input)
