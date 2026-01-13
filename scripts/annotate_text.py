import argparse
import sys
from pathlib import Path

# Add src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from ai_eval_tools.utils.io_utils import read_jsonl, write_jsonl
from ai_eval_tools.utils.logger import logger

def annotate_entry(entry: dict) -> dict:
    print("\n" + "="*50)
    print(f"ID: {entry.get('id')}")
    print(f"Text: {entry.get('text')}")
    print("-" * 50)
    label = input("Input Label (or 'skip' to skip): ").strip()
    if label.lower() != 'skip':
        entry['label'] = label
    return entry

def main(input_path: str, output_path: str):
    logger.info("Starting manual annotation session...")
    try:
        dataset = read_jsonl(input_path)
        annotated = []
        
        for i, entry in enumerate(dataset):
            logger.info(f"Processing {i+1}/{len(dataset)}")
            annotated.append(annotate_entry(entry))
            
        write_jsonl(annotated, output_path)
        logger.info("Session complete.")
        
    except KeyboardInterrupt:
        logger.warning("\nSession interrupted by user. Saving progress...")
        if annotated:
             write_jsonl(annotated, output_path)
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)
