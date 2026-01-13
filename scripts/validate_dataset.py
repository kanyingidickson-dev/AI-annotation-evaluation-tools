import argparse
import sys
import pandas as pd # type: ignore
from pathlib import Path
from typing import List

# Add src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from ai_eval_tools.utils.io_utils import read_jsonl
from ai_eval_tools.utils.logger import logger

def validate(input_path: str, export_csv: bool = False):
    logger.info(f"Validating dataset: {input_path}")
    data = read_jsonl(input_path)
    
    df = pd.DataFrame(data)
    
    report = {
        "Total Entries": len(df),
        "Missing Labels": df['label'].isnull().sum() if 'label' in df.columns else "N/A",
        "Duplicate Texts": df.duplicated(subset=['text']).sum() if 'text' in df.columns else 0
    }
    
    print("\n=== Validation Report ===")
    for k, v in report.items():
        print(f"{k:<20}: {v}")
    
    if export_csv:
        csv_path = input_path.replace(".jsonl", "_report.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Report exported to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv", action="store_true", help="Export to CSV")
    args = parser.parse_args()
    validate(args.input, args.csv)
