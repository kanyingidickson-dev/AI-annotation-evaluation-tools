import argparse
import sys
import pandas as pd # type: ignore
from pathlib import Path

# Add src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from ai_eval_tools.utils.io_utils import read_jsonl
from ai_eval_tools.utils.logger import logger

def analyze_dataset(input_path: str):
    logger.info(f"Analyzing metrics for: {input_path}")
    
    try:
        data = read_jsonl(input_path)
        if not data:
            logger.warning("Dataset is empty.")
            return

        df = pd.DataFrame(data)
        
        print("\n" + "="*50)
        print("Dataset Stats Analytics")
        print("="*50)
        
        print(f"Total Records: {len(df)}")
        
        if 'label' in df.columns:
            print("\nLabel Distribution:")
            print(df['label'].value_counts().to_string())
            
        if 'score' in df.columns:
            # Ensure score is numeric
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            print("\nScore Statistics:")
            print(df['score'].describe().to_string())
            
            print("\nLow Confidence Items (Score < 0.5):")
            low_conf = df[df['score'] < 0.5]
            print(f"Count: {len(low_conf)}")
            if not low_conf.empty:
                print("IDs:", low_conf['id'].tolist()[:10], "...(truncated)" if len(low_conf) > 10 else "")
                
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Analytics failure: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate statistical analytics for annotated datasets.")
    parser.add_argument("--input", required=True, help="Path to processed JSONL file.")
    
    args = parser.parse_args()
    analyze_dataset(args.input)
