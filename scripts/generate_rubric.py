import argparse
import sys
from pathlib import Path

# Add src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from ai_eval_tools.core.llm_client import LLMClient
from ai_eval_tools.utils.logger import logger

def generate_rubric(description: str, model: str = "gpt-4"):
    """
    Generates a professional evaluation rubric based on a dataset description.
    """
    logger.info(f"Generating rubric for description: {description}")
    
    prompt = f"""
    Act as a Senior Data Scientist.
    Create a detailed evaluation rubric for an AI dataset based on the following description:
    "{description}"
    
    The rubric should be 0-1 score based.
    Provide the output as a concise string suitable for inclusion in an LLM system prompt.
    Focus on Accuracy, Consistency, and Domain Relevance.
    """
    
    try:
        client = LLMClient(model=model)
        rubric = client.generate_text(prompt)
        
        print("\n" + "="*60)
        print("Generated Rubric Proposal")
        print("="*60)
        print(rubric)
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to generate rubric: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation rubrics using an LLM.")
    parser.add_argument("--desc", required=True, help="Description of the dataset or task.")
    parser.add_argument("--model", default="gpt-4", help="Model to use.")
    
    args = parser.parse_args()
    generate_rubric(args.desc, args.model)
