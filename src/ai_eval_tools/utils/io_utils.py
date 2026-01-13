import json
from typing import List, Dict, Any, Union
from pathlib import Path
from .logger import logger

def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Reads a JSONL file robustly."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): 
                    continue
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {i+1} in {path}")
        logger.info(f"Loaded {len(data)} entries from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        raise

def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Writes a list of dicts to JSONL atomically (ish)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(data)} entries to {path}")
    except Exception as e:
        logger.error(f"Failed to write to {path}: {e}")
        raise
