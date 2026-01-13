# Enterprise AI Evaluation & Annotation Toolkit

A professional-grade, autonomous pipeline for Data Operations in AI Development. This toolkit provides a robust infrastructure for annotating datasets, evaluating LLM outputs, and generating embedding indices for RAG systems.

Designed for high-throughput environments requiring reliability, observability, and extensibility.

## Key Capabilities

### 1. Autonomous Data Pipeline
- **Automated Labeling**: Leverages SOTA LLMs (GPT-4, Claude 3) to categorize text data.
- **Auto-Evaluation**: Scoring rubrics (0-1) for accuracy, hallucination detection, and relevance.
- **RAG Optimization**: Automatically generates and indexes vector embeddings.
- **Enterprise Logging**: Structured logging and automatic retries using tenacity.

### 2. Quality Assurance
- Schema validation using Pydantic.
- Duplicate detection and coverage reporting using Pandas.

### 3. Analytics & Insights
- Statistical analysis of dataset distribution.
- Automated rubric generation for consistent evaluation.

### 4. Human-in-the-Loop
- CLI interface for manual review and correction of edge cases.

## Installation

**Prerequisites:** Python 3.9+

```bash
git clone <repo-url>
cd ai-annotation-evaluation-tools
pip install -r requirements.txt
```

## Quick Start

### Configuration
Set your environment variables (or use a .env file):
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
```

### 1. Run Autonomous Pipeline
Multilingual (e.g., Swahili or Spanish) ready.

```bash
python scripts/pipeline_autonomous.py \
  --input data/demo_texts_sw.jsonl \
  --output data/processed_sw.jsonl \
  --embeddings data/embeddings/swahili_index.json \
  --model gpt-4 \
  --language sw
```

### 2. Validate Results
```bash
python scripts/validate_dataset.py --input data/processed_sw.jsonl --csv
```

### 3. Generate Analytics
```bash
python scripts/analytics.py --input data/processed_sw.jsonl
```

## Project Structure

```
.
├── data/                   # Dataset storage
├── scripts/                # CLI Entry points
│   ├── pipeline_autonomous.py
│   ├── validate_dataset.py
│   ├── analytics.py
│   ├── generate_rubric.py
│   └── annotate_text.py
├── src/
│   └── ai_eval_tools/      # Core Package
│       ├── core/           # LLM Clients & Config
│       └── utils/          # IO, Logging, Embeddings
└── pyproject.toml          # Build configuration
```

## License
MIT License
