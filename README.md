# AI Annotation & Evaluation Toolkit

This repository demonstrates a **fully autonomous AI dataset pipeline** for multilingual data (Swahili + English).  

It performs:
- **LLM-assisted annotation**  
- **AI evaluation + scoring**  
- **RAG-ready embedding generation**  

This setup is ideal for high-value AI projects like those at Welocalize/Welo Data.

---

## Features
- Fully automated labeling using GPT-4 or Claude  
- AI scoring with feedback (accuracy, relevance, clarity, cultural correctness)  
- Embeddings generated for downstream retrieval applications  
- Multilingual support: Swahili + English  
- Batch-ready and modular  

---

## Setup

1. Clone the repo:

```bash
git clone https://github.com/kanyingidickson-dev/ai-annotation-evaluation-tools.git
cd ai-annotation-evaluation-tools
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

---

## Demo Usage

### Run the autonomous pipeline (Swahili dataset)

```bash
python scripts/pipeline_autonomous.py \
  --input data/demo_texts_sw.jsonl \
  --output data/demo_annotated_scored_sw.jsonl \
  --embeddings data/embeddings/demo_embeddings_sw.json \
  --model gpt-4 \
  --api_key $OPENAI_API_KEY \
  --language sw
```

### Run the autonomous pipeline (English dataset)

```bash
python scripts/pipeline_autonomous.py \
  --input data/demo_texts_en.jsonl \
  --output data/demo_annotated_scored_en.jsonl \
  --embeddings data/embeddings/demo_embeddings_en.json \
  --model gpt-4 \
  --api_key $OPENAI_API_KEY \
  --language en
```

---

## Output

* **Annotated + scored dataset**: JSONL with fields `id`, `text`, `label`, `score`, `feedback`
* **Embeddings JSON**: ready for retrieval-augmented generation (RAG) applications

---

## Extending the Toolkit

* Swap in your own datasets in Swahili/English
* Customize LLM prompts for specialized annotation
* Integrate into AI training pipelines or QA workflows

---

## Contact

* **Author**: Dickson Kanyingi
* **LinkedIn**: [https://www.linkedin.com/in/kanyingidickson-dev](https://www.linkedin.com/in/kanyingidickson-dev)
