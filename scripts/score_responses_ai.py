import argparse
import sys
import os
import json
import openai 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import read_jsonl, write_jsonl

# -----------------------------
# Configuration
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def score_with_llm(text, response, language="en"):
    """
    Sends the text + response to GPT-4 and asks for scoring.
    Returns a dict with score (0-1), feedback, and reasoning.
    """
    prompt = f"""
You are an AI evaluator. Score the following response for correctness, relevance, and clarity.

Text: {text}
Response: {response}
Language: {language}

Return JSON in the format:
{{"score": 0-1, "feedback": "..."}}
"""
    try:
        # Note: newer openai versions use client = OpenAI(); client.chat.completions.create
        # But for compatibility with older scripts or keeping it simple as per prompt:
        # If user has new openai lib, this might fail (v1.0.0+).
        # I'll check if I can import OpenAI from openai and use that pattern which is safer for v1+
        if hasattr(openai, "OpenAI"):
             client = openai.OpenAI(api_key=OPENAI_API_KEY)
             completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
             content = completion.choices[0].message.content
        else:
            # Fallback for old openai < 1.0.0
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = completion.choices[0].message['content']
            
        return json.loads(content)
    except Exception as e:
        return {"score": None, "feedback": str(e)}

def main(input_path, output_path, language="en"):
    responses = read_jsonl(input_path)
    scored = []
    for r in responses:
        result = score_with_llm(r['text'], r['response'], language)
        r.update(result)
        scored.append(r)
    write_jsonl(scored, output_path)
    print(f"AI-scored responses saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to AI responses JSONL")
    parser.add_argument("--output", required=True, help="Path to save scored responses")
    parser.add_argument("--language", default="en", help="Language of responses (en/sw)")
    args = parser.parse_args()
    main(args.input, args.output, args.language)
