import json
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def save_embeddings(data, file_path):
    """
    data: list of dicts with keys ['id', 'text']
    """
    embeddings = []
    for item in data:
        emb = embed_text(item['text'])
        embeddings.append({"id": item['id'], "text": item['text'], "embedding": emb})
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

def load_embeddings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_similar(query, embeddings, top_k=3):
    query_emb = np.array(embed_text(query)).reshape(1, -1)
    corpus_emb = np.array([e['embedding'] for e in embeddings])
    sims = cosine_similarity(query_emb, corpus_emb)[0]
    results = sorted(zip(sims, embeddings), key=lambda x: -x[0])[:top_k]
    return [{"id": r['id'], "text": r['text'], "score": s} for s, r in results]
