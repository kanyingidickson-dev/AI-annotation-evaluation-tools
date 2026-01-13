import json
import os
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..utils.logger import logger
from ..core.config import config

class EmbeddingManager:
    def __init__(self, api_key: str = None):
        key = api_key or config.OPENAI_API_KEY
        if key:
             self.client = OpenAI(api_key=key)
        else:
             logger.warning("OpenAI API Key missing for embeddings.")
             self.client = None

    def embed_text(self, text: str) -> List[float]:
        try:
            if not self.client:
                raise ValueError("Client not initialized.")
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def save_embeddings(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Generates and saves embeddings for a dataset.
        data: list of dicts with keys ['id', 'text']
        """
        embeddings = []
        logger.info("Starting embedding generation...")
        
        for i, item in enumerate(data):
            try:
                emb = self.embed_text(item['text'])
                embeddings.append({
                    "id": item.get('id'), 
                    "text": item.get('text'), 
                    "embedding": emb
                })
                if i % 10 == 0:
                     logger.debug(f"Processed {i}/{len(data)} embeddings.")
            except Exception as e:
                logger.error(f"Failed to process item {item.get('id')}: {e}")
                continue

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings file: {e}")
            raise

    @staticmethod
    def load_embeddings(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search_similar(self, query: str, embeddings: List[Dict], top_k: int = 3) -> List[Dict]:
        try:
            query_emb = np.array(self.embed_text(query)).reshape(1, -1)
            corpus_emb = np.array([e['embedding'] for e in embeddings])
            
            if len(corpus_emb) == 0:
                return []

            sims = cosine_similarity(query_emb, corpus_emb)[0]
            # argsort is ascending, so we take last top_k and reverse
            top_indices = sims.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    "id": embeddings[idx]['id'],
                    "text": embeddings[idx]['text'],
                    "score": float(sims[idx])
                })
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
