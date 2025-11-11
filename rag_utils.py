from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model (small and efficient)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the annotated knowledge base
with open("legal_knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

# Precompute embeddings for each label
embeddings = {label: model.encode(texts) for label, texts in knowledge.items()}

def retrieve_answer(question, top_k=2):
    """
    Retrieve the top-k most relevant text chunks for a given lawyer question.
    Returns: (best_label, best_score, top_contexts)
    """
    q_emb = model.encode([question])
    best_label, best_score, top_contexts = None, 0, []

    # Find the most semantically similar label and its top chunks
    for label, embs in embeddings.items():
        sims = cosine_similarity(q_emb, embs)[0]
        # Get indices of top-k chunks (sorted by descending similarity)
        top_indices = np.argsort(sims)[::-1][:top_k]
        avg_score = np.mean(sims[top_indices])

        if avg_score > best_score:
            best_label = label
            best_score = avg_score
            # Pick the actual top-k chunks in ranked order
            top_contexts = [knowledge[label][i] for i in top_indices]

    return best_label, best_score, top_contexts
