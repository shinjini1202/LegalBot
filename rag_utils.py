import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model (small and efficient)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the annotated knowledge base
with open("legal_knowledge.json", "r", encoding="utf-8") as f:
    knowledge = json.load(f)

# Load the questions mapping
with open("questions_mapping.json", "r", encoding="utf-8") as f:
    questions_mapping = json.load(f)

# Precompute embeddings for each label
embeddings = {label: model.encode(texts) for label, texts in knowledge.items()}

def retrieve_answer(question, top_k=2):
    """
    Retrieve the top-k most relevant text chunks using question mapping.
    Returns: (best_labels, relevance_score, top_contexts)
    """
    relevance_score = 0.0
    
    # First, try to find exact or similar question in mapping
    if question in questions_mapping:
        labels = questions_mapping[question]
        relevance_score = 1.0  # Perfect match
    else:
        # Fallback: use semantic search if question not in mapping
        q_emb = model.encode([question])
        best_label, best_score = None, 0
        
        for label, embs in embeddings.items():
            sims = cosine_similarity(q_emb, embs)[0]
            avg_score = np.mean(sims)
            if avg_score > best_score:
                best_label = label
                best_score = avg_score
        
        labels = [best_label] if best_label else []
        relevance_score = best_score
    
    # Fetch top-k chunks from mapped labels
    top_contexts = []
    for label in labels:
        if label in knowledge:
            top_contexts.extend(knowledge[label][:top_k])
    
    return labels, relevance_score, top_contexts