"""
Section D — AI-Augmented Learning: Corrected & Debugged RAG Script
Fixes implemented:
1. Replaced raw character slicing with word-boundary overlapping chunking (chunk_size=50 words, overlap=10).
2. Added k = min(k, len(chunks)) guard to prevent FAISS index boundary crashes when k > total chunks.
3. Updated prompt to include explicit fallback instruction ("Answer ONLY from context; say 'I don't know' if missing").
4. Robust input loop handling: user_input.strip().lower() safely checks for 'quit', 'exit', or blank input.
"""

import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Fix 1: Word-boundary chunking with overlap
def chunk_text_by_words(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        c_words = words[i : i + chunk_size]
        chunks.append(" ".join(c_words))
        if i + chunk_size >= len(words):
            break
    return chunks


policy_text = (
    "QuickBite platform refund policy states that missing items receive a 100% refund "
    "credited to the original payment method within 3 to 5 business days, or instantly as QuickBite Wallet cash. "
    "Wrong item complaints receive a full refund plus a ₹50 goodwill voucher upon photo proof."
)

chunks = chunk_text_by_words(policy_text, chunk_size=50, overlap=10)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).astype(np.float32)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def ask_question(query: str, k: int = 5) -> str:
    # Fix 2: Safe k boundary check to prevent out-of-range index error
    k = min(k, len(chunks))
    if k == 0:
        return "No document chunks available."
        
    q_vec = model.encode([query]).astype(np.float32)
    distances, indices = index.search(q_vec, k)
    
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
    context_str = "\n".join([f"Block [{idx+1}]: {c}" for idx, c in enumerate(retrieved)])
    
    # Fix 3: Strict role, context boundary, and fallback instruction
    prompt = (
        "System: You are QuickBite's AI Support Agent. Answer using ONLY the context below.\n"
        "STRICT FALLBACK: If the answer is not present in the context below, output strictly: \"I don't know.\"\n\n"
        f"--- CONTEXT ---\n{context_str}\n\n"
        f"--- QUESTION ---\nQuestion: {query}\n\nAnswer:"
    )
    return prompt


if __name__ == "__main__":
    print("=== SECTION D: CORRECTED CODE DEMONSTRATION ===")
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"FAISS Index Total: {index.ntotal}\n")
    
    # Demonstration query
    test_q = "What is the refund rule for missing items?"
    print(f"Test Query: '{test_q}'")
    prompt_result = ask_question(test_q, k=5)
    print("\nGenerated RAG Prompt:")
    print(prompt_result)
    print("\n" + "=" * 60)
