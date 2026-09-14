"""
Session 2: Chunking & Embeddings
Task 1: Python function chunk_text(text, chunk_size, overlap)
Task 2: Chunk privacy policy (chunk_size=100, overlap=20)
Task 3: Embeddings using sentence-transformers (all-MiniLM-L6-v2)
Task 4: Cosine similarity calculation for query "How is my personal data shared with third parties?"
Task 5: Explanation of Chunk Overlap Importance
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==========================================
# Task 1: Chunking Function
# ==========================================
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits text into chunks of `chunk_size` words with `overlap` words shared between consecutive chunks.
    
    Args:
        text (str): Input document text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.
        
    Returns:
        list[str]: List of text chunks.
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap.")
        
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        if i + chunk_size >= len(words):
            break
            
    return chunks


# ==========================================
# Task 2: Chunking Privacy Policy Text
# ==========================================
def run_task_2(file_path: str):
    print("=" * 60)
    print("TASK 2: Chunk Privacy Policy Text (chunk_size=100, overlap=20)")
    print("=" * 60)
    
    with open(file_path, "r", encoding="utf-8") as f:
        privacy_policy = f.read()
        
    chunks = chunk_text(privacy_policy, chunk_size=100, overlap=20)
    print(f"Total Chunks Created: {len(chunks)}\n")
    
    for idx, chunk in enumerate(chunks[:2], 1):
        word_count = len(chunk.split())
        print(f"--- Chunk {idx} (Word count: {word_count}) ---")
        print(chunk)
        print("-" * 50)
        
    return chunks


# ==========================================
# Task 3: SentenceTransformers Embeddings
# ==========================================
def run_task_3(chunks: list[str]):
    print("\n" + "=" * 60)
    print("TASK 3: Generate Embeddings using all-MiniLM-L6-v2")
    print("=" * 60)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    first_3_chunks = chunks[:3]
    
    embeddings = model.encode(first_3_chunks)
    
    print(f"Number of chunks embedded: {len(embeddings)}")
    print(f"Embedding Vector Shape (Dimensions): {embeddings[0].shape}")
    print(f"First 5 numbers of Chunk 1 Embedding Vector:\n{embeddings[0][:5]}\n")
    
    return model, embeddings, first_3_chunks


# ==========================================
# Task 4: Cosine Similarity Search
# ==========================================
def run_task_4(model: SentenceTransformer, chunk_embeddings: np.ndarray, chunks: list[str]):
    print("=" * 60)
    print("TASK 4: Cosine Similarity for Data Sharing Query")
    print("=" * 60)
    
    query = "How is my personal data shared with third parties?"
    print(f"Query: '{query}'\n")
    
    query_embedding = model.encode([query])
    
    # Calculate Cosine Similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    best_idx = int(np.argmax(similarities))
    
    for idx, (score, chunk) in enumerate(zip(similarities, chunks), 1):
        print(f"Chunk {idx} Cosine Similarity Score: {score:.4f}")
        print(f"Preview: {chunk[:100]}...\n")
        
    print(f"=> MOST RELEVANT CHUNK: Chunk {best_idx + 1} with score {similarities[best_idx]:.4f}")
    print(f"Content of Chunk {best_idx + 1}:\n{chunks[best_idx]}\n")


# ==========================================
# Task 5: Importance of Chunk Overlap
# ==========================================
def print_task_5_explanation():
    explanation = """
============================================================
TASK 5: Why is Chunk Overlap Important in RAG?
============================================================

1. Preserving Semantic Context Across Boundaries:
   When documents are cut abruptly into fixed-size chunks (e.g., exactly at 100 words), critical ideas, sentences, or cause-and-effect clauses often span across the boundary line. Without overlap, the subject might be in Chunk 1 and the predicate/action in Chunk 2.

2. Concrete Example of Missing Overlap Failure:
   - Original Text: "...We never sell user emails to marketers. However, personal data may be shared with Meta partners for fraud prevention..."
   - Without Overlap (Cut between 'However' and 'personal'):
     * Chunk 1: "...We never sell user emails to marketers. However,"
     * Chunk 2: "personal data may be shared with Meta partners for fraud prevention..."
   - User Query: "Do you sell my email to marketers?"
   - Failure Mode: Chunk 2 contains 'personal data shared with Meta' but lacks the negative context ('We never sell...'), while Chunk 1 contains the answer but misses the subject ('personal data'). The vector similarity match loses context, leading the LLM to hallucinate or misinterpret the policy!

3. Benefit of Overlap:
   By overlapping 20 words, the phrase 'We never sell user emails to marketers' appears intact inside Chunk 2 as well, guaranteeing complete semantic representation in vector space.
"""
    print(explanation)


if __name__ == "__main__":
    policy_path = "session_2_chunking_embeddings/privacy_policy_sample.txt"
    chunks = run_task_2(policy_path)
    model, embeddings, first_3_chunks = run_task_3(chunks)
    run_task_4(model, embeddings, first_3_chunks)
    print_task_5_explanation()
