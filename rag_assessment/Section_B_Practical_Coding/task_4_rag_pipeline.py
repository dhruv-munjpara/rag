"""
Section B — Task 4: RAG-Powered Food Delivery Policy Q&A Pipeline
Requirements:
- Multi-paragraph policy text (300+ words).
- Overlapping chunking (100 words, 20 overlap), print chunk count.
- Embeddings via sentence-transformers + FAISS IndexFlatL2 index.
- retrieve(query, k=3) returning top-k chunk strings.
- build_rag_prompt(query, retrieved_chunks) with explicit fallback ('I don't know').
- End-to-end demonstration for 'What is the refund policy for missing items?'.
"""

import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Multi-paragraph Food Delivery Policy Text (300+ words)
POLICY_DOCUMENT = """
QuickBite Platform Operating Policy Handbook — Section 4: Refunds, Cancellations & Service SLAs

1. Order Cancellation Policy:
Customers may cancel any placed order free of charge within 60 seconds of order submission before the merchant restaurant formally accepts the order. If a cancellation request is submitted after restaurant confirmation but prior to driver dispatch, a cancellation fee of 50% of the total order value will apply. Once the delivery agent has been dispatched or the food has been prepared, no cancellation or monetary refund is permitted under any circumstances.

2. Refund Rules & Eligibility:
Refunds are strictly governed by order verification status. If an order arrives with confirmed missing items, the customer is entitled to a full refund for the individual cost of the missing items credited to their original payment method within 3 to 5 business days, or instantly as QuickBite Wallet credit. For wrong item deliveries where an incorrect dish was delivered, a 100% refund of the affected dish plus a ₹50 goodwill voucher will be granted upon photo verification submitted through the app within 2 hours of delivery.

3. Delivery Windows & SLA Guarantees:
QuickBite guarantees delivery within the estimated arrival time window shown at checkout. If a delivery is delayed by more than 45 minutes past the maximum estimated ETA due to driver or platform issues (excluding extreme weather or city-wide traffic disruptions), the customer automatically receives a ₹100 delay compensation credit. 

4. Damaged Food & Quality Disputes:
In cases where food container seals are broken or contents are completely spilled during transport, customers must upload at least two clear photographs of the damaged packaging within 90 minutes of order arrival. Upon verification by our automated support team, a full order refund or an immediate free redelivery will be arranged.
"""


def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    Splits text into chunks of `chunk_size` words with `overlap` words.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


# Process Chunks
chunks = chunk_text(POLICY_DOCUMENT, chunk_size=100, overlap=20)

# Initialize SentenceTransformer & FAISS
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks).astype(np.float32)

dimension = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(chunk_embeddings)


def retrieve(query: str, k: int = 3) -> list[str]:
    """
    Retrieves the top-k most relevant policy chunk strings for a given query.
    """
    k = min(k, len(chunks))
    query_vector = model.encode([query]).astype(np.float32)
    distances, indices = faiss_index.search(query_vector, k)
    
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks


def build_rag_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """
    Assembles structured RAG prompt with role instruction, context blocks, query, and fallback.
    """
    prompt = (
        "System: You are QuickBite's AI Support Assistant. Answer the customer question using ONLY the provided context blocks below.\n"
        "STRICT RULE: Do NOT use any external knowledge. If the exact answer is not present in the context below, respond strictly with: \"I don't know.\"\n\n"
        "=== RETRIEVED CONTEXT BLOCKS ===\n"
    )
    
    for idx, chunk in enumerate(retrieved_chunks, 1):
        prompt += f"Context Block [{idx}]:\n{chunk}\n\n"
        
    prompt += (
        "=== USER QUESTION ===\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt


def run_rag_pipeline(query: str):
    """
    Executes the full end-to-end RAG pipeline and prints all intermediate artifacts.
    """
    print("=" * 60)
    print(f"RAG PIPELINE EXECUTION FOR QUERY: '{query}'")
    print("=" * 60)
    
    # Step 1: Retrieval
    retrieved = retrieve(query, k=3)
    print(f"\n--- STEP 1: RETRIEVED CHUNKS ({len(retrieved)} retrieved) ---")
    for i, c in enumerate(retrieved, 1):
        print(f"[Chunk {i}]: {c[:120]}...\n")
        
    # Step 2: Prompt Assembly
    assembled_prompt = build_rag_prompt(query, retrieved)
    print("--- STEP 2: ASSEMBLED RAG PROMPT ---")
    print(assembled_prompt)
    print("\n" + "-" * 60)
    
    # Step 3: Simulated LLM Generation Output
    simulated_answer = (
        "Based on QuickBite Policy Handbook Section 4.2:\n"
        "If an order arrives with confirmed missing items, you are entitled to a full refund for the individual cost "
        "of the missing items. Refunds are credited to your original payment method within 3 to 5 business days, "
        "or credited instantly to your QuickBite Wallet."
    )
    print("\n--- STEP 3: SIMULATED LLM GENERATED ANSWER ---")
    print(simulated_answer)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("=== TASK 4 DEMONSTRATION ===")
    print(f"Total Words in Policy Document: {len(POLICY_DOCUMENT.split())}")
    print(f"Total Chunks Created (100 words, 20 overlap): {len(chunks)}")
    print(f"FAISS Index Size: {faiss_index.ntotal}\n")
    
    # Hardcoded test query
    test_query = "What is the refund policy for missing items?"
    run_rag_pipeline(test_query)
