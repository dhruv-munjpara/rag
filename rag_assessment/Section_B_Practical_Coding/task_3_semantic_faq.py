"""
Section B — Task 3: Semantic Search Over Restaurant FAQs
Requirements:
- Define at least 6 FAQ strings covering food delivery topics.
- Generate embeddings using SentenceTransformer('all-MiniLM-L6-v2').
- Build FAISS IndexFlatL2 index and confirm index size matches FAQ count.
- Implement search_faq(query, k=2) returning top-k FAQs + L2 distances.
- Test with non-keyword matching queries (e.g. 'How do I get money back?').
"""

import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Reconfigure stdout for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Dataset of 6 Restaurant FAQs
FAQ_DATASET = [
    "FAQ 1: Standard delivery time is 30 to 45 minutes depending on traffic conditions and restaurant distance.",
    "FAQ 2: Orders can be cancelled free of charge within 60 seconds of placing the order before restaurant confirmation.",
    "FAQ 3: Full refunds are automatically issued to your original payment method for missing items or cancelled orders within 3 to 5 business days.",
    "FAQ 4: If an ordered item is out of stock, restaurants will attempt to call you for an approved item substitution.",
    "FAQ 5: You can contact our 24/7 customer support team directly through the 'Help' section in your mobile app or via live chat.",
    "FAQ 6: Minimum order value for free delivery is ₹200 for regular members and ₹0 for Gold subscribers."
]

# Initialize SentenceTransformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate Embeddings & Convert to float32 NumPy Array
embeddings = model.encode(FAQ_DATASET).astype(np.float32)

# Build FAISS Index (IndexFlatL2)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)


def search_faq(query: str, k: int = 2) -> list[tuple[str, float]]:
    """
    Encodes query, searches FAISS index, and returns top-k FAQ strings with L2 distances.
    """
    query_vector = model.encode([query]).astype(np.float32)
    distances, indices = faiss_index.search(query_vector, k)
    
    results = []
    for rank in range(k):
        idx = indices[0][rank]
        dist = float(distances[0][rank])
        faq_text = FAQ_DATASET[idx]
        results.append((faq_text, dist))
        
    return results


if __name__ == "__main__":
    print("=== TASK 3 DEMONSTRATION ===")
    print(f"Total FAQs Indexed in FAISS: {faiss_index.ntotal}")
    print(f"Embedding Dimensions: {dimension}")
    print(f"Index Size Verification: {'PASS' if faiss_index.ntotal == len(FAQ_DATASET) else 'FAIL'}\n")
    print("=" * 60 + "\n")
    
    # Test Query 1: Uses non-keyword wording ("get money back") for FAQ 3 ("refund policy")
    query_1 = "How do I get money back if my food was not delivered?"
    print(f"🔍 Test Query 1: '{query_1}'")
    results_1 = search_faq(query_1, k=2)
    for rank, (faq, dist) in enumerate(results_1, 1):
        print(f"  Rank {rank} (L2 Distance: {dist:.4f}):\n   {faq}")
    print("\n" + "-" * 60 + "\n")
    
    # Test Query 2: Uses non-keyword wording ("talk to a human") for FAQ 5 ("contact support")
    query_2 = "I need to talk to a human agent about a problem."
    print(f"🔍 Test Query 2: '{query_2}'")
    results_2 = search_faq(query_2, k=2)
    for rank, (faq, dist) in enumerate(results_2, 1):
        print(f"  Rank {rank} (L2 Distance: {dist:.4f}):\n   {faq}")
    print("\n" + "=" * 60 + "\n")
