"""
Session 3: Vector DB & Retrieval
Task 1: FAISS Index with 4D vectors
Task 2: ChromaDB / Vector Search for 5 Restaurant Descriptions (Zomato)
Task 3: FAISS Index with Movie Descriptions & Sentence Transformers ("a scary space movie")
Task 4: ChromaDB for 5 Instagram Captions ("healthy lifestyle")
Task 5: Flipkart Product Search Workflow Pseudocode & Prompt Construction
"""

import sys
import numpy as np
import faiss
import chromadb
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

model = SentenceTransformer('all-MiniLM-L6-v2')


# Custom Embedding Function matching ChromaDB EmbeddingFunction interface
class STEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        super().__init__()
        
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = model.encode(input)
        return embeddings.tolist()

emb_fn = STEmbeddingFunction()


# ==========================================
# Task 1: Basic FAISS Index (4-Dimensional)
# ==========================================
def run_task_1():
    print("=" * 60)
    print("TASK 1: FAISS Index with 4-Dimensional Vectors")
    print("=" * 60)
    
    np.random.seed(42)
    vectors = np.array([
        [0.1, 0.8, 0.3, 0.5],
        [0.9, 0.2, 0.1, 0.4],
        [0.2, 0.1, 0.9, 0.7],
        [0.5, 0.5, 0.4, 0.6]
    ], dtype=np.float32)
    
    d = 4
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    print(f"Number of vectors in FAISS index: {index.ntotal}")
    
    query_vector = np.array([[0.15, 0.75, 0.35, 0.48]], dtype=np.float32)
    k = 1
    
    distances, indices = index.search(query_vector, k)
    
    print(f"Query Vector: {query_vector.tolist()}")
    print(f"Closest Vector Index/ID: {indices[0][0]}")
    print(f"L2 Distance: {distances[0][0]:.4f}")
    print(f"Closest Vector Content: {vectors[indices[0][0]].tolist()}\n")


# ==========================================
# Task 2: ChromaDB Restaurant Search (Zomato)
# ==========================================
def run_task_2():
    print("=" * 60)
    print("TASK 2: ChromaDB Restaurant Search (Zomato Listings)")
    print("=" * 60)
    
    client = chromadb.Client()
    collection = client.create_collection(name="zomato_restaurants", embedding_function=emb_fn)
    
    restaurants = [
        "Spicy Punjab Express: Famous for extremely hot and spicy chicken tikka, fiery curry, and spicy naan.",
        "Green Leaf Cafe: Fresh organic salads, vegan smoothie bowls, and cold-pressed juices.",
        "Ocean Catch: Premium seafood restaurant offering butter garlic prawns and grilled salmon.",
        "Sweet Treats Bakery: Delicious chocolate cakes, colorful cupcakes, and warm pastries.",
        "Szechuan Dragon: Authentic spicy Indo-Chinese noodles, hot and sour soup, and chilli paneer."
    ]
    
    ids = [f"rest_{i}" for i in range(1, 6)]
    
    collection.add(documents=restaurants, ids=ids)
    
    query = "spicy food"
    results = collection.query(query_texts=[query], n_results=1)
    
    print(f"Query: '{query}'")
    print(f"Top Matching Restaurant ID: {results['ids'][0][0]}")
    print(f"Top Matching Restaurant Description:\n{results['documents'][0][0]}\n")


# ==========================================
# Task 3: FAISS Top-2 Movie Search with SentenceTransformers
# ==========================================
def run_task_3():
    print("=" * 60)
    print("TASK 3: FAISS Top-2 Movie Search ('a scary space movie')")
    print("=" * 60)
    
    movies = [
        "Alien: Resurrection - A terrifying sci-fi horror movie about monstrous extraterrestrial creatures hunting crew members on a space station.",
        "Interstellar - A futuristic space exploration movie where astronauts travel through a wormhole near Saturn to save humanity.",
        "The Hangover - A comedy film about three friends who lose their groom-to-be during a wild bachelor party in Las Vegas.",
        "The Conjuring - A creepy supernatural horror film about paranormal investigators exploring a haunted farmhouse.",
        "Gravity - A suspenseful space thriller about an astronaut stranded in space after satellite debris destroys her shuttle."
    ]
    
    embeddings = model.encode(movies).astype(np.float32)
    
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    query = "a scary space movie"
    query_embedding = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_embedding)
    
    k = 2
    distances, indices = index.search(query_embedding, k)
    
    print(f"Query: '{query}'")
    print("Top 2 Retrieved Movies:")
    for rank in range(k):
        idx = indices[0][rank]
        score = distances[0][rank]
        print(f" Rank {rank + 1} (Similarity Score: {score:.4f}):")
        print(f"  {movies[idx]}")
    print()


# ==========================================
# Task 4: ChromaDB Instagram Captions ("healthy lifestyle")
# ==========================================
def run_task_4():
    print("=" * 60)
    print("TASK 4: ChromaDB Instagram Captions Search")
    print("=" * 60)
    
    client = chromadb.Client()
    collection = client.create_collection(name="instagram_captions", embedding_function=emb_fn)
    
    captions = [
        "Crushing my morning workout! Fueling my body with green smoothie bowls and daily cardio routines. #fitness #healthylifestyle",
        "Wanderlust vibes! Exploring the serene beaches of Bali during sunset. #travelgram #vacation",
        "Cheat meal day! Double cheese smash burger with extra crispy fries and chocolate shake.",
        "Mindfulness and yoga session under the sun. Balance, clean eating, and positive energy! #wellness #healthyliving",
        "OOTD: Wearing vintage denim jacket with white sneakers for Sunday brunch! #fashion #style"
    ]
    
    ids = [f"cap_{i}" for i in range(1, 6)]
    collection.add(documents=captions, ids=ids)
    
    query = "healthy lifestyle"
    results = collection.query(query_texts=[query], n_results=1)
    
    print(f"Query: '{query}'")
    print(f"Retrieved Top Caption:\n{results['documents'][0][0]}\n")


# ==========================================
# Task 5: Flipkart Product Search RAG Architecture & Flow
# ==========================================
def print_task_5_architecture():
    arch = """
============================================================
TASK 5: Flipkart RAG Search Architecture ("cheap noise-canceling headphones")
============================================================

1. Offline Ingestion & Embedding Pipeline:
   - Flipkart product catalog (Title, Price, Features, ANC Specs, User Reviews) is converted into vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
   - The product vector embeddings + metadata (price=1499 INR, brand=boAt, rating=4.5) are indexed into a FAISS Vector Database.

2. Real-Time Query Workflow:
   Step A: User Query Input
           User types: "cheap noise-canceling headphones"
   
   Step B: Vectorization
           Sentence Transformer encodes "cheap noise-canceling headphones" into a 384D query vector.
   
   Step C: FAISS Retrieval & Metadata Filtering
           FAISS executes Cosine Similarity search to find the top K (e.g., 5) products matching "noise-canceling headphones".
           Metadata filter applies: price < ₹2,000.
   
   Step D: System Prompt Construction & LLM Generation
           The retrieved top 3 products are formatted into context for ChatGPT (GPT-4o / Llama-3):

           ------------------- PROMPT TO LLM -------------------
           System: You are Flipkart's AI Shopping Assistant. Use ONLY the product context below to answer the user request.
           
           Context:
           Product 1: boAt Nirvana Ion ANC | Price: ₹1,499 | ANC: 32dB Noise Cancellation | Rating: 4.4/5
           Product 2: Boult Audio Anchor | Price: ₹1,799 | ANC: Active Noise Cancellation | Rating: 4.3/5
           Product 3: Noise Two Wireless | Price: ₹1,299 | Active Noise Cancellation | Rating: 4.1/5

           User Query: Recommend cheap noise-canceling headphones.
           ------------------------------------------------------

   Step E: LLM Response
           "Here are the top budget-friendly noise-canceling headphones on Flipkart under ₹2,000:
            1. boAt Nirvana Ion ANC (₹1,499) - 32dB active noise cancellation with 4.4 rating.
            2. Noise Two Wireless (₹1,299) - Most affordable ANC option."
"""
    print(arch)


if __name__ == "__main__":
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
    print_task_5_architecture()
