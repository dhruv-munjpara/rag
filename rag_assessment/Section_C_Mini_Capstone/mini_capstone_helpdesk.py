"""
Section C — Mini Capstone Project: Interactive Helpdesk Support Console
Features:
- Menu-driven loop (Option 1: RAG Q&A, Option 2: Few-Shot Complaint Classifier, Option 3: Exit).
- RAG Pipeline: Loads 500+ word food_delivery_policy.txt, chunks text, builds FAISS index, retrieves top-3 chunks, prints RAG prompt + context.
- Few-Shot Classifier: 4+ categories, builds few-shot prompt, prints predicted category & nearest example match.
- Input Validation: Rejects empty/whitespace inputs with friendly error messages.
- Persistent Session Logging: Appends timestamped logs to session_log.txt and displays summary stats on exit.
"""

import os
import sys
import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Reconfigure stdout for Windows console UTF-8 support
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global Constants & Paths
POLICY_FILE_PATH = os.path.join(os.path.dirname(__file__), "food_delivery_policy.txt")
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "session_log.txt")

# Initialize SentenceTransformer
print("⚡ Initializing AI Embeddings & Vector Index for Helpdesk...")
model = SentenceTransformer('all-MiniLM-L6-v2')


# ==========================================
# 1. Session Logging Utility
# ==========================================
class SessionLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.rag_queries_count = 0
        self.classifier_queries_count = 0

    def log(self, mode: str, user_input: str, output_summary: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{timestamp}] | MODE: {mode:<20} | INPUT: \"{user_input}\" | OUTPUT: \"{output_summary}\"\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
            
        if mode == "RAG Policy Q&A":
            self.rag_queries_count += 1
        elif mode == "Few-Shot Classifier":
            self.classifier_queries_count += 1

    def print_summary(self):
        total = self.rag_queries_count + self.classifier_queries_count
        print("\n" + "=" * 60)
        print("📊 HELPDESK SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Queries Processed : {total}")
        print(f" - RAG Policy Q&A       : {self.rag_queries_count}")
        print(f" - Few-Shot Classifications: {self.classifier_queries_count}")
        print(f"Session Log Saved To    : {os.path.abspath(self.log_path)}")
        print("=" * 60 + "\n")


logger = SessionLogger(LOG_FILE_PATH)


# ==========================================
# 2. RAG Policy Knowledge Engine
# ==========================================
class RAGPolicyEngine:
    def __init__(self, policy_path: str):
        self.chunks = []
        self.faiss_index = None
        self.load_and_index_policy(policy_path)

    def load_and_index_policy(self, policy_path: str):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found at: {policy_path}")
            
        with open(policy_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        # Overlapping Chunking (100 words, 20 word overlap)
        words = full_text.split()
        chunk_size, overlap = 100, 20
        step = chunk_size - overlap
        
        for i in range(0, len(words), step):
            c_words = words[i : i + chunk_size]
            self.chunks.append(" ".join(c_words))
            if i + chunk_size >= len(words):
                break
                
        # Generate Embeddings & Index in FAISS
        embeddings = model.encode(self.chunks).astype(np.float32)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings)
        print(f"✅ RAG Engine Loaded: {len(self.chunks)} chunks indexed into FAISS.")

    def query(self, user_question: str, k: int = 3) -> tuple[str, list[str]]:
        # Retrieve top-k
        q_vec = model.encode([user_question]).astype(np.float32)
        distances, indices = self.faiss_index.search(q_vec, k)
        
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        
        # Build Structured Prompt
        prompt = (
            "System Prompt: You are QuickBite's Helpdesk Support Bot. Answer using ONLY the context blocks below.\n"
            "STRICT FALLBACK: If the answer is not contained in the context below, output strictly: \"I don't know.\"\n\n"
            "--- RETRIEVED POLICY CONTEXT ---\n"
        )
        for i, chunk in enumerate(retrieved_chunks, 1):
            prompt += f"Block [{i}]: {chunk}\n\n"
            
        prompt += f"--- QUESTION ---\nQuestion: {user_question}\n\nAnswer:"
        
        return prompt, retrieved_chunks


# Initialize RAG Engine
rag_engine = RAGPolicyEngine(POLICY_FILE_PATH)


# ==========================================
# 3. Few-Shot Complaint Classifier Engine
# ==========================================
FEW_SHOT_BANK = [
    {"text": "I ordered pizza 1 hour ago and the driver hasn't left the store.", "label": "Late Delivery"},
    {"text": "I ordered a Paneer Tikka roll but got a Chicken Shawarma roll.", "label": "Wrong Item"},
    {"text": "The delivery bag was missing the Pepsi bottle and extra cheese dip.", "label": "Missing Item"},
    {"text": "The curry container leaked all over the bag and the bread was burnt.", "label": "Poor Quality"}
]


def classify_complaint_few_shot(complaint_text: str) -> tuple[str, str, float]:
    """
    Constructs few-shot prompt and finds closest matching exemplar via cosine similarity.
    """
    example_texts = [ex["text"] for ex in FEW_SHOT_BANK]
    ex_embs = model.encode(example_texts)
    q_emb = model.encode([complaint_text])
    
    # Cosine Similarity with exemplar bank
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_emb, ex_embs)[0]
    best_idx = int(np.argmax(sims))
    predicted_label = FEW_SHOT_BANK[best_idx]["label"]
    confidence_score = float(sims[best_idx])
    
    # Assemble Few-Shot Prompt for LLM
    prompt = (
        "System: Classify the complaint into 'Late Delivery', 'Wrong Item', 'Missing Item', or 'Poor Quality'.\n\n"
    )
    for ex in FEW_SHOT_BANK:
        prompt += f"Input: \"{ex['text']}\"\nOutput: {ex['label']}\n\n"
        
    prompt += f"Input: \"{complaint_text}\"\nOutput: {predicted_label}"
    
    return predicted_label, prompt, confidence_score


# ==========================================
# 4. Interactive Console CLI Loop
# ==========================================
def display_menu():
    print("\n" + "=" * 60)
    print("🍔 QUICKBITE AI HELPDESK SUPPORT CONSOLE")
    print("=" * 60)
    print("1. Ask a Policy Question (RAG Pipeline)")
    print("2. Classify a Customer Complaint (Few-Shot Prompting)")
    print("3. Exit Console")
    print("-" * 60)


def run_helpdesk():
    print("\n🎉 Welcome to QuickBite Automated Helpdesk System!")
    
    while True:
        display_menu()
        choice = input("Enter your selection (1-3): ").strip()
        
        if choice == "1":
            question = input("\n👉 Enter customer policy question: ").strip()
            if not question:
                print("⚠️ ERROR: Question cannot be empty or blank. Please try again.")
                continue
                
            prompt, retrieved = rag_engine.query(question, k=3)
            
            print("\n" + "=" * 60)
            print("🔍 RAG RETRIEVAL & PROMPT RESULT")
            print("=" * 60)
            print(f"Top Retrieved Context Preview: {retrieved[0][:120]}...")
            print("\n--- ASSEMBLED RAG PROMPT ---")
            print(prompt)
            print("=" * 60)
            
            summary = f"Retrieved {len(retrieved)} chunks. Prompt assembled."
            logger.log("RAG Policy Q&A", question, summary)
            
        elif choice == "2":
            complaint = input("\n👉 Enter customer complaint text: ").strip()
            if not complaint:
                print("⚠️ ERROR: Complaint text cannot be empty or blank. Please try again.")
                continue
                
            label, prompt, confidence = classify_complaint_few_shot(complaint)
            
            print("\n" + "=" * 60)
            print("🏷️ FEW-SHOT CLASSIFICATION RESULT")
            print("=" * 60)
            print(f"Predicted Category : {label}")
            print(f"Semantic Match Score: {confidence:.4f}")
            print("\n--- FEW-SHOT CLASSIFIER PROMPT ---")
            print(prompt)
            print("=" * 60)
            
            summary = f"Predicted Category: {label} (Score: {confidence:.4f})"
            logger.log("Few-Shot Classifier", complaint, summary)
            
        elif choice == "3":
            print("\nExiting Helpdesk Console... Goodbye!")
            logger.print_summary()
            break
            
        else:
            print("⚠️ INVALID SELECTION: Please enter 1, 2, or 3.")


if __name__ == "__main__":
    # If run non-interactively or directly in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n--- RUNNING NON-INTERACTIVE CAPSTONE VERIFICATION TEST ---")
        q_test = "What is the refund rule for missing items?"
        p, r = rag_engine.query(q_test, k=3)
        logger.log("RAG Policy Q&A", q_test, f"Retrieved {len(r)} chunks")
        
        c_test = "My pizza arrived 2 hours late and was cold."
        lbl, pr, sc = classify_complaint_few_shot(c_test)
        logger.log("Few-Shot Classifier", c_test, f"Predicted: {lbl}")
        
        logger.print_summary()
    else:
        run_helpdesk()
