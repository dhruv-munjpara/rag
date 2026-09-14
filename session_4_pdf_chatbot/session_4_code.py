"""
Session 4: Build PDF Chatbot
Task 1: Extract text from PDF using PyPDF2
Task 2: Split text into 500-character chunks
Task 3: Convert chunks to embeddings using paraphrase-MiniLM-L6-v2
Task 4: Implement answer_question(user_query) using cosine similarity
Task 5: Real-world PDF Chatbot Applications (Legal, Medical, Textbooks)
"""

import os
from PyPDF2 import PdfReader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==========================================
# Task 1: Extract Text from PDF using PyPDF2
# ==========================================
def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    """
    Extracts all text from a PDF file using PyPDF2.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        tuple[str, int]: (full_text, total_pages)
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    return full_text, total_pages


def run_task_1(pdf_path: str):
    print("=" * 60)
    print("TASK 1: PyPDF2 Text Extraction")
    print("=" * 60)
    
    full_text, total_pages = extract_text_from_pdf(pdf_path)
    
    print(f"PDF File: {pdf_path}")
    print(f"Total Pages Extracted: {total_pages}")
    print(f"Total Characters Extracted: {len(full_text)}")
    print("\n--- First 200 Characters Preview ---")
    print(full_text[:200])
    print("-" * 50 + "\n")
    
    return full_text


# ==========================================
# Task 2: Chunk Text into 500 Characters
# ==========================================
def chunk_by_characters(text: str, chunk_size: int = 500) -> list[str]:
    """
    Splits text into chunks of specified character length.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def run_task_2(full_text: str):
    print("=" * 60)
    print("TASK 2: Chunk Text into 500-Character Chunks")
    print("=" * 60)
    
    chunks = chunk_by_characters(full_text, chunk_size=500)
    print(f"Total Chunks Created (500 chars each): {len(chunks)}\n")
    
    for idx, c in enumerate(chunks[:2], 1):
        print(f"--- Chunk {idx} (Length: {len(c)} chars) ---")
        print(c)
        print("-" * 50)
    print()
    return chunks


# ==========================================
# Task 3: Generate Embeddings using paraphrase-MiniLM-L6-v2
# ==========================================
def run_task_3(chunks: list[str]):
    print("=" * 60)
    print("TASK 3: Convert Chunks to Embeddings (paraphrase-MiniLM-L6-v2)")
    print("=" * 60)
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    chunk_embeddings = model.encode(chunks)
    
    print(f"Model: paraphrase-MiniLM-L6-v2")
    print(f"Total Embedded Chunks: {len(chunk_embeddings)}")
    print(f"Embedding Vector Dimensions: {chunk_embeddings[0].shape}\n")
    
    return model, chunk_embeddings


# ==========================================
# Task 4: Q&A Function answer_question(user_query)
# ==========================================
def build_qa_engine(model: SentenceTransformer, chunk_embeddings: np.ndarray, chunks: list[str]):
    def answer_question(user_query: str):
        print("=" * 60)
        print("TASK 4: PDF Q&A Search Engine")
        print("=" * 60)
        print(f"User Query: '{user_query}'")
        
        query_emb = model.encode([user_query])
        similarities = cosine_similarity(query_emb, chunk_embeddings)[0]
        
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        
        print(f"Top Matching Chunk Index: {best_idx}")
        print(f"Cosine Similarity Score: {best_score:.4f}")
        print("\n--- Most Relevant Retrieved PDF Chunk ---")
        print(chunks[best_idx])
        print("=" * 60 + "\n")
        
        return chunks[best_idx], best_score
        
    return answer_question


# ==========================================
# Task 5: Real-World PDF Chatbot Applications
# ==========================================
def print_task_5_applications():
    apps = """
============================================================
TASK 5: Real-World PDF Chatbot Applications
============================================================

1. Legal Contracts & NDAs (LegalTech)
   - Use Case: Corporate lawyers and clients uploading 100-page lease agreements or merger contracts.
   - How RAG Helps: Users can ask "What are the termination penalties and notice period under Clause 4?" The RAG bot retrieves the exact clause paragraphs and summarizes liabilities without hallucination.

2. Medical Diagnostic Reports & Clinical Guidelines (Healthcare)
   - Use Case: Doctors and patients analyzing multi-page MRI/Blood lab test reports or FDA drug dosage guidelines.
   - How RAG Helps: A physician asks "What was the patient's HbA1c trend over the last 3 reports?" The bot retrieves exact lab values from historical PDFs and plots/explains the progression safely.

3. Educational Textbooks & Research Papers (EdTech / Academia)
   - Use Case: Students studying complex 500-page physics or engineering textbooks.
   - How RAG Helps: A student asks "Explain Quantum Entanglement according to Chapter 7 with examples." The RAG chatbot retrieves page excerpts from Chapter 7 and generates a simplified explanation with exact page citations.
"""
    print(apps)


if __name__ == "__main__":
    pdf_path = "session_4_pdf_chatbot/sample_resume.pdf"
    
    # If sample resume doesn't exist yet, we create it
    if not os.path.exists(pdf_path):
        from create_sample_pdfs import create_resume_pdf
        create_resume_pdf(pdf_path)
        
    full_text = run_task_1(pdf_path)
    chunks = run_task_2(full_text)
    model, chunk_embeddings = run_task_3(chunks)
    
    qa_bot = build_qa_engine(model, chunk_embeddings, chunks)
    
    # Test query
    qa_bot("What are Dhruv's skills in Retrieval-Augmented Generation and Vector DBs?")
    
    print_task_5_applications()
