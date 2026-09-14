"""
Helper script to generate sample PDF files for testing PyPDF2 extraction.
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_resume_pdf(filename: str):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "RESUME - DHRUV MUNJPARA")
    c.drawString(100, 730, "Email: dhruv@example.com | Phone: +91 8849162891")
    c.drawString(100, 710, "Role: Data Scientist & AI Engineer")
    
    c.drawString(100, 680, "SUMMARY:")
    c.drawString(100, 665, "Passionate Data Science student specializing in Retrieval-Augmented Generation (RAG),")
    c.drawString(100, 650, "Vector Databases (FAISS, ChromaDB), SentenceTransformers, and Large Language Models.")
    
    c.drawString(100, 620, "SKILLS:")
    c.drawString(100, 605, "- Python, PyTorch, Scikit-Learn, Pandas, NumPy")
    c.drawString(100, 590, "- RAG Architecture, FAISS, ChromaDB, SentenceTransformers, PyPDF2")
    c.drawString(100, 575, "- Next.js, React, Node.js, Fast API")
    
    c.drawString(100, 545, "PROJECTS:")
    c.drawString(100, 530, "1. Enterprise PDF RAG Chatbot: Created multi-document PDF Q&A bot using FAISS.")
    c.drawString(100, 515, "2. E-Commerce Recommendation Engine: Built product search pipeline using sentence embeddings.")
    
    c.showPage()
    c.drawString(100, 750, "EDUCATION & CERTIFICATIONS:")
    c.drawString(100, 735, "- TOPS Technologies Data Science & AI Master Certification (2024-2026)")
    c.drawString(100, 720, "- Bachelor of Technology in Computer Engineering")
    c.save()
    print(f"Generated {filename}")

if __name__ == "__main__":
    create_resume_pdf("session_4_pdf_chatbot/sample_resume.pdf")
