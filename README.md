# Retrieval-Augmented Generation (RAG) — Complete Course Assignment

**Student Name**: Dhruv Munjpara  
**Enrollment / Credentials**: `8849162891`  
**Course**: Data Science & AI Master Program — TOPS Technologies  
**Portal**: [https://careercenter.tops-int.com/dashboard](https://careercenter.tops-int.com/dashboard)

---

## 📁 Repository Structure

```
d:\DA&DS\rag/
├── README.md                           # Master README and submission documentation
├── RAG_Full_Assignment_Notebook.ipynb  # Complete all-in-one Jupyter Notebook with outputs
├── session_1_intro/
│   ├── session_1_answers.md            # Conceptual answers, workflow diagram, comparison table
│   └── rag_workflow_diagram.png        # RAG workflow architectural diagram
├── session_2_chunking_embeddings/
│   ├── session_2_code.py               # Chunking function, embeddings & similarity script
│   └── privacy_policy_sample.txt       # WhatsApp privacy policy sample text
├── session_3_vectordb_retrieval/
│   └── session_3_code.py               # FAISS 4D/Movie search & ChromaDB Zomato/Insta search
└── session_4_pdf_chatbot/
    ├── session_4_code.py               # PyPDF2 text extraction, chunking, QA engine script
    ├── create_sample_pdfs.py           # ReportLab helper to generate sample resume PDF
    └── sample_resume.pdf               # Test PDF document
```

---

## 🚀 How to Run the Solutions

### 1. Environment Setup
Install required dependencies:
```bash
pip install sentence-transformers faiss-cpu chromadb PyPDF2 scikit-learn reportlab notebook pandas numpy
```

### 2. Run Session Scripts
- **Session 2 (Chunking & Embeddings)**:
  ```bash
  python session_2_chunking_embeddings/session_2_code.py
  ```
- **Session 3 (Vector DB & Retrieval)**:
  ```bash
  python session_3_vectordb_retrieval/session_3_code.py
  ```
- **Session 4 (PDF Chatbot)**:
  ```bash
  python session_4_pdf_chatbot/session_4_code.py
  ```

### 3. Open All-In-One Jupyter Notebook
```bash
jupyter notebook RAG_Full_Assignment_Notebook.ipynb
```

---

## 📌 Session-by-Session Submission Answers & Summaries

### Session 1: Introduction to RAG & Why It's Needed
1. **RAG Definition**: RAG combines vector retrieval with LLM generation to inject up-to-date, accurate private data into LLM prompts.
2. **Everyday App Example**: Zomato live menu, GPS open status, and active bank discounts passed into LLM prompt for real-time recommendations.
3. **Outdated LLM Drawback**: A 2022 LLM cannot answer IPL 2024 results or post-2022 Vande Bharat train schedules.
4. **Comparison**:
   * *RAG*: Low cost, real-time data freshness, fast implementation.
   * *Fine-tuning*: High GPU cost, static knowledge snapshot, slow implementation.
   * *Prompt Engineering*: Very low cost, temporary context limit, immediate implementation.
5. **Real-world Case Study**: Notion AI Q&A using vector search across employee docs.

### Session 2: Chunking & Embeddings
1. **`chunk_text(text, chunk_size=100, overlap=20)`** created.
2. **Privacy Policy**: Split into 100-word chunks with 20-word overlap.
3. **Embeddings**: `all-MiniLM-L6-v2` produces 384-dimensional dense vectors.
4. **Cosine Similarity**: Relevancy score computed for query *"How is my personal data shared with third parties?"*.
5. **Overlap Importance**: Prevents splitting critical sentences across boundaries.

### Session 3: Vector DB & Retrieval
1. **FAISS 4D Vector Index**: `IndexFlatL2` query returns closest distance & vector ID.
2. **ChromaDB Zomato Search**: Query `"spicy food"` retrieves *Spicy Punjab Express*.
3. **FAISS Movie Search**: Query `"a scary space movie"` retrieves *Alien: Resurrection* and *Gravity*.
4. **ChromaDB Instagram Captions**: Query `"healthy lifestyle"` retrieves workout & wellness captions.
5. **Flipkart Architecture**: FAISS filters products < ₹2,000; top 3 context passed to ChatGPT.

### Session 4: Build PDF Chatbot
1. **PyPDF2**: Extracted text from `sample_resume.pdf`.
2. **500-Char Chunking**: Structured document into character chunks.
3. **`paraphrase-MiniLM-L6-v2`**: Converted PDF chunks into dense embeddings.
4. **`answer_question(query)`**: Returns top matched chunk with similarity score.
5. **Applications**: Legal contracts (NDAs), Medical reports, and Educational textbooks.
