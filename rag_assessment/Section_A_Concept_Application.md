# Section A — Concept Application Answers

---

## Scenario S1: Prompt Clarity & Consistency in Customer Support Chatbots

### Question Analysis:
In the food delivery customer support chatbot scenario, vague prompts cause inconsistent behavior (refusing valid refunds or approving unverified refunds) because the LLM lacks explicit decision rules, clear role boundaries, and strict output format specifications.

### Distinguishing a Clear Prompt from an Unclear Prompt:
* **Unclear Prompt**: A vague prompt like *"Help the user with their refund request"* leaves decision criteria ambiguous. The LLM must guess what constitutes a valid refund, leading to hallucinations, variable policy enforcement, and erratic formatting.
* **Clear Prompt**: A clear prompt defines an explicit role (e.g., *"Food Delivery Support Agent"*), provides deterministic business rules (e.g., *"Only approve refund if order status is 'Cancelled by Restaurant' or 'Missing Item' is confirmed"*), and specifies strict response limits (e.g., *"Respond in under 80 words in JSON or structured bullet format"*).

### Two Specific Changes & Justifications:

1. **Change 1 — Explicit Decision Rule Matrix & Escalation Logic**:
   * *Modification*: Add a strict conditional decision matrix inside the System Prompt:
     ```text
     DECISION RULES:
     1. IF issue_type == 'missing_item' AND proof_provided == True THEN approve refund for missing item value only.
     2. IF issue_type == 'late_delivery' AND delay_minutes < 45 THEN reject monetary refund and offer a 10% coupon.
     3. IF issue_type NOT IN ['missing_item', 'late_delivery', 'wrong_item'] THEN escalate to human agent.
     ```
   * *Justification*: Providing unambiguous IF-THEN rules eliminates the LLM's subjective interpretation of refund validity, enforcing uniform policy compliance across all customer interactions.

2. **Change 2 — Strict Output Schema & Length Constraint**:
   * *Modification*: Add format enforcement to the System Prompt:
     ```text
     OUTPUT FORMAT CONSTRAINTS:
     - Keep responses under 80 words.
     - Always output a 2-sentence structure: (1) Decision summary, (2) Next steps or coupon code.
     - Do not promise refunds without confirming order ID and issue verification.
     ```
   * *Justification*: Setting strict structural boundaries prevents prompt creep, overly verbose responses, and unauthorized promises that degrade customer trust.

---

## Scenario S2: Zero-Shot vs. Few-Shot Prompting for Complaint Classification

### Question Analysis:
Classifying incoming food delivery complaints into 4 categories (`'Late Delivery'`, `'Wrong Item'`, `'Missing Item'`, `'Poor Quality'`) using 200 available labelled examples under a limited LLM context window constraint.

### Comparison of Zero-Shot vs. Few-Shot Prompting:

| Feature | Zero-Shot Prompting | Few-Shot Prompting |
| :--- | :--- | :--- |
| **Definition** | Providing only category definitions and asking the LLM to classify without seeing any examples. | Providing 4 to 8 concrete input-output examples in the prompt before the target input. |
| **Context Window Usage** | Minimal token footprint. | Slightly higher token footprint, but fits easily within standard context windows. |
| **Accuracy on Edge Cases** | Moderate; struggles with ambiguous complaints (e.g., *"Food was cold and late"*). | High; teaches the LLM exact label boundaries and preferred priority handling. |

### Justification of Chosen Approach:
**Chosen Approach: Few-Shot Prompting**.
* *Reasoning*: Although 200 examples cannot fit in a single prompt context window, selecting **4 to 8 high-quality exemplar pairs** (1–2 per category) consumes only ~200 tokens while dramatically improving classification precision. Few-shot prompting demonstrates the exact syntax and handles ambiguous complaints (e.g., clarifying whether a cold meal should be labeled `'Late Delivery'` or `'Poor Quality'`).

### Factors Determining the Number of Examples in Few-Shot Prompts:
1. **Context Window Token Limit**: Prompt size must leave adequate space for the user query and LLM completion.
2. **Category Balance**: Every target category must have equal representation (e.g., 1 or 2 examples per label).
3. **Edge-Case Diversity**: Including examples that resolve borderline/ambiguous complaints improves generalization far more than redundant straightforward examples.
4. **Token Cost & Latency**: Fewer concise examples reduce API latency and per-request inference cost.

---

## Scenario S3: Chain-of-Thought (CoT) Prompting for Delivery Route Sequence

### Question Analysis:
Calculating the optimal stop sequence for a delivery agent given 5 order addresses, traffic delays, and priority levels. Direct prompts yield incorrect route sequences without explanation.

### How Chain-of-Thought (CoT) Improves Quality:
CoT prompting forces the LLM to decompose a complex multi-variable optimization problem into sequential reasoning steps before outputting the final sequence. By generating intermediate thoughts (evaluating priority orders first, factoring in traffic delays second, and calculating spatial distance third), the model avoids hasty heuristic errors and maintains mathematical logic.

### Restructured CoT Prompt Example:

```text
System: You are an AI Logistics Planner. Calculate the optimal delivery stop sequence step-by-step.

User:
Orders to Deliver:
1. Order A: Priority HIGH, Distance 2 km, Traffic Delay +10 mins.
2. Order B: Priority LOW, Distance 1 km, Traffic Delay 0 mins.
3. Order C: Priority HIGH, Distance 3 km, Traffic Delay 0 mins.
4. Order D: Priority MEDIUM, Distance 1.5 km, Traffic Delay +5 mins.
5. Order E: Priority LOW, Distance 4 km, Traffic Delay 0 mins.

Let's solve this step-by-step:
Step 1: Group orders by priority level (HIGH > MEDIUM > LOW).
Step 2: Calculate effective travel time for HIGH priority orders considering distance and traffic delays.
Step 3: Determine closest subsequent stops for MEDIUM and LOW priority orders to minimize total route deviation.
Step 4: Finalize optimal stop sequence.

Optimal Stop Sequence:
```

### One Limitation of Chain-of-Thought Prompting:
* **Increased Token Cost & Latency**: Generating step-by-step reasoning tokens significantly increases latency (response time) and inference API cost compared to direct zero-shot completion, which can be problematic for real-time dispatch systems needing sub-second decisions.

---

## Scenario S4: RAG vs. Fine-Tuning for 150-Page Quarterly Policy Handbook

### Question Analysis:
Choosing between fine-tuning an LLM vs. building a RAG system for querying a 150-page company policy handbook updated every quarter.

### Justification of RAG Over Fine-Tuning (2 Key Reasons):

1. **Handling Frequent Quarterly Document Updates**:
   * *RAG*: Updating the knowledge base requires only re-chunking and replacing document embeddings in a vector database (taking seconds and costing virtually nothing).
   * *Fine-Tuning*: Fine-tuning requires curating new dataset pairs, running GPU retraining pipelines, and re-deploying model weights every quarter, which is expensive and time-consuming.

2. **Verifiability, Transparency & Zero Hallucinations**:
   * *RAG*: RAG explicitly retrieves text chunks from the handbook and passes them to the LLM, allowing the system to cite exact page numbers and paragraphs for payout rules or SLA disputes.
   * *Fine-Tuning*: Fine-tuned models store facts implicitly in parametric weights, making it impossible to guarantee that answers are free of hallucinations or to provide page-level citations.

### Scenario Where Fine-Tuning Would Be Better:
Fine-tuning would be superior if the primary goal were **teaching the model a specialized domain language, corporate brand voice, or strict output formatting syntax** (e.g., converting unstructured customer logs into custom internal XML/JSON data schemas) rather than retrieving evolving factual information.

---

## Scenario S5: Chunk Size & Chunk Overlap Optimization for 500 PDF Menus

### Question Analysis:
RAG retrieval across 500 PDF restaurant menus (8 pages each) suffers because chunks cut mid-sentence, separating dish names from their prices.

### Role of Chunk Size and Chunk Overlap:
* **Chunk Size**: Determines the character/word window of each stored document segment. If too small (e.g., 50 words), semantic context is fragmented; if too large (e.g., 1000 words), irrelevant noise dilutes vector similarity relevance.
* **Chunk Overlap**: Defines the shared text window between consecutive chunks (e.g., 20–50 words). Overlap ensures that facts spanning chunk boundaries (like a dish name followed by price in the next sentence) remain intact in at least one chunk.

### Parameter Adjustment Strategy:
1. **Increase Chunk Size from Small Window to ~250 Words (or 1000 Characters)**: Menu items and pricing notes usually span full sections. A 250-word chunk size ensures complete dish descriptions, ingredient lists, and pricing blocks stay unified.
2. **Set Chunk Overlap to 50 Words (or 200 Characters)**: Adding a 50-word overlap guarantees that if a dish name ends near a chunk boundary, its associated price and add-on costs in the next line will appear together in the overlapping window.
3. **Use Structure-Aware Layout Chunking**: Instead of arbitrary character counts, split PDFs by natural structural delimiters (e.g., section headers or menu category blocks).

### Introduced Trade-off:
* **Higher Storage & Context Token Usage**: Larger chunk sizes with overlap increase the total number of vectors in the database and pass larger context blocks to the LLM prompt, slightly increasing LLM API token costs per query.

---

## Scenario S6: Vector Store Comparison — FAISS vs. ChromaDB & Outdated Records

### Question Analysis:
Evaluating FAISS vs. ChromaDB for complaint resolution stored as embeddings, addressing query speed vs. outdated resolved complaint records.

### Comparison of FAISS and ChromaDB:

| Metric / Feature | FAISS (Facebook AI Similarity Search) | ChromaDB |
| :--- | :--- | :--- |
| **Primary Focus** | High-performance, lightweight vector indexing and similarity search in memory/C++. | Developer-friendly, full-featured open-source vector database with metadata filtering and persistence. |
| **Best Fit Scenario** | **Large-scale static vector datasets** where maximum retrieval speed and memory efficiency are paramount. | **Dynamic application databases** requiring metadata filtering (e.g., `status == 'active'`), document management, and persistent storage. |
| **CRUD & Updates** | Basic index structures (like `IndexFlatL2`) do not support easy in-place key-value updates or deletions without rebuilding. | Native support for document deletion (`collection.delete(ids=[...])`) and metadata updates. |

### Would Switching to ChromaDB Solve the Outdated-Record Problem?
**Yes, partially**, because ChromaDB supports **metadata filtering** (`where={"status": "unresolved"}`) and direct record deletion (`collection.delete(ids=[...])`), allowing resolved complaints to be filtered out during queries or deleted immediately.

### Essential Additional Step Required Regardless of Vector Store:
Regardless of vector store choice, an **Automated Lifecycle Synchronization Pipeline** must be implemented:
1. **Metadata Tagging**: Attach metadata fields (`complaint_status: "resolved"|"active"`, `resolved_timestamp`) to every embedding.
2. **Pre-Filter Querying**: Apply strict metadata pre-filtering (`status == 'active'`) on every similarity search call so resolved complaints are never returned to the LLM.
