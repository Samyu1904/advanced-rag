# 🧠 Graph RAG — Knowledge Graph Based Retrieval-Augmented Generation

> A Graph-based Retrieval-Augmented Generation system that converts information from a PDF into a **Knowledge Graph** of entities and relationships, retrieves relevant graph relationships for user questions, and uses **Google Gemini** to generate grounded answers.

---

## 🌟 Overview

Traditional RAG systems retrieve text chunks from a vector database and provide those chunks to an LLM.

**Graph RAG** takes a different approach.

Instead of relying only on text similarity, this project:

```text
📄 PDF Document
      ↓
📝 Text Extraction
      ↓
✂️ Text Chunking
      ↓
🧠 Entity & Relationship Extraction
      ↓
🕸️ Knowledge Graph
      ↓
🔎 Graph Retrieval
      ↓
📚 Relevant Relationships
      ↓
🤖 Gemini
      ↓
💬 Final Answer
```

The system represents information as:

```text
Entity ──[Relationship]──> Entity
```

For example:

```text
Air Pollution
      │
      ├──[affects]──> Human Health
      │
      ├──[contains]──> Particulate Matter
      │
      └──[causes]──> Respiratory Problems
```

This makes it possible to retrieve **relationships between entities**, rather than retrieving only isolated text chunks.

---

# ✨ Features

### 🕸️ Knowledge Graph Construction

The system automatically extracts:

* Entities
* Entity types
* Relationships
* Source entities
* Target entities

from PDF document chunks.

---

### 📦 Batched Graph Extraction

Instead of making one Gemini request for every chunk, chunks are processed in batches.

```text
Chunk 1
Chunk 2
Chunk 3
      ↓
  Gemini
      ↓
Entities + Relationships
```

The current implementation uses:

```python
batch_size=3
```

This reduces unnecessary API requests during graph construction.

---

### 🔎 Graph-Based Retrieval

The retriever:

1. Tokenizes the user's question.
2. Finds matching graph entities.
3. Finds relationships connected to those entities.
4. Expands the search to connected entities.
5. Returns the most relevant graph relationships.

---

### 🤖 Gemini-Powered Answer Generation

Google Gemini generates the final response using the retrieved knowledge graph context.

The answer generator is instructed to:

* Use only graph information.
* Avoid inventing facts.
* Avoid outside knowledge.
* Clearly indicate when the graph does not contain enough information.

---

### 💬 Conversation Memory

Graph RAG supports conversational interaction.

Example:

```text
User:
What are the health effects of air pollution?

Graph RAG:
...

User:
What about children?

Graph RAG:
...
```

The system maintains recent conversation history and uses the previous question when processing supported follow-up questions.

---

### 🧹 Memory Management

The interactive application provides:

```text
history
clear
exit
```

#### `history`

Displays previous questions and answers.

#### `clear`

Clears the conversation memory.

#### `exit`

Returns to the main RAG menu.

---

### 🌍 General Knowledge Fallback

If the knowledge graph does not contain a matching entity or relationship, the system can fall back to Gemini's general knowledge response.

```text
Question
   ↓
Graph Search
   ↓
No graph match
   ↓
General Knowledge
   ↓
Gemini Answer
```

---

### 🛡️ API Error Handling

The implementation handles Gemini API failures, including quota-related errors.

For example, when a `429` error occurs, the system provides a user-friendly message instead of crashing.

---

# 🏗️ Architecture

The Graph RAG implementation consists of six main components.

```text
                         ┌───────────────────┐
                         │    PDF Document   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    PDF Loader     │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Text Splitter   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  Graph Builder    │
                         │      Gemini       │
                         └─────────┬─────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                    ┌──────────┐      ┌──────────┐
                    │ Entities │      │Relations │
                    └────┬─────┘      └────┬─────┘
                         └────────┬────────┘
                                  ▼
                         ┌───────────────────┐
                         │ Knowledge Graph   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Graph Retriever   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Graph Context     │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ Gemini Answer     │
                         │ Generator         │
                         └─────────┬─────────┘
                                   │
                                   ▼
                            💬 Final Answer
```

---

# 📁 Project Structure

```text
advanced-rag/
│
├── data/
│   └── documents/
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   └── parent_child_splitter.py
│
├── retrieval/
│   ├── retriever.py
│   ├── answer_generator.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── reranking_retriever.py
│   ├── parent_store.py
│   ├── parent_child_retriever.py
│   └── parent_child_answer_generator.py
│
├── memory/
│   ├── conversation_memory.py
│   └── question_handler.py
│
├── vectorstore/
│   ├── faiss_store.py
│   └── child_vectorstore.py
│
├── advanced/
│   ├── multi_query_retriever.py
│   ├── advanced_retriever.py
│   ├── advanced_reranking_retriever.py
│   ├── advanced_answer_generator.py
│   ├── advanced_question_handler.py
│   └── advanced_pipeline.py
│
├── crag/
│   ├── crag_evaluator.py
│   ├── crag_corrector.py
│   ├── crag_retriever.py
│   ├── crag_answer_generator.py
│   ├── crag_question_handler.py
│   └── crag_pipeline.py
│
├── self_rag/
│   ├── self_rag_evaluator.py
│   ├── self_rag_retriever.py
│   ├── self_rag_answer_generator.py
│   ├── self_rag_question_handler.py
│   └── self_rag_pipeline.py
│
├── agentic_rag/
│   ├── agentic_rag_tools.py
│   ├── agentic_rag_agent.py
│   ├── agentic_rag_evaluator.py
│   ├── agentic_rag_answer_generator.py
│   ├── agentic_rag_question_handler.py
│   └── agentic_rag_pipeline.py
│
├── graph_rag/
│   ├── graph_builder.py
│   ├── graph_retriever.py
│   ├── graph_answer_generator.py
│   ├── graph_evaluator.py
│   ├── graph_question_handler.py
│   └── graph_pipeline.py
│
├── mcp/
│
├── app.py
├── app1.py
├── config.py
├── requirements.txt
├── .env
└── .gitignore
```

---

# 🔍 Graph RAG Components

## 1. `graph_builder.py`

Responsible for creating the knowledge graph.

### Main responsibilities

```text
PDF chunks
    ↓
Gemini
    ↓
Entities
+
Relationships
    ↓
Knowledge Graph
```

The builder processes document chunks in batches.

Example:

```python
builder = GraphBuilder(
    batch_size=3
)
```

The graph contains:

```python
{
    "nodes": {},
    "edges": []
}
```

---

## 2. `graph_retriever.py`

Responsible for retrieving relevant graph information.

The retrieval process is:

```text
User Question
      ↓
Tokenization
      ↓
Entity Matching
      ↓
Relationship Matching
      ↓
Graph Expansion
      ↓
Relevant Relationships
```

A relationship is represented as:

```text
source --[relationship]--> target
```

For example:

```text
Air Pollution --[affects]--> Human Health
```

---

## 3. `graph_answer_generator.py`

Uses Gemini to generate the final answer.

The model receives:

```text
Question
+
Graph Context
```

and generates an answer based only on the retrieved graph information.

The prompt explicitly instructs the model not to invent facts.

---

## 4. `graph_question_handler.py`

Responsible for conversational questions.

It handles:

* New questions
* Follow-up questions
* Conversation history
* Memory clearing

Example:

```text
Question 1:
What are the effects of air pollution?

Question 2:
What about children?
```

The second question can be converted into a search question using the previous question as context.

---

## 5. `graph_pipeline.py`

Connects all Graph RAG components.

```text
Question Handler
       ↓
Graph Retriever
       ↓
Graph Context
       ↓
Answer Generator
       ↓
Conversation Memory
```

It is the central orchestration layer of Graph RAG.

---

## 6. `graph_evaluator.py`

Provides lightweight graph and answer evaluation.

It checks whether:

* Graph context exists.
* An answer was generated.
* The answer has graph context available for support.

The current optimized pipeline does not perform additional Gemini calls for evaluation.

This helps reduce unnecessary API usage.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Navigate into the project:

```bash
cd advanced-rag
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

For Git Bash:

```bash
source .venv/Scripts/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Configuration

Create a `.env` file in the project root:

```text
advanced-rag/
└── .env
```

Add:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace:

```text
your_gemini_api_key_here
```

with your actual Gemini API key.

### ⚠️ Important

Never upload your `.env` file to GitHub.

The project already uses:

```text
.env
```

inside `.gitignore`.

---

# 📄 Running Graph RAG Independently

You can test Graph RAG independently using:

```text
app1.py
```

Run:

```bash
python app1.py
```

The application asks for the PDF path:

```text
Enter PDF path:
```

Example:

```text
C:\Users\samyu\Downloads\air-quality-and-health.pdf
```

---

# 🖥️ Running the Complete Advanced RAG Application

Run:

```bash
python app.py
```

The application provides multiple RAG architectures:

```text
1. Basic RAG
2. Hybrid RAG
3. Parent-Child RAG + LangGraph
4. Reranking RAG
5. Advanced RAG
6. CRAG
7. Self-RAG
8. Agentic RAG
9. Graph RAG
10. Exit
```

Select:

```text
9
```

to start Graph RAG.

---

# 🧪 Graph RAG Interaction

After the knowledge graph is created:

```text
======================================================================
GRAPH RAG READY
======================================================================

Commands:
  history -> show conversation history
  clear   -> clear conversation memory
  exit    -> return to main menu
```

You can then ask questions about the uploaded PDF.

Example:

```text
Question: What are the health effects of air pollution?
```

The system performs:

```text
Question
   ↓
Entity Matching
   ↓
Relationship Retrieval
   ↓
Graph Context
   ↓
Gemini
   ↓
Answer
```

---

# 💬 Conversation Example

```text
Question:
What are the health effects of air pollution?

Answer:
[Answer generated from the retrieved knowledge graph]

Question:
What about children?

Answer:
[Follow-up answer using the previous question as context]
```

---

# 🧹 Conversation Commands

## View History

Type:

```text
history
```

Example:

```text
Conversation 1
----------------------------------------------------------------------
Question:
What are the health effects of air pollution?

Answer:
...

Conversation 2
----------------------------------------------------------------------
Question:
What about children?

Answer:
...
```

---

## Clear Memory

Type:

```text
clear
```

This clears:

* Conversation history
* Previous search question

---

## Exit Graph RAG

Type:

```text
exit
```

This returns to the main Advanced RAG menu.

---

# 🧠 How Graph RAG Differs from Traditional RAG

### Traditional Vector RAG

```text
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
Similar Text Chunks
   ↓
LLM
   ↓
Answer
```

### Graph RAG

```text
Question
   ↓
Entity Matching
   ↓
Relationship Search
   ↓
Connected Graph Information
   ↓
LLM
   ↓
Answer
```

The key difference is the representation of information.

Traditional RAG primarily retrieves **text similarity**.

Graph RAG retrieves **relationships between entities**.

---

# 📊 Knowledge Graph Representation

Graph RAG represents information using nodes and edges.

### Nodes

Nodes represent entities.

```text
Air Pollution
Human Health
Particulate Matter
Respiratory Disease
```

### Edges

Edges represent relationships.

```text
Air Pollution
      │
      │ affects
      ▼
Human Health
```

Programmatically:

```python
{
    "source": "Air Pollution",
    "relationship": "affects",
    "target": "Human Health"
}
```

---

# ⚡ API Optimization

A major design consideration in this project is reducing unnecessary Gemini API requests.

An earlier approach could potentially make multiple LLM calls during:

```text
Graph Construction
+
Graph Evaluation
+
Answer Generation
+
Answer Evaluation
+
Regeneration
```

The optimized implementation instead focuses on:

```text
Graph Construction
        +
Answer Generation
```

and uses deterministic Python logic for graph retrieval and lightweight evaluation.

This reduces unnecessary API usage and helps avoid hitting free-tier request limits.

---

# 🛡️ Grounding Strategy

Graph answers are generated using the retrieved graph context.

The answer generator is instructed:

```text
Use only information contained
in the knowledge graph.

Do not invent facts.

Do not add outside knowledge.
```

If sufficient graph information is unavailable:

```text
I could not find enough information
in the knowledge graph.
```

This helps reduce unsupported answers.

---

# 🧰 Technologies Used

| Technology                  | Purpose                                |
| --------------------------- | -------------------------------------- |
| 🐍 Python                   | Application development                |
| 📄 PyPDF                    | PDF loading                            |
| ✂️ LangChain Text Splitters | Document chunking                      |
| 🤗 Hugging Face             | Embeddings in other RAG components     |
| 🧠 Google Gemini            | Graph extraction and answer generation |
| 🔗 LangChain                | LLM integration                        |
| 🕸️ Knowledge Graph         | Entity-relationship representation     |
| 💾 Conversation Memory      | Follow-up questions                    |
| 🖥️ VS Code                 | Development environment                |

---

# 📌 Current Graph RAG Workflow

```text
                    PDF
                     │
                     ▼
              ┌──────────────┐
              │  PDF Loader  │
              └──────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Text Splitter │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Graph Builder │
             │    Gemini     │
             └───────┬───────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
          Entities      Relationships
              │             │
              └──────┬──────┘
                     ▼
             ┌───────────────┐
             │ Knowledge     │
             │ Graph         │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │Graph Retriever│
             └───────┬───────┘
                     │
                     ▼
              Graph Context
                     │
                     ▼
             ┌───────────────┐
             │ Gemini Answer │
             │  Generator    │
             └───────┬───────┘
                     │
                     ▼
                 Final Answer
```

---

# 🎯 Learning Objectives

This implementation demonstrates several important concepts in modern RAG systems:

* Document ingestion
* Document chunking
* Knowledge graph construction
* Entity extraction
* Relationship extraction
* Graph-based retrieval
* Context expansion
* LLM-based answer generation
* Conversational memory
* Follow-up question handling
* API optimization
* Error handling
* Grounded generation

---

# 🔮 Possible Future Improvements

The current implementation can be extended with:

### 1. Persistent Graph Storage

Store the graph using technologies such as:

```text
Neo4j
NetworkX
ArangoDB
Amazon Neptune
```

### 2. Semantic Graph Retrieval

Combine:

```text
Keyword Matching
+
Embeddings
+
Graph Traversal
```

for more advanced retrieval.

### 3. Graph Visualization

Visualize:

```text
Nodes
+
Relationships
```

using graph visualization libraries.

### 4. Hybrid Graph + Vector RAG

Combine:

```text
Vector Search
       +
Graph Search
       ↓
Combined Context
       ↓
LLM
```

### 5. Multi-Hop Reasoning

Support questions that require traversing multiple relationships:

```text
Entity A
   ↓
Relationship 1
   ↓
Entity B
   ↓
Relationship 2
   ↓
Entity C
```

---

# 📚 Project Progression

This Graph RAG implementation is part of a larger Advanced RAG project.

The project currently explores:

```text
Basic RAG
    ↓
Hybrid RAG
    ↓
Parent-Child RAG
    ↓
Reranking RAG
    ↓
Advanced RAG
    ↓
CRAG
    ↓
Self-RAG
    ↓
Agentic RAG
    ↓
Graph RAG
```

Each architecture explores a different strategy for improving retrieval, reasoning, grounding, and interaction with documents.

---

# 👩‍💻 Author

**Thadikamalla Sai Madhu Samyuktha**

B.Tech — Information Technology

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Generative AI
* 🔎 Retrieval-Augmented Generation
* 🕸️ Knowledge Graphs
* 🐍 Python
* 💻 Full-Stack Development

---

# ⭐ Project Highlights

```text
📄 PDF → Knowledge Graph
🧠 Gemini-powered extraction
🕸️ Entity + Relationship representation
🔎 Graph-based retrieval
🤖 Grounded answer generation
💬 Conversational follow-ups
🧹 Conversation memory management
⚡ Batched API processing
🛡️ API error handling
🌍 General knowledge fallback
```

---

## 🚀 Final Result

Graph RAG transforms unstructured document information into a structured network of entities and relationships.

Instead of asking only:

> **"Which text chunk is similar to my question?"**

the system can ask:

> **"Which entities are related to the concepts in my question, and what relationships connect them?"**

That graph-based representation provides a foundation for more advanced **multi-hop reasoning, relationship-aware retrieval, and knowledge-intensive RAG applications.**

---

## 📜 License

This project is intended for learning, experimentation, and educational purposes.
