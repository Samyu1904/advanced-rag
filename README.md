# Advanced RAG System

An **Advanced Retrieval-Augmented Generation (RAG)** system built with Python, LangChain, FAISS, BM25, Multi-Query Retrieval, Cross-Encoder Reranking, Gemini, and conversational memory.

This project extends the previous RAG architectures by combining multiple retrieval and ranking techniques into a single advanced pipeline for more reliable document question answering.

---


# Project Overview

Traditional RAG retrieves documents using a single retrieval method.

This project progressively builds a more advanced RAG architecture by combining:

* PDF document loading
* Recursive text splitting
* Hugging Face embeddings
* FAISS vector search
* BM25 keyword search
* Hybrid retrieval
* Parent-Child retrieval
* Cross-Encoder reranking
* Multi-Query retrieval
* Conversational memory
* Follow-up question handling
* Gemini-based answer generation

The **Advanced RAG** architecture combines Multi-Query Retrieval, Hybrid Retrieval, and Cross-Encoder Reranking before generating the final answer.

---

# RAG Architecture Evolution

The project has been developed progressively through multiple RAG architectures.

```text
Basic RAG
   ↓
Hybrid RAG
   ↓
Parent-Child RAG + LangGraph
   ↓
Reranking RAG
   ↓
Advanced RAG
   ↓
CRAG
   ↓
Future RAG Architectures
```

The `advanced/` folder contains the implementation of the Advanced RAG layer.

---

# Advanced RAG Architecture

The Advanced RAG pipeline follows this workflow:

```text
                    User Question
                          │
                          ▼
                Conversation Memory
                          │
                          ▼
                Follow-up Detection
                          │
                          ▼
                 Contextual Question
                          │
                          ▼
                Document Relevance Check
                          │
                          ▼
                 Multi-Query Generation
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        FAISS Retrieval            BM25 Retrieval
             │                         │
             └────────────┬────────────┘
                          ▼
                 Candidate Documents
                          │
                          ▼
              Cross-Encoder Reranking
                          │
                          ▼
                 Top Relevant Documents
                          │
                          ▼
                Gemini Answer Generator
                          │
                          ▼
                    Final Answer
                          │
                          ▼
                Conversation Memory
```

---

# Main Advanced RAG Components

## 1. Multi-Query Retrieval

A single user question may not match the wording used in the document.

The Multi-Query component uses Gemini to generate multiple search queries representing the same information need from different perspectives.

For example:

```text
Original Question
        ↓
Query 1
Query 2
Query 3
        ↓
Document Retrieval
```

This improves the possibility of finding relevant information when the user's wording differs from the document wording.

### File

```text
advanced/multi_query_retriever.py
```

---

# 2. Advanced Retriever

The Advanced Retriever combines:

* Multi-Query generation
* FAISS retrieval
* BM25 retrieval
* Candidate document collection
* Duplicate removal

The generated queries are sent through both semantic and keyword retrieval systems.

```text
User Question
      ↓
Multiple Queries
      ↓
 ┌───────────────┐
 │               │
 ▼               ▼
FAISS           BM25
 │               │
 └───────┬───────┘
         ▼
Candidate Documents
         ↓
Duplicate Removal
         ↓
Selected Candidates
```

### File

```text
advanced/advanced_retriever.py
```

---

# 3. Cross-Encoder Reranking

Retrieval systems may return several potentially relevant documents.

The Cross-Encoder reranker evaluates the relationship between:

```text
Question + Document
```

and assigns a relevance score.

The candidates are then sorted according to the reranker scores.

### Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### File

```text
retrieval/reranker.py
```

---

# 4. Advanced Reranking Retriever

This component combines the Advanced Retriever with the Cross-Encoder reranker.

Workflow:

```text
Multi-Query Retrieval
        ↓
FAISS + BM25
        ↓
Candidate Documents
        ↓
Cross-Encoder
        ↓
Reranked Documents
        ↓
Top K Documents
```

### File

```text
advanced/advanced_reranking_retriever.py
```

---

# 5. Advanced Answer Generator

The final retrieved documents are passed to Gemini to generate the answer.

The generator instructs Gemini to:

* Use the retrieved document context
* Avoid inventing information
* Give a clear answer
* Avoid exposing internal retrieval details
* State when the required information cannot be found

### File

```text
advanced/advanced_answer_generator.py
```

---

# 6. Conversational Memory

The system maintains recent conversation history.

This allows the application to handle follow-up questions such as:

```text
User:
What are the health effects of air pollution?

User:
What about children?
```

The second question can be interpreted using the context of the previous question.

### Existing component

```text
memory/conversation_memory.py
```

---

# 7. Advanced Question Handler

The Advanced Question Handler detects follow-up questions and creates a contextual search question.

For example:

```text
Previous:
What are the health effects of air pollution?

Follow-up:
What about children?
```

The system creates a contextual search question similar to:

```text
What are the health effects of air pollution?
with specific focus on what about children?
```

### File

```text
advanced/advanced_question_handler.py
```

---

# 8. Advanced RAG Pipeline

The complete Advanced RAG workflow is controlled by:

```text
advanced/advanced_pipeline.py
```

The pipeline handles:

1. User question
2. Follow-up detection
3. Contextual question creation
4. Document relevance checking
5. Advanced retrieval
6. Cross-Encoder reranking
7. Answer generation
8. Conversation memory

---

# Project Structure

```text
advanced-rag/
│
├── advanced/
│   ├── multi_query_retriever.py
│   ├── advanced_retriever.py
│   ├── advanced_reranking_retriever.py
│   ├── advanced_answer_generator.py
│   ├── advanced_question_handler.py
│   └── advanced_pipeline.py
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
├── vectorstore/
│   ├── faiss_store.py
│   └── child_vectorstore.py
│
├── memory/
│   ├── conversation_memory.py
│   └── question_handler.py
│
├── graph/
│
├── mcp/
│
├── app1.py
├── config.py
├── requirements.txt
├── .env
└── .gitignore
```

---

# Technologies Used

| Technology            | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| Python                | Core programming language                           |
| LangChain             | RAG components and LLM integration                  |
| LangGraph             | Workflow orchestration used in earlier architecture |
| FAISS                 | Vector similarity search                            |
| BM25                  | Keyword-based retrieval                             |
| Sentence Transformers | Embeddings and Cross-Encoder reranking              |
| Hugging Face          | Embedding model                                     |
| Gemini                | Query generation and answer generation              |
| PyPDF                 | PDF document loading                                |
| python-dotenv         | Environment variable management                     |

---

# Models Used

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used for generating document and query embeddings.

---

## Cross-Encoder Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Used for reranking retrieved candidate documents.

---

## Gemini Model

```text
gemini-3.6-flash
```

Used for:

* Multi-query generation
* Document-based answer generation
* General answer generation

---

# Document Processing

The application accepts a PDF document at runtime.

The document is processed using the following pipeline:

```text
PDF
 ↓
PyPDFLoader
 ↓
Pages
 ↓
Recursive Character Text Splitter
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
```

The default text splitting configuration is:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

---

# Retrieval Strategy

Advanced RAG uses multiple retrieval approaches.

## Semantic Retrieval

FAISS searches documents using vector similarity.

```text
Question
   ↓
Embedding
   ↓
FAISS
   ↓
Similar Documents
```

## Keyword Retrieval

BM25 searches using lexical keyword matching.

```text
Question
   ↓
Tokens
   ↓
BM25
   ↓
Keyword-Matching Documents
```

## Combined Retrieval

The Advanced Retriever combines candidates from both systems.

```text
             Question
                 │
        ┌────────┴────────┐
        ▼                 ▼
      FAISS              BM25
        │                 │
        └────────┬────────┘
                 ▼
        Candidate Documents
```

---

# Reranking

After candidate retrieval, the Cross-Encoder evaluates the candidates more directly.

```text
Candidate Documents
        ↓
Cross-Encoder
        ↓
Relevance Scores
        ↓
Sorted Documents
        ↓
Top 3 Documents
```

The final top documents are then passed to Gemini.

---

# Relevance Handling

Before running the complete Advanced retrieval pipeline, the system performs an initial FAISS relevance check.

If the question is unrelated to the uploaded document, the system can generate a general knowledge response instead of treating irrelevant document chunks as the source.

Example:

```text
Document:
Air Quality and Health

Question:
What is the capital of France?
```

The system can identify that the question is unrelated to the uploaded document and use the general-answer path.

---

# Follow-Up Questions

The system supports contextual follow-up questions.

Example:

```text
Question 1:
What are the health effects of air pollution?

Question 2:
What about children?
```

The question handler uses the previous conversation context to create a more complete search question.

This allows the retrieval system to search using the combined context rather than treating the follow-up as an isolated question.

---

# Dynamic PDF Input

The Advanced RAG application does not require a fixed PDF path.

At runtime, the user provides:

```text
Enter the full path of your PDF:
```

Example:

```text
C:\Users\samyu\Downloads\air-quality-and-health.pdf
```

The application then loads and processes the selected document.

---

# Running Advanced RAG

Activate the virtual environment:

```bash
.venv\Scripts\Activate.ps1
```

Run the Advanced RAG application:

```bash
python app1.py
```

The application asks for the PDF path at runtime.

Example:

```text
Enter the full path of your PDF:
C:\Users\samyu\Downloads\air-quality-and-health.pdf
```

---

# Available Commands

Inside the Advanced RAG chat:

```text
clear
```

Clears the conversation memory.

```text
exit
```

Returns from the Advanced RAG application.

---

# Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

The API key should not be committed to GitHub.

The `.gitignore` file contains:

```text
.env
```

---

# Installation

Create and activate the virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Advanced RAG Pipeline Summary

The complete architecture can be summarized as:

```text
                         USER
                          │
                          ▼
                   User Question
                          │
                          ▼
                Conversation Memory
                          │
                          ▼
                Question Processing
                          │
                          ▼
                Relevance Checking
                          │
                          ▼
                 Multi-Query LLM
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Semantic Search          Keyword Search
            FAISS                    BM25
              │                       │
              └───────────┬───────────┘
                          ▼
                 Candidate Documents
                          │
                          ▼
                 Cross-Encoder
                   Reranking
                          │
                          ▼
                   Top Documents
                          │
                          ▼
                    Gemini LLM
                          │
                          ▼
                    Final Answer
                          │
                          ▼
                Conversation Memory
```

---

# What This Advanced RAG Adds

Compared with a basic RAG implementation, this architecture introduces:

* Multi-query generation
* Multiple retrieval strategies
* Candidate document aggregation
* Cross-Encoder reranking
* Follow-up question handling
* Conversational memory
* Document relevance checking
* Dynamic PDF input
* Dedicated end-to-end Advanced RAG pipeline

---

# Development Approach

The RAG system was developed incrementally.

```text
Basic RAG
   ↓
Hybrid Retrieval
   ↓
Parent-Child Retrieval
   ↓
LangGraph Integration
   ↓
Reranking
   ↓
Multi-Query Retrieval
   ↓
Advanced Reranking
   ↓
Advanced RAG Pipeline
```

Each architecture was tested independently before being integrated into the larger RAG project.

---

# Future Work

The main project is planned to continue with additional RAG architectures, including:

```text
Advanced RAG
     ↓
CRAG
     ↓
Additional RAG Architectures
```

The current branch focuses specifically on the **Advanced RAG implementation**.

---

# Author

**Thadikamalla Sai Madhu Samyuktha**

GitHub: [Samyu1904](https://github.com/Samyu1904)

---

## License

This project is intended for educational and portfolio purposes.
