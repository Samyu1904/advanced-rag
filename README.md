# Self-RAG

## Overview

Self-RAG (Self-Reflective Retrieval-Augmented Generation) is a RAG architecture that evaluates its own retrieved documents and generated answers before producing the final response.

This implementation is built as a standalone Self-RAG application using:

- FAISS for semantic retrieval
- BM25 for keyword-based retrieval
- Google Gemini for evaluation and answer generation
- Conversation memory for follow-up questions
- Self-reflection for retrieved documents and generated answers

The standalone application is implemented through `app1.py`.

---

## Project Structure

```text
self-rag/
│
├── self_rag/
│   ├── self_rag_evaluator.py
│   ├── self_rag_retriever.py
│   ├── self_rag_answer_generator.py
│   ├── self_rag_question_handler.py
│   └── self_rag_pipeline.py
│
├── app1.py
│
├── ingestion/
│   ├── pdf_loader.py
│   └── text_splitter.py
│
├── retrieval/
│   ├── retriever.py
│   └── bm25_retriever.py
│
├── vectorstore/
│   └── faiss_store.py
│
└── README.md
