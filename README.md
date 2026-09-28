# Reranking RAG

A Retrieval-Augmented Generation (RAG) system that improves document retrieval by combining **FAISS** and **BM25** retrieval with a **Cross-Encoder reranker** before generating the final answer using **Gemini**.

## Overview

Traditional retrieval methods may return relevant documents, but the most relevant document may not always appear at the top.

Reranking RAG solves this by:

1. Retrieving multiple candidate documents.
2. Combining results from FAISS and BM25.
3. Removing duplicate documents.
4. Passing the candidate documents through a Cross-Encoder.
5. Ranking the documents based on their relevance to the question.
6. Selecting the top relevant documents.
7. Sending them to Gemini for answer generation.

## Reranking Pipeline

```text
User Question
      ↓
FAISS Retrieval
      ↓
BM25 Retrieval
      ↓
Candidate Documents
      ↓
Remove Duplicates
      ↓
Cross-Encoder Reranking
      ↓
Top Relevant Documents
      ↓
Gemini
      ↓
Final Answer
```

## Technologies Used

* Python
* LangChain
* FAISS
* BM25
* Sentence Transformers
* Cross-Encoder
* Gemini
* PyPDF

## Reranker Model

The project uses the following Cross-Encoder model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model receives a pair consisting of:

```text
(question, document)
```

and produces a relevance score.

The documents are then sorted according to their scores, and the highest-scoring documents are selected.

## Retrieval Strategy

The system first retrieves candidate documents using:

### FAISS

FAISS performs semantic similarity search using document embeddings.

### BM25

BM25 performs keyword-based retrieval and helps find documents containing important terms from the question.

The results from both retrieval methods are combined and duplicate documents are removed.

## Reranking

The combined candidate documents are passed to the Cross-Encoder.

The Cross-Encoder evaluates:

```text
Question + Document
```

and assigns a relevance score.

The documents with the highest scores are selected as the final context for Gemini.

Current configuration:

```text
Candidate documents: 10
Final documents: 3
```

## Project Files

```text
retrieval/
├── reranker.py
└── reranking_retriever.py
```

### `reranker.py`

Contains the `DocumentReranker` class.

Responsibilities:

* Load the Cross-Encoder model.
* Create question-document pairs.
* Calculate relevance scores.
* Sort documents by relevance.
* Return the top documents.

### `reranking_retriever.py`

Contains the `RerankingRetriever` class.

Responsibilities:

* Retrieve candidates using FAISS.
* Retrieve candidates using BM25.
* Combine retrieval results.
* Remove duplicate documents.
* Send candidates to the Cross-Encoder.
* Return the top-ranked documents.

## Testing

The Reranking RAG pipeline was tested using:

```text
What are the health effects of air pollution?
```

The system successfully:

* Retrieved candidate documents.
* Combined FAISS and BM25 results.
* Removed duplicate documents.
* Reranked the candidates using the Cross-Encoder.
* Selected the top relevant documents.
* Generated the final answer using Gemini.

## Author

**Thadikamalla Sai Madhu Samyuktha**

B.Tech – Information Technology
Pragati Engineering College
GitHub: [Samyu1904](https://github.com/Samyu1904)
