# Advanced RAG System

An Advanced Retrieval-Augmented Generation (RAG) system built with **Python, LangChain, FAISS, Hugging Face embeddings, Gemini, BM25, Parent-Child Retrieval, Conversation Memory, and LangGraph**.

The project demonstrates how a RAG system can evolve from basic semantic retrieval to hybrid retrieval and finally to a **Parent-Child RAG architecture with LangGraph-based workflow orchestration**.

---

## 🚀 Features

### Basic RAG

* PDF document loading
* Text chunking
* Hugging Face embeddings
* FAISS vector search
* Similarity-based retrieval
* Gemini-powered answer generation

### Hybrid RAG

* Combines:

  * FAISS semantic search
  * BM25 keyword-based search
* Uses both semantic and lexical retrieval
* Removes duplicate documents before generating the answer

### Parent-Child RAG

* Splits documents into larger **parent chunks**
* Splits each parent into smaller **child chunks**
* Performs retrieval against child chunks
* Uses the corresponding parent documents as the final context
* Provides broader context to the LLM while keeping retrieval precise

### LangGraph

The Parent-Child RAG workflow is orchestrated using LangGraph.

The workflow includes:

* Question processing
* Follow-up question resolution
* Clarification handling
* Parent-Child retrieval
* Relevance checking
* Document-based answer generation
* General knowledge answer generation
* Conversation memory

### Conversation Memory

* Stores recent questions and answers
* Supports follow-up questions
* Resolves questions such as:

```text
What are the health effects of air pollution?
```

followed by:

```text
What about children?
```

into:

```text
What are the effects of air pollution on children?
```

### Clarification Handling

If an incomplete question is asked without previous context:

```text
What about children?
```

the system asks:

```text
Could you please provide more context for your question?
```

### General Knowledge Fallback

If the question is unrelated to the uploaded document, the system can use Gemini's general knowledge.

Example:

```text
What is the capital of France?
```

Output:

```text
Paris
```

---

## 🏗️ Architecture

### Basic RAG

```text
PDF
 │
 ▼
PDF Loader
 │
 ▼
Text Splitter
 │
 ▼
Embeddings
 │
 ▼
FAISS
 │
 ▼
Retriever
 │
 ▼
Gemini
 │
 ▼
Answer
```

### Hybrid RAG

```text
                    PDF
                     │
                     ▼
              Text Splitting
                 /       \
                /         \
               ▼           ▼
            FAISS         BM25
          Semantic       Keyword
          Retrieval      Retrieval
               \           /
                \         /
                 ▼       ▼
                 Hybrid
                Retriever
                    │
                    ▼
                  Gemini
                    │
                    ▼
                  Answer
```

### Parent-Child RAG + LangGraph

```text
                         User Question
                              │
                              ▼
                    ┌──────────────────┐
                    │    LangGraph     │
                    └────────┬─────────┘
                             │
                             ▼
                     Process Question
                             │
                  ┌──────────┴──────────┐
                  │                     │
             Follow-up?            Normal Question
                  │                     │
                  ▼                     ▼
          Resolve using            Child FAISS
          conversation memory       Retrieval
                                        │
                                        ▼
                                Relevance Check
                                  /          \
                                Yes           No
                                │             │
                                ▼             ▼
                         Parent Documents   General
                                │           Knowledge
                                │             │
                                └──────┬──────┘
                                       ▼
                                    Gemini
                                       │
                                       ▼
                               Save Conversation
```

---

## 📂 Project Structure

```text
advanced-rag/
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
│   ├── parent_store.py
│   ├── parent_child_retriever.py
│   └── parent_child_answer_generator.py
│
├── memory/
│   ├── conversation_memory.py
│   └── question_handler.py
│
├── vectorstore/
│
├── graph/
│
├── mcp/
│
├── app.py
├── config.py
├── requirements.txt
├── .env
└── .gitignore
```

---

## 🛠️ Technologies Used

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| Python                | Application development            |
| LangChain             | RAG components and LLM integration |
| LangGraph             | Workflow orchestration             |
| FAISS                 | Vector similarity search           |
| BM25                  | Keyword-based retrieval            |
| Hugging Face          | Text embeddings                    |
| Sentence Transformers | `all-MiniLM-L6-v2` embeddings      |
| Gemini                | Answer generation                  |
| PyPDF                 | PDF document loading               |
| python-dotenv         | Environment variable management    |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Samyu1904/advanced-rag.git
cd advanced-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Git Bash:

```bash
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Do not commit the `.env` file to GitHub.

It is already included in `.gitignore`.

---

## ▶️ Running the Application

Run:

```bash
python app.py
```

The application will ask for the full path of a PDF.

Example:

```text
Enter the full path of your PDF:
C:\Users\YourName\Downloads\air-quality-and-health.pdf
```

Then select a mode:

```text
========================================
              SELECT MODE
========================================

1. Basic RAG
2. Hybrid RAG
3. Parent-Child RAG + LangGraph
4. Exit
```

---

## 🧪 Parent-Child RAG Example

Example document question:

```text
What are the health effects of air pollution?
```

The system retrieves relevant child chunks and then returns their corresponding parent documents as context.

Example output:

```text
Searching Parent-Child chunks...
Best child FAISS score: 0.63

Source: Parent Document

Relevant to PDF:
True
```

The answer is then generated using the retrieved parent context.

---

## 🔄 Follow-Up Question Example

First question:

```text
What are the health effects of air pollution?
```

Follow-up:

```text
What about children?
```

The question handler resolves it to:

```text
What are the effects of air pollution on children?
```

The resolved question is then sent through the Parent-Child retrieval pipeline.

---

## ❓ Clarification Example

If the conversation starts with:

```text
What about children?
```

there is no previous question to provide context.

The system detects the incomplete question:

```text
Waiting for Clarification:
True
```

and responds:

```text
Could you please provide more context for your question?
```

---

## 🌐 General Knowledge Example

For a question unrelated to the uploaded document:

```text
What is the capital of France?
```

the system checks the retrieval relevance.

If the question is not relevant to the PDF:

```text
Relevant to PDF:
False

Source: General Knowledge
```

Gemini then generates the general answer:

```text
Paris
```

---

## 🌿 Git Branches

The repository contains separate branches for the RAG implementations.

### `main`

Contains the existing:

* Basic RAG
* Hybrid RAG

### `parent-child-rag`

Contains:

* Basic RAG
* Hybrid RAG
* Parent-Child RAG
* LangGraph workflow
* Conversation memory
* Follow-up question handling
* Clarification handling
* General knowledge fallback

Switch to the Parent-Child branch:

```bash
git checkout parent-child-rag
```

Or:

```bash
git switch parent-child-rag
```

---

## 🧠 Why Parent-Child RAG?

Traditional RAG commonly uses the same chunks for both retrieval and LLM context.

Parent-Child RAG separates these responsibilities.

### Child chunks

Smaller chunks are used for retrieval.

Advantages:

* More precise matching
* Better semantic search
* Smaller retrieval units

### Parent chunks

Larger chunks are returned after a child match.

Advantages:

* More surrounding context
* Better information completeness
* Reduces the chance of answering from an isolated sentence

The basic idea is:

```text
Large Parent Document
        │
        ├── Child 1
        ├── Child 2
        ├── Child 3
        └── Child 4

Question
   │
   ▼
Search Child Chunks
   │
   ▼
Find Matching Child
   │
   ▼
Get Parent
   │
   ▼
Send Parent Context to LLM
```

---

## 🧩 Why LangGraph?

LangGraph allows the RAG system to be represented as a workflow rather than a simple sequence of function calls.

The current workflow handles different paths:

```text
Question
   │
   ▼
Process Question
   │
   ├── Needs clarification ──► Clarification
   │
   ▼
Retrieve
   │
   ├── Relevant ──► PDF Answer
   │
   └── Not Relevant ──► General Answer
                              │
                              ▼
                         Save Memory
```

This makes the application easier to extend with additional agents, tools, routing logic, and memory in future versions.

---

## 📌 Current Implementation

The current Parent-Child implementation uses:

```text
Parent Chunk Size: 2000
Parent Chunk Overlap: 200

Child Chunk Size: 500
Child Chunk Overlap: 100

Embedding Model:
sentence-transformers/all-MiniLM-L6-v2
```

The tested example PDF produced:

```text
Parent chunks: 5
Child chunks: 22
```

---

## 🔒 Security

The Gemini API key is stored in `.env`.

The following files/directories are excluded from Git:

```text
.env
.venv/
__pycache__/
*.pyc
vectorstore/
retrieval/vetorstore/
```

Never commit API keys or other secrets to the repository.

---

## 🚀 Future Enhancements

Possible future improvements include:

* LangGraph multi-agent workflows
* MCP integration
* Persistent conversation memory
* Better hybrid score fusion
* Cross-encoder reranking
* Metadata filtering
* Multiple PDF support
* Document citations
* Streaming responses
* Web interface
* Evaluation metrics
* Retrieval quality evaluation
* Persistent vector databases

---

## 👩‍💻 Author

**T. Sai Madhu Samyuktha**

GitHub:

https://github.com/Samyu1904
