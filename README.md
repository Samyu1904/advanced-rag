# Corrective RAG (CRAG)

A **Corrective Retrieval-Augmented Generation (CRAG)** system that improves the reliability of document-based question answering by evaluating retrieved documents and correcting the retrieval process when the retrieved information is not sufficiently relevant.

This project is implemented as part of an **Advanced RAG system** using Python, FAISS, BM25, Cross-Encoder reranking, and Google Gemini.

---

## 📌 What is CRAG?

Traditional RAG retrieves documents and directly sends them to an LLM.

CRAG adds an important step:

```text
User Question
      ↓
Initial Retrieval
      ↓
Relevance Evaluation
      ↓
 ┌───────────────┐
 │ Relevant?     │
 └───────┬───────┘
       Yes │ No
          ↓   ↓
      Use Docs  Correct Retrieval
          │          ↓
          │     FAISS + BM25
          │          ↓
          │     Re-evaluation
          │          ↓
          └──────┬───┘
                 ↓
            Answer Generation
```

The system checks whether the retrieved documents are relevant before generating the final answer.

If the documents are not relevant enough, CRAG performs another retrieval step using FAISS and BM25.

---

# 🚀 Features

* PDF-based question answering
* FAISS semantic retrieval
* BM25 keyword retrieval
* Cross-Encoder relevance evaluation
* Corrective retrieval
* Gemini-powered answer generation
* Conversation memory
* Follow-up question handling
* General knowledge fallback
* Conversation history
* Clear conversation memory
* Interactive command-line interface
* Integrated with the main Advanced RAG application

---

# 🛠️ Technologies Used

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| Python                | Core programming language          |
| LangChain             | LLM integration and RAG components |
| FAISS                 | Semantic vector retrieval          |
| BM25                  | Keyword-based retrieval            |
| Sentence Transformers | Cross-Encoder reranking            |
| Google Gemini         | Answer generation                  |
| PyPDF                 | PDF document loading               |
| python-dotenv         | Environment variable management    |

---

# 📂 Project Structure

```text
advanced-rag/
│
├── crag/
│   ├── crag_evaluator.py
│   ├── crag_corrector.py
│   ├── crag_retriever.py
│   ├── crag_answer_generator.py
│   ├── crag_question_handler.py
│   └── crag_pipeline.py
│
├── ingestion/
│   ├── pdf_loader.py
│   └── text_splitter.py
│
├── retrieval/
│   ├── retriever.py
│   ├── bm25_retriever.py
│   └── reranker.py
│
├── memory/
│   └── conversation_memory.py
│
├── vectorstore/
│   └── faiss_store.py
│
├── app1.py
├── app.py
├── config.py
├── requirements.txt
├── .env
└── .gitignore
```

---

# 🔄 CRAG Workflow

## 1. Load PDF

The system loads the user-provided PDF using `PyPDFLoader`.

```text
PDF
 ↓
Document Pages
```

---

## 2. Split the Document

The PDF content is divided into smaller chunks.

```text
PDF
 ↓
Text Chunks
```

These chunks are used by the retrieval systems.

---

## 3. Initial Retrieval

The first retrieval step uses **FAISS**.

FAISS performs semantic similarity search between the user's question and document chunks.

```text
Question
   ↓
FAISS
   ↓
Retrieved Documents
```

---

## 4. Relevance Evaluation

The retrieved documents are evaluated using a **Cross-Encoder**.

The Cross-Encoder compares:

```text
Question + Retrieved Document
```

and produces a relevance score.

The CRAG evaluator uses a configurable relevance threshold.

Current threshold:

```text
1.0
```

If the best score meets or exceeds the threshold, the retrieved documents are considered relevant.

---

# 🔧 Corrective Retrieval

If the initial documents are not sufficiently relevant, CRAG activates the correction stage.

The correction stage uses both:

### FAISS

For semantic retrieval.

### BM25

For keyword-based retrieval.

```text
Initial Retrieval
       ↓
Relevance Evaluation
       ↓
Not Relevant
       ↓
Correction
   ┌─────────┐
   │  FAISS  │
   └────┬────┘
        │
   ┌────▼────┐
   │  BM25   │
   └────┬────┘
        ↓
Combine Results
        ↓
Remove Duplicates
        ↓
Re-evaluate
```

If the corrected documents are relevant, they are used to generate the answer.

If they are still not relevant, the system falls back to general knowledge.

---

# 🤖 Answer Generation

Google Gemini is used to generate the final answer.

There are two possible sources:

### PDF Answer

When relevant documents are found:

```text
Retrieved Documents
        ↓
Gemini
        ↓
Document-based Answer
```

### General Knowledge Answer

When no relevant documents are found after correction:

```text
No Relevant Documents
        ↓
Gemini
        ↓
General Knowledge Answer
```

The answer generator is instructed not to invent information when answering from retrieved document context.

---

# 🧠 Conversation Memory

CRAG also supports conversation memory.

The system stores recent conversations:

```text
Question
   ↓
Answer
   ↓
Conversation Memory
```

The current implementation keeps up to:

```text
5 conversations
```

---

# 💬 Follow-up Questions

CRAG supports follow-up questions such as:

```text
What are the health effects of air pollution?
```

followed by:

```text
What about children?
```

The question handler combines the previous search context with the follow-up question.

Example:

```text
Original:
What are the health effects of air pollution?

Follow-up:
What about children?

Search Question:
What are the health effects of air pollution?
with specific focus on What about children?
```

This allows the retrieval system to search using the previous context.

---

# 📊 CRAG Components

## `crag_evaluator.py`

Responsible for evaluating retrieved documents.

Main responsibilities:

* Run Cross-Encoder scoring
* Identify the best relevance score
* Compare the score with the relevance threshold
* Decide whether to use or correct the retrieval

---

## `crag_corrector.py`

Responsible for corrective retrieval.

It performs:

```text
FAISS Retrieval
+
BM25 Retrieval
↓
Combine Results
↓
Remove Duplicates
↓
Return Corrected Documents
```

---

## `crag_retriever.py`

Coordinates the complete CRAG retrieval process.

```text
Initial Retrieval
        ↓
Evaluation
        ↓
Relevant?
   ┌────┴────┐
  YES       NO
   ↓         ↓
Use Docs   Correct
             ↓
        Re-evaluate
             ↓
       Relevant?
        ┌────┴────┐
       YES       NO
        ↓         ↓
     Use Docs   No Docs
```

---

## `crag_answer_generator.py`

Responsible for generating answers using Google Gemini.

It supports:

* Document-based answers
* General knowledge answers

---

## `crag_question_handler.py`

Responsible for:

* Question processing
* Follow-up question detection
* Search-question construction
* Conversation memory
* Clearing memory

---

## `crag_pipeline.py`

Connects all CRAG components together.

```text
Question Handler
       ↓
CRAG Retriever
       ↓
CRAG Evaluator
       ↓
CRAG Corrector
       ↓
Answer Generator
       ↓
Conversation Memory
```

---

# 🧪 Testing

The CRAG implementation was tested using a PDF about:

**Air Quality and Health**

Example question:

```text
What are the health effects of air pollution?
```

The system successfully:

1. Retrieved relevant documents.
2. Evaluated their relevance.
3. Used the retrieved documents.
4. Generated a document-based answer.

---

## Follow-up Question Test

Example:

```text
What are the health effects of air pollution?
```

Followed by:

```text
What about children?
```

The system successfully maintained the previous search context and generated a focused search question.

---

## Unrelated Question Test

Example:

```text
What is the capital of France?
```

When the document did not provide relevant information, CRAG attempted corrective retrieval.

When relevant document information was still not found, the system used the general knowledge fallback.

---

# ▶️ Running CRAG

## 1. Open the project

```powershell
cd C:\Users\samyu\OneDrive\Desktop\advanced-rag
```

---

## 2. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Make sure your Gemini API key is configured

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit the `.env` file to GitHub.

---

## 4. Run the CRAG application

For the standalone CRAG application:

```powershell
python app1.py
```

The application will ask for the full PDF path.

Example:

```text
C:\Users\samyu\Downloads\air-quality-and-health.pdf
```

---

# 💻 CRAG Commands

Inside the CRAG application:

```text
history
```

Displays conversation history.

```text
clear
```

Clears conversation memory.

```text
exit
```

Returns to the main menu.

---

# 🔗 Main Application Integration

CRAG is also integrated into the main RAG application.

Run:

```powershell
python app.py
```

Then select:

```text
6. CRAG
```

The main application already loads the PDF and creates the shared:

```text
FAISS Retriever
BM25 Retriever
Cross-Encoder Reranker
```

CRAG reuses these components instead of loading the PDF again.

Architecture:

```text
                    PDF
                     ↓
                Text Chunks
                     ↓
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        FAISS       BM25    Cross-Encoder
          │          │          │
          └──────────┼──────────┘
                     ↓
                    CRAG
                     ↓
                   Gemini
                     ↓
                  Answer
```

---

# 📦 Installation

Install the required dependencies using:

```powershell
pip install -r requirements.txt
```

Important packages include:

```text
faiss-cpu
langchain
langchain-community
langchain-google-genai
langchain-huggingface
langchain-text-splitters
sentence-transformers
pypdf
rank-bm25
python-dotenv
```

---

# 🔐 Environment Variables

The project uses:

```text
GEMINI_API_KEY
```

Store the API key inside `.env`.

The `.env` file is excluded from Git using `.gitignore`.

---

# 🎯 Learning Objectives

This CRAG implementation demonstrates:

* Retrieval-Augmented Generation
* Semantic search
* Keyword search
* Hybrid retrieval concepts
* Cross-Encoder relevance scoring
* Corrective retrieval
* LLM-based answer generation
* Conversation memory
* Follow-up question handling
* Retrieval evaluation
* Fallback mechanisms
* Modular RAG architecture

---

# 🔮 Future Extensions

Possible future improvements include:

* Web search as an external corrective source
* Query rewriting
* Multi-query retrieval
* Better document grading
* Source citations
* Confidence estimation
* LangGraph-based CRAG workflow
* MCP integration
* Agentic RAG
* Additional RAG architectures

---

# 👩‍💻 Author

**Thadikamalla Sai Madhu Samyuktha**

B.Tech – Information Technology

Pragati Engineering College

---

## ⭐ Project

This project is part of an **Advanced RAG learning and implementation project**, progressing from Basic RAG to more advanced retrieval architectures.
