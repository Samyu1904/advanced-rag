# Advanced RAG – Hybrid Retrieval System

An advanced Retrieval-Augmented Generation (RAG) system built with Python, LangChain, FAISS, BM25, and Gemini.

The project started as a Basic RAG pipeline and was extended to support hybrid retrieval, relevance detection, conversation memory, and follow-up questions.

---

## 🚀 Features

- PDF document ingestion
- Recursive text chunking
- Hugging Face sentence embeddings
- FAISS semantic/vector search
- BM25 keyword-based search
- Hybrid retrieval using FAISS + BM25
- Relevance detection using FAISS similarity scores
- Gemini-powered answer generation
- General knowledge fallback for unrelated questions
- Conversation memory
- Follow-up question handling
- Persistent FAISS vector store
- Runtime PDF selection
- Environment variable based API-key management

---

## 🏗️ Architecture

```text
                         User Question
                              │
                              ▼
                     Question Handler
                              │
                              ▼
                      Hybrid Retriever
                       ┌──────┴──────┐
                       │             │
                       ▼             ▼
                    FAISS          BM25
                  Semantic        Keyword
                   Search         Search
                       │             │
                       └──────┬──────┘
                              │
                              ▼
                     Relevance Check
                       ┌──────┴──────┐
                       │             │
                    Relevant     Not Relevant
                       │             │
                       ▼             ▼
                 PDF Context   General Knowledge
                       │             │
                       └──────┬──────┘
                              ▼
                            Gemini
                              │
                              ▼
                           Answer
                              │
                              ▼
                    Conversation Memory
##🛠️ Tech Stack 
Technology	Purpose
Python	Application development
LangChain	RAG components and LLM integration
FAISS	Semantic vector retrieval
BM25	Keyword-based retrieval
Hugging Face	Text embeddings
Gemini	Answer generation
PyPDF	PDF document loading
LangGraph	Planned/extended workflow support
Git/GitHub	Version control
##📁 Project Structure
advanced-rag/
│
├── ingestion/
│   ├── embeddings.py
│   ├── pdf_loader.py
│   └── text_splitter.py
│
├── retrieval/
│   ├── answer_generator.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── retriever.py
│
├── memory/
│   ├── conversation_memory.py
│   └── question_handler.py
│
├── graph/
│
├── mcp/
│
├── data/
│
├── vectorstore/
│   └── Local FAISS index
│
├── app.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md

The local FAISS vector store and environment files are excluded from Git using .gitignore.

##🔍 How Hybrid Retrieval Works

Traditional semantic retrieval searches for documents based on their meaning.

FAISS performs this type of semantic search using embeddings.

BM25 performs keyword-based retrieval. It is useful when the exact words used in the user's question are important.

This project combines both approaches:

User Query
    │
    ├──────────────► FAISS
    │                  │
    │                  ▼
    │             Semantic Results
    │
    └──────────────► BM25
                       │
                       ▼
                  Keyword Results
                       │
                       ▼
                Combined Results
                       │
                       ▼
                     Gemini

This allows the system to use both semantic similarity and keyword matching.

##🧠 Relevance Detection

The system also checks whether the question is related to the uploaded document.

A configurable FAISS similarity threshold is used.

For example:

Relevant question:
"What are the health effects of air pollution?"

FAISS score ≈ 0.58
Threshold = 1.2

0.58 <= 1.2
→ Relevant
→ Use PDF context

For an unrelated question:

"What is the capital of France?"

FAISS score ≈ 1.75
Threshold = 1.2

1.75 > 1.2
→ Not relevant
→ Use general knowledge

The threshold is empirical and depends on the embedding model and document collection.

##💬 Conversation Memory

The system maintains short-term conversation history.

For example:

User:
What are the health effects of air pollution?

Assistant:
[Answer from PDF]

User:
What about children?

Assistant:
[Answer using the previous conversation context]

The question handler detects follow-up patterns and uses the previous question to resolve incomplete follow-up queries.

##📄 PDF Processing Pipeline

The uploaded PDF goes through the following process:

PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Documents
 │
 ▼
Recursive Text Splitter
 │
 ▼
Chunks
 │
 ▼
Hugging Face Embeddings
 │
 ▼
FAISS Vector Store

BM25 also receives the document chunks to build its keyword index.

##🤖 Answer Generation

Gemini is used as the answer generator.

When the question is relevant to the uploaded PDF:

Question
   +
Retrieved PDF Context
   ↓
Gemini
   ↓
Document-based Answer

When the question is unrelated to the PDF:

Question
   ↓
Relevance Check
   ↓
General Knowledge
   ↓
Gemini
   ↓
General Answer
##⚙️ Setup
1. Clone the repository
git clone https://github.com/Samyu1904/advanced-rag.git
2. Open the project
cd advanced-rag
3. Create a virtual environment
python -m venv .venv
4. Activate the environment

Windows:

.venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt
6. Create .env

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Do not commit the .env file to GitHub.

##▶️ Run the Application

Run:

python app.py

The application asks for the full path of a PDF:

Enter the full path of your PDF:

Example:

C:\Users\YourName\Downloads\air-quality-and-health.pdf

After processing the PDF, you can ask questions interactively.

Type:

exit

to stop the application.

##🧪 Example
Document-related question
Question: What are the health effects of air pollution?

Source: PDF
Follow-up question
Question: What about children?

Source: PDF
Unrelated question
Question: What is the capital of France?

Source: General Knowledge
##🔐 Security

The following files are excluded from Git:

.env
.venv/
vectorstore/
vetorstore/
__pycache__/
*.pyc

The Gemini API key should always be stored in .env and never directly inside Python source code.

##📌 Current Project Stage

Current implementation:

Hybrid RAG

Basic RAG
    ↓
FAISS Semantic Retrieval
    ↓
BM25 Keyword Retrieval
    ↓
Hybrid Retrieval
    ↓
Relevance Detection
    ↓
Conversation Memory
    ↓
Gemini Generation

Future improvements can include:

Score normalization and fusion
Query transformation
Reranking
Multi-query retrieval
Parent-child retrieval
Self-RAG
Corrective RAG
Agentic RAG
Multimodal RAG
Graph RAG
##👩‍💻 Author

Thadikamalla Sai Madhu Samyuktha

GitHub:
https://github.com/Samyu1904


### Step 2

Save the file.

Then in Git Bash, run:

```bash
git status

You should see:

README.md
