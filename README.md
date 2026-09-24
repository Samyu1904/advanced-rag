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
```

---

## 🧠 How Hybrid Retrieval Works

This project combines two different retrieval techniques:

### 1. FAISS Semantic Search

FAISS is used for semantic/vector-based retrieval.

The PDF text is converted into vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

FAISS then searches for document chunks that are semantically similar to the user's question.

This helps when the question uses different words but has a similar meaning to the document content.

---

### 2. BM25 Keyword Search

BM25 is used for keyword-based retrieval.

It looks for important words from the user's question inside the document chunks.

For example:

```text
Question:
health effects of air pollution
```

BM25 can directly match keywords such as:

```text
health
effects
air
pollution
```

This provides a different retrieval signal from semantic search.

---

### 3. Hybrid Retrieval

The system combines results from:

```text
FAISS
+
BM25
```

Duplicate document chunks are removed.

The final set of document chunks is then passed to the answer-generation stage.

The current implementation gives FAISS results priority when combining the two retrieval outputs.

---

## 🎯 Relevance Detection

The system uses the FAISS similarity score to determine whether a question is related to the uploaded PDF.

The current threshold is:

```text
1.2
```

The threshold is configurable and was selected empirically for this project.

### Example: Relevant Question

```text
Question:
What are the health effects of air pollution?
```

Example FAISS score:

```text
0.58
```

Since:

```text
0.58 <= 1.2
```

the question is considered relevant.

The system therefore uses:

```text
PDF Context
```

to generate the answer.

---

### Example: Unrelated Question

```text
Question:
What is the capital of France?
```

Example FAISS score:

```text
1.75
```

Since:

```text
1.75 > 1.2
```

the question is considered unrelated to the PDF.

The system therefore uses:

```text
General Knowledge
```

to answer the question.

---

## 💬 Conversation Memory

The system maintains short-term conversation history.

The current configuration stores the last:

```text
5 conversations
```

Each conversation contains:

```text
Question
Answer
```

This allows the system to understand follow-up questions.

For example:

```text
User:
What are the health effects of air pollution?

Assistant:
Air pollution can affect respiratory and cardiovascular health...
```

Then the user can ask:

```text
What about children?
```

The system recognizes this as a follow-up question and uses the previous conversation context to resolve it.

---

## 🔄 Follow-Up Question Handling

The project contains a question handler that detects incomplete questions.

Examples include:

```text
What about children?
How about this?
What are its effects?
Tell me more.
More about this.
```

If the user asks an incomplete question after a previous question, the system attempts to resolve the follow-up using the previous question stored in conversation memory.

Example:

```text
Previous question:
What are the health effects of air pollution?

Follow-up:
What about children?
```

The system converts the follow-up into a more complete question before performing retrieval.

This improves retrieval because the retriever receives a question containing the required context.

---

## 📄 PDF Processing Pipeline

The PDF processing pipeline contains three major stages.

### Step 1 – PDF Loading

The system uses LangChain's:

```text
PyPDFLoader
```

to load the PDF document.

The loader extracts the text page by page.

---

### Step 2 – Text Splitting

The extracted text is divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Current configuration:

```text
Chunk Size: 1000
Chunk Overlap: 200
```

The overlap helps preserve context between neighboring chunks.

---

### Step 3 – Embeddings

Each document chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These vectors are then stored in FAISS for semantic retrieval.

---

## 🗄️ Persistent Vector Store

The project saves the FAISS index locally so that the vector store does not need to be recreated every time the application starts.

The vector store is stored in:

```text
vectorstore/current_index
```

The application calculates a SHA-256 hash of the uploaded PDF.

If the same PDF is used again, the existing FAISS index can be loaded instead of recreating the embeddings.

If a different PDF is provided, the application creates a new vector store.

Metadata is stored in:

```text
vectorstore/current_index/metadata.json
```

The metadata contains:

```text
PDF path
PDF hash
```

---

## 🤖 Answer Generation

Gemini is used as the Large Language Model for generating answers.

The project uses:

```text
gemini-3.6-flash
```

The system has two answer-generation paths.

### PDF-Based Answer

When the question is relevant to the uploaded document:

```text
User Question
      ↓
Hybrid Retrieval
      ↓
Relevant Document Chunks
      ↓
Gemini
      ↓
Answer
```

The Gemini prompt instructs the model to use only the retrieved document context.

---

### General Knowledge Answer

When the question is not related to the uploaded document:

```text
User Question
      ↓
Relevance Check
      ↓
Not Related to PDF
      ↓
Gemini General Knowledge
      ↓
Answer
```

This allows the application to continue answering general questions instead of returning an unnecessary "not found" response.

---

## 🔐 Security

The Gemini API key is not stored directly in the Python source code.

Instead, it is stored in an environment file:

```text
.env
```

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git using `.gitignore`.

The API key should never be committed to GitHub.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| LangChain | RAG framework and document processing |
| LangGraph | Graph-based workflow support |
| FAISS | Vector similarity search |
| BM25 | Keyword-based retrieval |
| Hugging Face | Text embeddings |
| Sentence Transformers | Embedding model |
| Gemini | LLM / answer generation |
| PyPDF | PDF text extraction |
| python-dotenv | Environment variable management |

---

## 📁 Project Structure

```text
advanced-rag/
│
├── data/
│   └── documents/
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   └── embeddings.py
│
├── retrieval/
│   ├── retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   └── answer_generator.py
│
├── memory/
│   ├── conversation_memory.py
│   └── question_handler.py
│
├── graph/
│
├── mcp/
│
├── vectorstore/
│   └── current_index/
│
├── app.py
├── config.py
├── requirements.txt
├── .env
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Samyu1904/advanced-rag.git
```

Move into the project directory:

```bash
cd advanced-rag
```

---

### 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

### 4. Configure Gemini API Key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace:

```text
your_api_key_here
```

with your Gemini API key.

Do not commit the `.env` file to GitHub.

---

## ▶️ Run the Application

Start the application using:

```bash
python app.py
```

The application will ask for the full path of a PDF.

Example:

```text
Enter the full path of your PDF:
C:\Users\Samyu\Documents\student-handbook.pdf
```

The application then loads and processes the PDF.

---

## 🧪 Example Usage

### Question Related to the PDF

```text
Question:
What are the health effects of air pollution?
```

The application performs:

```text
Question
   ↓
FAISS Search
   ↓
BM25 Search
   ↓
Relevance Check
   ↓
PDF Context
   ↓
Gemini
   ↓
Answer
```

Output:

```text
Source: PDF

Answer:
Air pollution can have several effects on human health,
including respiratory and cardiovascular problems...
```

---

### Follow-Up Question

```text
Question:
What about children?
```

The system uses the previous conversation to understand the question.

Output:

```text
Source: PDF
```

The answer is generated using the relevant document context.

---

### Unrelated Question

```text
Question:
What is the capital of France?
```

The relevance check identifies that the question is not related to the uploaded PDF.

Output:

```text
Source: General Knowledge
```

Gemini then generates the general answer.

---

## 📊 Current Project Stage

The project has evolved through multiple stages:

```text
Basic RAG
   ↓
FAISS Vector Search
   ↓
LangChain Integration
   ↓
LangGraph Concepts
   ↓
Conversation Memory
   ↓
Follow-Up Question Handling
   ↓
Relevance Detection
   ↓
General Knowledge Fallback
   ↓
BM25 Keyword Retrieval
   ↓
Hybrid RAG
```

The current implementation focuses on combining semantic retrieval and keyword retrieval while maintaining conversational context.

---

## 🚀 Future Improvements

Possible future improvements include:

- Score normalization between FAISS and BM25
- More advanced result fusion
- Reranking retrieved documents
- Better query rewriting
- Long-term memory
- LangGraph-based workflow orchestration
- MCP tool integration
- Streaming responses
- Multiple document support
- Document management interface
- Web-based user interface
- Evaluation metrics for retrieval quality
- Automated RAG evaluation
- Better citation and source tracking

---

## 👩‍💻 Author

**Thadikamalla Sai Madhu Samyuktha**

GitHub:

https://github.com/Samyu1904
