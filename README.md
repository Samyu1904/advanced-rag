# 🤖 Agentic RAG — Intelligent Retrieval & Decision-Making System

> **An intelligent Retrieval-Augmented Generation system that dynamically decides how to search, evaluate, answer, retry, and fall back to general knowledge.**

---

## 🌟 Overview

**Agentic RAG (Retrieval-Augmented Generation)** extends traditional RAG by introducing an **AI decision-making agent** into the retrieval and generation process.

Instead of following a fixed pipeline such as:

```text
Question → Retrieve → Generate Answer
```

Agentic RAG dynamically decides what action should be taken based on the available information.

The system can:

* 🔎 Search using **FAISS**
* 🔤 Search using **BM25**
* 🧠 Decide which retrieval strategy to use
* 📄 Evaluate retrieved documents
* ✍️ Generate answers from documents
* ✅ Evaluate whether answers are supported
* 🔄 Regenerate answers when necessary
* 🌐 Fall back to general knowledge
* 💬 Understand follow-up questions
* 🧠 Maintain conversation history
* 🔁 Retry when retrieval or generation is insufficient
* 🛑 Stop when a reliable answer is obtained

---

# 🧠 What Makes It Agentic?

Traditional RAG generally follows a predefined workflow:

```text
User Question
      ↓
Retrieve Documents
      ↓
Generate Answer
      ↓
Final Answer
```

Agentic RAG introduces **decision-making and reflection**:

```text
                    ┌──────────────────┐
                    │   User Question  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Question Handler │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Initial FAISS    │
                    │ Retrieval        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Decision Agent  │
                    └────────┬─────────┘
                             ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
         FAISS Search    BM25 Search   General Knowledge
              ↓              ↓
              └───────┬──────┘
                      ↓
              ┌───────────────┐
              │ Document      │
              │ Evaluation    │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Answer        │
              │ Generation    │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Answer        │
              │ Evaluation    │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Agent         │
              │ Reflection    │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ Final Answer  │
              └───────────────┘
```

The important difference is that the system **does not blindly follow one retrieval path**.

The agent observes the retrieved information and decides what should happen next.

---

# ✨ Key Features

| Feature                | Description                                     |
| ---------------------- | ----------------------------------------------- |
| 🔎 FAISS Retrieval     | Semantic vector-based document search           |
| 🔤 BM25 Retrieval      | Keyword-based document search                   |
| 🧠 Decision Agent      | Dynamically selects the next action             |
| 📄 Document Evaluation | Checks whether retrieved documents are relevant |
| ✍️ Answer Generation   | Generates answers using retrieved context       |
| ✅ Answer Evaluation    | Checks whether the answer is supported          |
| 🔄 Regeneration        | Generates a better answer when necessary        |
| 🌐 General Knowledge   | Handles questions outside the document          |
| 💬 Follow-up Questions | Supports conversational questions               |
| 🧠 Conversation Memory | Stores recent conversations                     |
| 🔁 Iterative Reasoning | Allows multiple agent iterations                |
| 🛡️ Fallback Mechanism | Prevents the system from getting stuck          |

---

# 🏗️ Architecture

The Agentic RAG implementation is organized into independent components.

```text
agentic_rag/
│
├── agentic_rag_tools.py
│
├── agentic_rag_agent.py
│
├── agentic_rag_evaluator.py
│
├── agentic_rag_answer_generator.py
│
├── agentic_rag_question_handler.py
│
└── agentic_rag_pipeline.py
```

---

# 📦 Components

## 1. `agentic_rag_tools.py`

Contains the tools available to the Agentic RAG system.

### Tools

#### 🔎 FAISS Search

Performs semantic similarity search over the uploaded document.

```text
Question
   ↓
Embedding
   ↓
FAISS
   ↓
Relevant Documents
```

#### 🔤 BM25 Search

Performs keyword-based retrieval.

```text
Question
   ↓
Token Matching
   ↓
BM25
   ↓
Relevant Documents
```

#### 🌐 General Knowledge

Uses Gemini to answer questions that cannot be answered using the uploaded document.

#### 📄 Document Answer

Generates an answer using the retrieved document context.

---

# 2. `agentic_rag_agent.py`

This is the **decision-making component** of the system.

The agent decides what should happen next.

### Initial Decision

Possible actions:

```text
FAISS_SEARCH
GENERAL_KNOWLEDGE
```

The agent receives the initial FAISS results and determines whether the uploaded document contains useful information.

---

### Retrieval Decision

After retrieval, the agent can choose:

```text
USE_DOCUMENTS
SEARCH_BM25
GENERAL_KNOWLEDGE
```

This allows the system to dynamically move between retrieval strategies.

---

### Reflection Decision

After generating an answer, the agent evaluates what should happen next:

```text
FINAL
RETRY
GENERAL_KNOWLEDGE
```

This gives the system an iterative reasoning loop.

---

# 3. `agentic_rag_evaluator.py`

The evaluator checks the quality of retrieved information and generated answers.

### Document Evaluation

The evaluator determines whether retrieved documents are:

```text
RELEVANT
```

or

```text
NOT_RELEVANT
```

---

### Answer Evaluation

The generated answer is evaluated as:

```text
SUPPORTED
```

or

```text
NOT_SUPPORTED
```

This helps prevent unsupported answers from being immediately returned to the user.

---

# 4. `agentic_rag_answer_generator.py`

Responsible for generating answers.

It supports:

### 📄 Document-Based Answers

Uses retrieved document context.

### 🌐 General Knowledge Answers

Uses Gemini's general knowledge when the uploaded document cannot answer the question.

### 🔄 Regeneration

If the generated answer is not sufficiently supported, the system can regenerate the answer using the retrieved context.

---

# 5. `agentic_rag_question_handler.py`

Handles conversational behavior.

It provides:

* Follow-up question detection
* Search-question creation
* Conversation memory
* History management
* Memory clearing

For example:

```text
User:
What are the health effects of air pollution?

Agent:
Provides answer.

User:
What about children?
```

The system understands that the second question is related to the previous question.

The search can therefore be expanded into a contextual query such as:

```text
What are the health effects of air pollution?
with specific focus on children
```

---

# 6. `agentic_rag_pipeline.py`

This is the **main orchestration layer**.

It connects:

```text
Question Handler
        ↓
FAISS
        ↓
Decision Agent
        ↓
BM25 / Documents / General Knowledge
        ↓
Document Evaluation
        ↓
Answer Generation
        ↓
Answer Evaluation
        ↓
Agent Reflection
        ↓
Final Answer
```

The pipeline also controls the maximum number of iterations.

---

# 🔄 Agentic Workflow

The complete workflow is:

### Step 1 — User Question

The user provides a question.

```text
What are the health effects of air pollution?
```

---

### Step 2 — Question Processing

The question handler determines whether the question is:

* A new question
* A follow-up question
* A question requiring clarification

---

### Step 3 — Initial FAISS Retrieval

The system performs an initial semantic search.

```text
User Question
      ↓
FAISS
      ↓
Top Documents
```

---

### Step 4 — Agent Decision

The agent examines the retrieved context.

It can decide:

```text
FAISS_SEARCH
```

or

```text
GENERAL_KNOWLEDGE
```

---

### Step 5 — Document Evaluation

Retrieved documents are evaluated.

```text
Relevant?
   │
   ├── YES → Continue
   │
   └── NO  → Try another strategy
```

---

### Step 6 — Retrieval Strategy

The agent can decide to:

```text
USE_DOCUMENTS
```

or:

```text
SEARCH_BM25
```

or:

```text
GENERAL_KNOWLEDGE
```

---

### Step 7 — Answer Generation

If useful documents are found, the system generates an answer using those documents.

---

### Step 8 — Answer Evaluation

The generated answer is checked.

```text
Answer Supported?
       │
       ├── YES → Continue
       │
       └── NO  → Regenerate / Retry
```

---

### Step 9 — Agent Reflection

The agent reflects on the generated answer.

Possible decisions:

```text
FINAL
RETRY
GENERAL_KNOWLEDGE
```

---

### Step 10 — Final Answer

The system returns the answer along with information about the source and number of iterations.

---

# 🧩 Decision Flow

```text
                     USER QUESTION
                           │
                           ↓
                    QUESTION HANDLER
                           │
                           ↓
                    INITIAL FAISS
                      RETRIEVAL
                           │
                           ↓
                   ┌───────────────┐
                   │ DECISION AGENT│
                   └───────┬───────┘
                           │
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
        FAISS SEARCH    BM25 SEARCH   GENERAL
             │             │          KNOWLEDGE
             ↓             ↓
       DOCUMENT EVALUATION
             │
             ↓
       ANSWER GENERATION
             │
             ↓
        ANSWER EVALUATION
             │
             ↓
         REFLECTION
             │
       ┌─────┼─────┐
       ↓     ↓     ↓
     FINAL RETRY GENERAL
             │
             ↓
       FINAL ANSWER
```

---

# 💾 Conversation Memory

Agentic RAG maintains recent conversation history.

Example:

```text
Conversation 1
Question:
What are the health effects of air pollution?

Answer:
...

Conversation 2
Question:
What about children?

Answer:
...
```

The system supports:

```text
history
```

to display previous conversations.

It also supports:

```text
clear
```

to clear the conversation memory.

---

# 🧪 Testing

The Agentic RAG implementation was tested using:

```text
air-quality-and-health.pdf
```

The following scenarios were tested.

### ✅ 1. Normal Document Question

Example:

```text
What are the health effects of air pollution?
```

Expected behavior:

```text
FAISS Retrieval
      ↓
Agent Decision
      ↓
Document Evaluation
      ↓
Document Answer
      ↓
Answer Evaluation
      ↓
FINAL
```

---

### ✅ 2. Follow-Up Question

Example:

```text
What about children?
```

The system uses the previous question as conversational context.

---

### ✅ 3. Chained Follow-Up

The system was tested with multiple contextual follow-up questions.

This verifies that conversation context is maintained across turns.

---

### ✅ 4. Unrelated Question

Example:

```text
What is the capital of France?
```

Since the answer is not expected to come from the uploaded document, the system can use:

```text
GENERAL_KNOWLEDGE
```

---

### ✅ 5. Conversation History

Command:

```text
history
```

Displays previous conversations.

---

### ✅ 6. Clear Memory

Command:

```text
clear
```

Clears the conversation memory.

---

### ✅ 7. Question After Clearing Memory

The system was tested with a new question after clearing the previous conversation.

---

### ✅ 8. New Document Question

A fresh document-related question was tested after clearing memory.

---

### ✅ 9. Exit

Command:

```text
exit
```

Cleanly exits the Agentic RAG application.

---

# 🖥️ Running Agentic RAG

From the project root:

```text
C:\Users\samyu\OneDrive\Desktop\advanced-rag
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run the standalone Agentic RAG application:

```powershell
python app1.py
```

Or run the complete RAG application:

```powershell
python app.py
```

Then select:

```text
8. Agentic RAG
```

---

# ⚙️ Technologies Used

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| 🐍 Python        | Core programming language       |
| 🧠 Gemini        | Decision making & generation    |
| 🔎 FAISS         | Semantic vector retrieval       |
| 🔤 BM25          | Keyword retrieval               |
| 🤗 HuggingFace   | Embedding model                 |
| 🦜 LangChain     | RAG components                  |
| 📄 PyPDF         | PDF document processing         |
| 🔐 python-dotenv | Environment variable management |

---

# 🤖 LLM

The project uses:

```text
Gemini 3.6 Flash
```

for:

* Agent decisions
* Document evaluation
* Answer evaluation
* Answer generation
* General knowledge responses
* Answer regeneration

---

# 🔐 Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Do **not** commit the `.env` file to GitHub.

The project uses `.gitignore` to protect sensitive environment variables.

---

# 📁 Project Integration

Agentic RAG is integrated into the main RAG application alongside:

```text
1. Basic RAG
2. Hybrid RAG
3. Parent-Child RAG + LangGraph
4. Reranking RAG
5. Advanced RAG
6. CRAG
7. Self-RAG
8. Agentic RAG
9. Exit
```

This allows different RAG architectures to be tested from a single application.

---

# 🆚 Traditional RAG vs Agentic RAG

| Traditional RAG                  | Agentic RAG                            |
| -------------------------------- | -------------------------------------- |
| Fixed workflow                   | Dynamic workflow                       |
| Fixed retrieval strategy         | Multiple retrieval strategies          |
| Retrieve → Generate              | Retrieve → Decide → Evaluate → Reflect |
| Limited self-correction          | Supports retries                       |
| Basic retrieval                  | FAISS + BM25                           |
| Static decisions                 | Agent-based decisions                  |
| Limited evaluation               | Document + answer evaluation           |
| Simple fallback                  | Intelligent fallback                   |
| Limited conversational reasoning | Follow-up handling + memory            |

---

# 🎯 Why Agentic RAG?

Agentic RAG is useful when a RAG application needs more than simple retrieval.

It provides a framework where the system can:

```text
OBSERVE
   ↓
DECIDE
   ↓
ACT
   ↓
EVALUATE
   ↓
REFLECT
   ↓
RETRY / FINISH
```

This makes the retrieval process more adaptive and allows the system to respond differently depending on the question and retrieved information.

---

# 🚀 Future Improvements

Potential future extensions include:

* 🔗 LangGraph-based agent orchestration
* 🛠️ More specialized tools
* 🌐 Web search integration
* 📊 Retrieval confidence scoring
* 🧠 Long-term memory
* 🔀 More retrieval strategies
* 👥 Multi-agent RAG
* 📈 Observability and tracing
* ⚡ Retrieval caching
* 🧪 Automated evaluation benchmarks

---

# 📚 Learning Outcomes

Through this implementation, the project demonstrates practical understanding of:

* Retrieval-Augmented Generation
* Vector databases
* Semantic search
* Keyword search
* Hybrid retrieval
* LLM-based decision making
* Tool-based agents
* Document evaluation
* Answer evaluation
* Reflection
* Regeneration
* Conversational memory
* Follow-up question handling
* Agentic workflows
* Intelligent fallback mechanisms

---

# 👩‍💻 Author

## Thadikamalla Sai Madhu Samyuktha

🎓 **B.Tech — Information Technology**

💡 **AI | Machine Learning | Generative AI | RAG | Agentic AI**

📍 India

📧 **Email:** [samyukthatadikamalla2005@gmail.com](mailto:samyukthatadikamalla2005@gmail.com)

---

> ⭐ **Agentic RAG — Moving from fixed retrieval pipelines to intelligent, decision-driven RAG systems.**
