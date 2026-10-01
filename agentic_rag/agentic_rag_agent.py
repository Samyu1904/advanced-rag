import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class AgenticRAGAgent:

    def __init__(self):

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found "
                "in the .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0
        )

        print(
            "Agentic RAG decision agent "
            "initialized successfully!"
        )

    # ============================================================
    # RESPONSE TEXT EXTRACTION
    # ============================================================

    def extract_text(self, response):

        if isinstance(response.content, str):

            return response.content.strip()

        return "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        ).strip()

    # ============================================================
    # INITIAL DECISION
    # ============================================================

    def decide_initial_action(
        self,
        question,
        faiss_documents
    ):

        context = "\n\n".join(
            item["document"].page_content
            for item in faiss_documents
        )

        prompt = f"""
You are the decision-making agent in an
Agentic RAG system.

The system has an uploaded document.

User Question:

{question}

Initial FAISS Retrieved Document Context:

{context}

Your task is to decide what the agent should
do next.

Available actions:

FAISS_SEARCH
GENERAL_KNOWLEDGE

Rules:

1. If the retrieved document context contains
   information that can help answer the user's
   question, choose FAISS_SEARCH.

2. If the retrieved document context clearly
   does not contain useful information for the
   question, choose GENERAL_KNOWLEDGE.

3. Prefer the uploaded document when it contains
   relevant information.

4. Return exactly one action.

5. Do not provide explanations.

Action:
"""

        response = self.llm.invoke(prompt)

        action = self.extract_text(
            response
        ).upper()

        if "FAISS_SEARCH" in action:

            selected_action = "FAISS_SEARCH"

        else:

            selected_action = "GENERAL_KNOWLEDGE"

        print(
            f"\nAgent initial decision: "
            f"{selected_action}"
        )

        return selected_action

    # ============================================================
    # RETRIEVAL DECISION
    # ============================================================

    def decide_after_retrieval(
        self,
        question,
        faiss_documents,
        bm25_documents
    ):

        faiss_context = "\n\n".join(
            item["document"].page_content
            for item in faiss_documents
        )

        bm25_context = "\n\n".join(
            item["document"].page_content
            for item in bm25_documents
        )

        prompt = f"""
You are the decision-making agent in an
Agentic RAG system.

User Question:

{question}

FAISS Retrieved Documents:

{faiss_context}

BM25 Retrieved Documents:

{bm25_context}

Decide what the system should do next.

Available actions:

USE_DOCUMENTS
SEARCH_BM25
GENERAL_KNOWLEDGE

Rules:

1. Choose USE_DOCUMENTS if the retrieved
   information can answer the question.

2. Choose SEARCH_BM25 if FAISS retrieval
   is insufficient but BM25 may find useful
   keyword-based information.

3. Choose GENERAL_KNOWLEDGE if the retrieved
   information does not contain the answer.

4. Return exactly one action.

5. Do not provide explanations.

Action:
"""

        response = self.llm.invoke(prompt)

        action = self.extract_text(
            response
        ).upper()

        if "USE_DOCUMENTS" in action:

            selected_action = "USE_DOCUMENTS"

        elif "SEARCH_BM25" in action:

            selected_action = "SEARCH_BM25"

        else:

            selected_action = "GENERAL_KNOWLEDGE"

        print(
            f"\nAgent retrieval decision: "
            f"{selected_action}"
        )

        return selected_action

    # ============================================================
    # ANSWER REFLECTION DECISION
    # ============================================================

    def decide_after_answer(
        self,
        question,
        answer,
        documents
    ):

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are the reflection component of an
Agentic RAG system.

User Question:

{question}

Retrieved Document Context:

{context}

Generated Answer:

{answer}

Determine what the agent should do next.

Available actions:

FINAL
RETRY
GENERAL_KNOWLEDGE

Rules:

1. Choose FINAL if the answer is supported
   by the retrieved document context.

2. Choose RETRY if the answer is not fully
   supported but the retrieved documents
   contain enough information to generate
   a better answer.

3. Choose GENERAL_KNOWLEDGE if the retrieved
   documents cannot answer the question.

4. Return exactly one action.

5. Do not provide explanations.

Action:
"""

        response = self.llm.invoke(prompt)

        action = self.extract_text(
            response
        ).upper()

        if "GENERAL_KNOWLEDGE" in action:

            selected_action = "GENERAL_KNOWLEDGE"

        elif "RETRY" in action:

            selected_action = "RETRY"

        else:

            selected_action = "FINAL"

        print(
            f"\nAgent reflection decision: "
            f"{selected_action}"
        )

        return selected_action