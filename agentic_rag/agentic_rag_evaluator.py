import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class AgenticRAGEvaluator:

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
            "Agentic RAG evaluator "
            "initialized successfully!"
        )

    # ============================================================
    # TEXT EXTRACTION
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
    # DOCUMENT RELEVANCE
    # ============================================================

    def evaluate_documents(
        self,
        question,
        documents
    ):

        if not documents:

            print(
                "\nAgentic evaluation: "
                "No documents available."
            )

            return {
                "is_relevant": False,
                "reason": "No documents available."
            }

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are evaluating document relevance.

User Question:

{question}

Retrieved Document Context:

{context}

Determine whether the documents contain
useful information for answering the question.

Return exactly one value:

RELEVANT

or

NOT_RELEVANT

Do not provide explanations.

Decision:
"""

        response = self.llm.invoke(prompt)

        decision = self.extract_text(
            response
        ).upper()

        is_relevant = (
            "RELEVANT" in decision
            and "NOT_RELEVANT" not in decision
        )

        print(
            "\nAgentic document evaluation:"
        )

        if is_relevant:

            print(
                "Documents are relevant."
            )

        else:

            print(
                "Documents are not relevant."
            )

        return {
            "is_relevant": is_relevant,
            "reason": decision
        }

    # ============================================================
    # ANSWER SUPPORT EVALUATION
    # ============================================================

    def evaluate_answer(
        self,
        question,
        answer,
        documents
    ):

        if not documents:

            return {
                "is_supported": False,
                "reason": "No documents available."
            }

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are evaluating whether an answer is
supported by retrieved documents.

User Question:

{question}

Document Context:

{context}

Generated Answer:

{answer}

Return exactly one value:

SUPPORTED

or

NOT_SUPPORTED

The answer is SUPPORTED only if its claims
are supported by the document context.

Do not provide explanations.

Decision:
"""

        response = self.llm.invoke(prompt)

        decision = self.extract_text(
            response
        ).upper()

        is_supported = (
            "SUPPORTED" in decision
            and "NOT_SUPPORTED" not in decision
        )

        print(
            "\nAgentic answer evaluation:"
        )

        if is_supported:

            print(
                "Answer is supported."
            )

        else:

            print(
                "Answer is not supported."
            )

        return {
            "is_supported": is_supported,
            "reason": decision
        }