import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class SelfRAGEvaluator:

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
            "Self-RAG evaluator initialized successfully!"
        )

    def extract_text(self, response):

        if isinstance(response.content, str):
            return response.content.strip()

        return "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        ).strip()

    def evaluate_retrieved_documents(
        self,
        question,
        documents
    ):

        if not documents:

            print(
                "Self-RAG retrieval evaluation: "
                "No documents found."
            )

            return {
                "is_relevant": False,
                "reason": "No documents were retrieved."
            }

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are a Self-RAG retrieval evaluator.

Your task is to determine whether the retrieved
documents contain information that is useful for
answering the user's question.

User Question:

{question}

Retrieved Documents:

{context}

Rules:

1. If the documents contain useful information
   for answering the question, return:
   RELEVANT

2. If the documents do not contain useful
   information, return:
   NOT_RELEVANT

3. Return exactly one of these two values.
4. Do not provide explanations.
5. Do not answer the question.

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
            "\nSelf-RAG retrieval evaluation:"
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
You are a Self-RAG answer evaluator.

Determine whether the generated answer is
supported by the provided document context.

User Question:

{question}

Document Context:

{context}

Generated Answer:

{answer}

Rules:

1. If the answer is supported by the document
   context, return:
   SUPPORTED

2. If the answer contains information that is
   not supported by the document context, return:
   NOT_SUPPORTED

3. Return exactly one of these two values.
4. Do not provide explanations.
5. Do not rewrite the answer.

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
            "\nSelf-RAG answer evaluation:"
        )

        if is_supported:

            print(
                "Answer is supported by the documents."
            )

        else:

            print(
                "Answer is not fully supported "
                "by the documents."
            )

        return {
            "is_supported": is_supported,
            "reason": decision
        }