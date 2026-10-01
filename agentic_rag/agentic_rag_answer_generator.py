import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class AgenticRAGAnswerGenerator:

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
            "Agentic RAG answer generator "
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
    # DOCUMENT ANSWER
    # ============================================================

    def generate_document_answer(
        self,
        question,
        documents
    ):

        if not documents:

            return (
                "I could not find relevant information "
                "in the provided document."
            )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are an Agentic RAG document
question-answering assistant.

Answer the user's question using ONLY
the provided document context.

Rules:

1. Use the document context as the source.
2. Do not invent information.
3. Do not use outside knowledge.
4. Give a clear and concise answer.
5. If the answer cannot be found in the
   context, say so clearly.
6. Do not mention Agentic RAG, agents,
   FAISS, BM25, retrieval, or evaluation.

Document Context:

{context}

User Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(
            response
        )

    # ============================================================
    # GENERAL KNOWLEDGE ANSWER
    # ============================================================

    def generate_general_answer(
        self,
        question
    ):

        prompt = f"""
You are a helpful AI assistant.

The user's question cannot be answered
using the uploaded document.

Answer the question using your general knowledge.

Give a clear and concise answer.

Do not mention Agentic RAG or internal
system details.

User Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(
            response
        )

    # ============================================================
    # RETRY ANSWER
    # ============================================================

    def regenerate_answer(
        self,
        question,
        documents,
        previous_answer
    ):

        if not documents:

            return (
                "I could not find relevant information "
                "in the provided document."
            )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are correcting a document-based answer.

User Question:

{question}

Document Context:

{context}

Previous Answer:

{previous_answer}

Generate a new answer using ONLY the
document context.

Rules:

1. Remove unsupported claims.
2. Do not use outside knowledge.
3. Do not invent information.
4. Keep the answer concise.
5. If the document does not contain
   enough information, clearly say so.
6. Do not mention agents or internal
   system details.

Corrected Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(
            response
        )