import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class SelfRAGAnswerGenerator:

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
            "Self-RAG answer generator "
            "initialized successfully!"
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
You are a Self-RAG document question-answering assistant.

Answer the user's question using ONLY the
provided document context.

Rules:

1. Use the document context as the primary source.
2. Do not invent information.
3. Give a clear and concise answer.
4. If the answer cannot be found in the context,
   say that the information could not be found
   in the provided document.
5. Do not mention Self-RAG, FAISS, BM25,
   retrieval, evaluation, or internal system details.

Document Context:

{context}

User Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(response)

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
You are a Self-RAG answer correction assistant.

The previous answer was not sufficiently supported
by the provided document context.

You must generate a new answer that is completely
supported by the document context.

Document Context:

{context}

User Question:

{question}

Previous Answer:

{previous_answer}

Rules:

1. Use ONLY the document context.
2. Do not use outside knowledge.
3. Remove unsupported claims.
4. Do not invent facts.
5. If the answer cannot be determined from
   the context, clearly say so.
6. Give a clear and concise answer.
7. Do not mention Self-RAG or internal system details.

Corrected Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(response)

    def generate_general_answer(
        self,
        question
    ):

        prompt = f"""
You are a helpful AI assistant.

The user's question is not related to the
uploaded document, or relevant information
could not be found in the document.

Answer the question using your general knowledge.

Give a clear and concise answer.

User Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(response)