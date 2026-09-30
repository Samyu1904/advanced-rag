import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class CRAGAnswerGenerator:

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
            "CRAG Gemini answer generator "
            "initialized successfully!"
        )

    def extract_text(self, response):

        if isinstance(response.content, str):
            return response.content

        return "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

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
You are a helpful document question-answering assistant.

Answer the user's question using the provided
document context.

Rules:

1. Use the document context as the primary source.
2. Do not invent information.
3. Give a clear and concise answer.
4. If the answer cannot be found in the context,
   say that the information could not be found
   in the provided document.
5. Do not mention FAISS, BM25, CRAG, reranking,
   retrieval, or internal system details.

Document Context:

{context}

User Question:

{question}

Answer:
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
uploaded document, or the CRAG system could
not find relevant information in the document.

Answer the question using your general knowledge.

Give a clear and concise answer.

User Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(response)