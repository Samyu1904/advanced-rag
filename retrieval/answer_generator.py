import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class AnswerGenerator:

    def __init__(self):

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found in the .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0
        )

        print("Gemini answer generator initialized successfully!")

    def extract_text(self, response):

        if isinstance(response.content, str):
            return response.content

        return "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    def generate_pdf_answer(self, question, results):

        context = "\n\n".join(
            document[0].page_content
            for document  in results
        )

        prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY
the information provided in the document.

Do not use outside knowledge.

If the answer cannot be found in the
provided context, say:

"I could not find the answer in the
provided document."

Document Context:

{context}

Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(response)

    def generate_general_answer(self, question):

        prompt = f"""
You are a helpful AI assistant.

The user's question is not related to
the uploaded document.

Answer the question using your
general knowledge.

Question:

{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return self.extract_text(response)