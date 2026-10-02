import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class GraphAnswerGenerator:
    def __init__(self):

        load_dotenv()

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

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
            "Graph RAG answer generator initialized successfully!"
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

    def generate_graph_answer(
        self,
        question,
        graph_context
    ):

        prompt = f"""
You are a Graph RAG answer generator.

Answer the user's question using ONLY
the knowledge graph relationships below.

Knowledge Graph:

{graph_context}

User Question:

{question}

Rules:

1. Use only information contained in
   the knowledge graph.

2. Do not invent facts.

3. Do not add outside knowledge.

4. If the graph does not contain enough
   information, say:

"I could not find enough information
in the knowledge graph."

5. Give a clear and concise answer.

Answer:
"""

        try:

            response = self.llm.invoke(prompt)

            return self.extract_text(response)

        except Exception as error:

            error_text = str(error)

            if "429" in error_text:
                return (
                    "The Gemini API quota has been "
                    "temporarily exhausted. "
                    "Please wait and try again."
                )

            return (
                "The graph answer could not be "
                "generated because of an API error."
            )

    def generate_general_answer(
        self,
        question
    ):

        prompt = f"""
You are a helpful AI assistant.

The user's question cannot be answered
using the uploaded document's knowledge graph.

Answer the question using general knowledge.

Question:

{question}

Answer:
"""

        try:

            response = self.llm.invoke(prompt)

            return self.extract_text(response)

        except Exception as error:

            if "429" in str(error):
                return (
                    "The Gemini API quota has been "
                    "temporarily exhausted. "
                    "Please wait and try again."
                )

            return (
                "The general answer could not be "
                "generated because of an API error."
            )