import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class MultiQueryRetriever:
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
            "Multi-Query Gemini model "
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

    def generate_queries(self, question, number_of_queries=3):
        prompt = f"""
You are a search query generation assistant.

The user has asked the following question:

{question}

Generate {number_of_queries} different search queries
that can be used to retrieve relevant information
from a document.

Each query should express the same information need
using different wording or perspectives.

Rules:

1. Generate exactly {number_of_queries} queries.
2. Each query must be on a separate line.
3. Do not number the queries.
4. Do not use bullet points.
5. Do not provide explanations.
6. Do not answer the user's question.
7. Return only the search queries.

Original question:

{question}
"""

        response = self.llm.invoke(prompt)

        text = self.extract_text(response)

        queries = []

        for line in text.splitlines():
            query = line.strip()

            if not query:
                continue

            if query.startswith("-"):
                query = query[1:].strip()

            if query:
                queries.append(query)

        queries = queries[:number_of_queries]

        if question not in queries:
            queries.insert(0, question)

        return queries[:number_of_queries]