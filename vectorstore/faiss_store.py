from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class FAISSVectorStore:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

        self.vector_store = None

    def create(self, documents):

        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )

        print("FAISS vector store created successfully!")

        return self.vector_store

    def save(self, path="vectorstore/faiss_index"):

        if self.vector_store is None:
            raise ValueError(
                "Vector store has not been created yet."
            )

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_store.save_local(path)

        print(f"FAISS vector store saved to: {path}")

    def load(self, path="vectorstore/faiss_index"):

        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"FAISS vector store loaded from: {path}")

        return self.vector_store