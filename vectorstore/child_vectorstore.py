from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class ChildVectorStore:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

        self.vector_store = None

        print(
            "Child vector store initialized successfully!"
        )

    def create(self, children):

        self.vector_store = FAISS.from_documents(
            children,
            self.embeddings
        )

        print(
            "Child FAISS vector store created successfully!"
        )

        print(
            f"Number of child chunks: {len(children)}"
        )

        return self.vector_store

    def search(self, question, top_k=3):

        if self.vector_store is None:

            raise ValueError(
                "Child vector store has not been created yet."
            )

        results = (
            self.vector_store.similarity_search_with_score(
                question,
                k=top_k
            )
        )

        return results