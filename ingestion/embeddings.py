from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully!")

    def embed_documents(self, documents):

        texts = [
            document.page_content
            for document in documents
        ]

        vectors = self.embeddings.embed_documents(texts)

        print("Document embeddings created!")
        print(f"Number of embeddings: {len(vectors)}")
        print(f"Embedding dimension: {len(vectors[0])}")

        return vectors

    def embed_query(self, query):

        vector = self.embeddings.embed_query(query)

        return vector