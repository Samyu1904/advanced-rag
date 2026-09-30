from advanced.advanced_retriever import AdvancedRetriever


class AdvancedRerankingRetriever:
    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        reranker,
        top_k=3,
        candidate_k=10
    ):
        self.advanced_retriever = AdvancedRetriever(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever,
            top_k=candidate_k,
            candidate_k=candidate_k
        )

        self.reranker = reranker
        self.top_k = top_k

        print(
            "Advanced Reranking Retriever "
            "initialized successfully!"
        )

    def retrieve(self, question):
        print("\nStarting Advanced RAG retrieval...")

        candidate_results = (
            self.advanced_retriever
            .retrieve(question)
        )

        candidate_documents = [
            document
            for document, score in candidate_results
        ]

        if not candidate_documents:
            print("No candidate documents found.")
            return []

        print(
            f"\nReranking "
            f"{len(candidate_documents)} documents..."
        )

        ranked_documents = self.reranker.rerank(
            question,
            candidate_documents,
            top_k=self.top_k
        )

        print("\nFinal reranked documents:")

        for index, (document, score) in enumerate(
            ranked_documents,
            start=1
        ):
            print(
                f"{index}. "
                f"Reranker score: {score:.4f}"
            )

        return ranked_documents