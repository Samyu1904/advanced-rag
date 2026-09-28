class RerankingRetriever:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        reranker,
        candidate_k=10,
        top_k=3
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k

    def retrieve(self, question):

        print("\nRetrieving candidate documents...")

        # --------------------------------
        # FAISS retrieval
        # --------------------------------

        faiss_results = (
            self.faiss_retriever.retrieve(question)
        )

        faiss_relevant = (
            self.faiss_retriever.is_relevant(
                faiss_results
            )
        )

        if not faiss_relevant:
            print(
                "Question is not relevant "
                "to the document."
            )
            return []

        # --------------------------------
        # BM25 retrieval
        # --------------------------------

        bm25_results = (
            self.bm25_retriever.retrieve(
                question,
                top_k=self.candidate_k
            )
        )

        # --------------------------------
        # Combine FAISS + BM25 documents
        # --------------------------------

        candidate_documents = []

        for document, score in faiss_results:
            candidate_documents.append(document)

        for document, score in bm25_results:
            candidate_documents.append(document)

        # --------------------------------
        # Remove duplicate documents
        # --------------------------------

        unique_documents = []

        seen = set()

        for document in candidate_documents:

            text = document.page_content

            if text not in seen:

                seen.add(text)

                unique_documents.append(
                    document
                )

        print(
            f"Candidate documents: "
            f"{len(unique_documents)}"
        )

        # --------------------------------
        # Reranking
        # --------------------------------

        print("Reranking candidate documents...")

        ranked_documents = self.reranker.rerank(
            question,
            unique_documents,
            top_k=self.top_k
        )

        # --------------------------------
        # Display reranking scores
        # --------------------------------

        print("\nReranked documents:")

        for index, (document, score) in enumerate(
            ranked_documents,
            start=1
        ):
            print(
                f"{index}. Reranker score: "
                f"{score:.4f}"
            )

        return ranked_documents