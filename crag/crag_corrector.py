class CRAGCorrector:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever

        print(
            "CRAG corrector initialized successfully!"
        )

    def correct(
        self,
        question,
        top_k=3
    ):
        print("\nStarting CRAG correction...")

        print("\nRunning FAISS correction retrieval...")

        faiss_results = (
            self.faiss_retriever.retrieve(
                question
            )
        )

        print(
            f"FAISS returned "
            f"{len(faiss_results)} documents."
        )

        print("\nRunning BM25 correction retrieval...")

        bm25_results = (
            self.bm25_retriever.retrieve(
                question,
                top_k=top_k
            )
        )

        print(
            f"BM25 returned "
            f"{len(bm25_results)} documents."
        )

        candidate_documents = []

        for document, score in faiss_results:
            candidate_documents.append(document)

        for document, score in bm25_results:
            candidate_documents.append(document)

        unique_documents = []

        seen = set()

        for document in candidate_documents:

            text = document.page_content

            if text not in seen:

                seen.add(text)

                unique_documents.append(
                    document
                )

        corrected_documents = (
            unique_documents[:top_k]
        )

        print(
            f"\nCRAG correction produced "
            f"{len(corrected_documents)} "
            f"unique documents."
        )

        return corrected_documents