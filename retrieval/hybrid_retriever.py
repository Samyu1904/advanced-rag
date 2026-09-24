class HybridRetriever:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        top_k=3
    ):

        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

    def retrieve(self, question):

        # -----------------------------
        # 1. FAISS semantic retrieval
        # -----------------------------

        faiss_results = (
            self.faiss_retriever.retrieve(
                question
            )
        )

        # -----------------------------
        # 2. Check FAISS relevance
        # -----------------------------

        faiss_relevant = (
            self.faiss_retriever.is_relevant(
                faiss_results
            )
        )

        # -----------------------------
        # 3. BM25 keyword retrieval
        # -----------------------------

        bm25_results = (
            self.bm25_retriever.retrieve(
                question,
                top_k=self.top_k
            )
        )

        # -----------------------------
        # 4. If FAISS says the question
        #    is not related to the PDF,
        #    return no results
        # -----------------------------

        if not faiss_relevant:

            return []

        # -----------------------------
        # 5. Combine FAISS + BM25
        # -----------------------------

        combined_documents = []

        for document, score in faiss_results:

            combined_documents.append(
                document
            )

        for document, score in bm25_results:

            combined_documents.append(
                document
            )

        # -----------------------------
        # 6. Remove duplicate chunks
        # -----------------------------

        unique_documents = []

        seen = set()

        for document in combined_documents:

            text = document.page_content

            if text not in seen:

                seen.add(text)

                unique_documents.append(
                    document
                )

        # -----------------------------
        # 7. Select final results
        # -----------------------------

        final_results = unique_documents[
            :self.top_k
        ]

        # -----------------------------
        # 8. Return document + score
        # -----------------------------

        return [
            (document, 0)
            for document in final_results
        ]