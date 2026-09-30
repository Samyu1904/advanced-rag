from advanced.multi_query_retriever import MultiQueryRetriever


class AdvancedRetriever:
    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        top_k=3,
        candidate_k=10
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.multi_query_retriever = MultiQueryRetriever()

        self.top_k = top_k
        self.candidate_k = candidate_k

        print(
            "Advanced Retriever initialized successfully!"
        )

    def retrieve(self, question):
        print("\nGenerating multiple queries...")

        queries = (
            self.multi_query_retriever
            .generate_queries(
                question,
                number_of_queries=3
            )
        )

        print("\nGenerated queries:")

        for index, query in enumerate(
            queries,
            start=1
        ):
            print(f"{index}. {query}")

        candidate_documents = []

        print("\nRetrieving documents...")

        for query in queries:

            faiss_results = (
                self.faiss_retriever
                .retrieve(query)
            )

            if self.faiss_retriever.is_relevant(
                faiss_results
            ):
                for document, score in faiss_results:
                    candidate_documents.append(document)

            bm25_results = (
                self.bm25_retriever
                .retrieve(
                    query,
                    top_k=self.candidate_k
                )
            )

            for document, score in bm25_results:
                candidate_documents.append(document)

        unique_documents = []

        seen = set()

        for document in candidate_documents:

            text = document.page_content

            if text not in seen:
                seen.add(text)
                unique_documents.append(document)

        print(
            f"\nUnique candidate documents: "
            f"{len(unique_documents)}"
        )

        final_documents = unique_documents[
            :self.top_k
        ]

        print(
            f"Selected documents: "
            f"{len(final_documents)}"
        )

        return [
            (document, 0)
            for document in final_documents
        ]