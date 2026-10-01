from self_rag.self_rag_evaluator import SelfRAGEvaluator


class SelfRAGRetriever:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        top_k=3
    ):

        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever

        self.evaluator = SelfRAGEvaluator()

        self.top_k = top_k

        print(
            "Self-RAG retriever initialized successfully!"
        )

    def retrieve(self, question):

        print("\n")
        print("=" * 70)
        print("SELF-RAG RETRIEVAL")
        print("=" * 70)

        print("\nRunning FAISS retrieval...")

        faiss_results = (
            self.faiss_retriever.retrieve(
                question
            )
        )

        print(
            f"FAISS returned "
            f"{len(faiss_results)} documents."
        )

        print("\nRunning BM25 retrieval...")

        bm25_results = (
            self.bm25_retriever.retrieve(
                question,
                top_k=self.top_k
            )
        )

        print(
            f"BM25 returned "
            f"{len(bm25_results)} documents."
        )

        candidate_documents = []

        for document, score in faiss_results:

            candidate_documents.append(
                document
            )

        for document, score in bm25_results:

            candidate_documents.append(
                document
            )

        unique_documents = []

        seen = set()

        for document in candidate_documents:

            text = document.page_content

            if text not in seen:

                seen.add(text)

                unique_documents.append(
                    document
                )

        documents = unique_documents[
            :self.top_k
        ]

        print(
            f"\nUnique documents selected: "
            f"{len(documents)}"
        )

        evaluation = (
            self.evaluator
            .evaluate_retrieved_documents(
                question,
                documents
            )
        )

        if not evaluation["is_relevant"]:

            print(
                "\nSelf-RAG decision: "
                "RETRIEVED DOCUMENTS ARE NOT RELEVANT"
            )

            return {
                "documents": [],
                "is_relevant": False,
                "evaluation": evaluation
            }

        print(
            "\nSelf-RAG decision: "
            "RETRIEVED DOCUMENTS ARE RELEVANT"
        )

        return {
            "documents": documents,
            "is_relevant": True,
            "evaluation": evaluation
        }