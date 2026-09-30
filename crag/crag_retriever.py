from crag.crag_evaluator import CRAGEvaluator
from crag.crag_corrector import CRAGCorrector


class CRAGRetriever:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        reranker,
        top_k=3,
        relevance_threshold=1.0
    ):
        self.faiss_retriever = faiss_retriever

        self.evaluator = CRAGEvaluator(
            reranker=reranker,
            relevant_threshold=relevance_threshold
        )

        self.corrector = CRAGCorrector(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever
        )

        self.top_k = top_k

        print(
            "CRAG retriever initialized successfully!"
        )

    def retrieve(self, question):

        print("\n")
        print("=" * 70)
        print("CRAG RETRIEVAL")
        print("=" * 70)

        print("\nInitial retrieval...")

        initial_results = (
            self.faiss_retriever.retrieve(
                question
            )
        )

        initial_documents = [
            document
            for document, score in initial_results
        ]

        print(
            f"Initial documents retrieved: "
            f"{len(initial_documents)}"
        )

        evaluation = self.evaluator.evaluate(
            question,
            initial_documents
        )

        if evaluation["action"] == "use":

            print("\nCRAG decision: USE RETRIEVED DOCUMENTS")

            return initial_documents[:self.top_k]

        print("\nCRAG decision: CORRECT RETRIEVAL")

        corrected_documents = (
            self.corrector.correct(
                question,
                top_k=self.top_k
            )
        )

        if not corrected_documents:

            print(
                "\nCRAG correction did not find "
                "useful documents."
            )

            return []

        print(
            "\nEvaluating corrected documents..."
        )

        corrected_evaluation = (
            self.evaluator.evaluate(
                question,
                corrected_documents
            )
        )

        if corrected_evaluation["is_relevant"]:

            print(
                "\nCRAG decision: "
                "CORRECTED DOCUMENTS ARE RELEVANT"
            )

            return corrected_documents[
                :self.top_k
            ]

        print(
            "\nCRAG decision: "
            "NO RELEVANT DOCUMENTS FOUND"
        )

        return []