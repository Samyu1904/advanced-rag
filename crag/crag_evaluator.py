class CRAGEvaluator:

    def __init__(
        self,
        reranker,
        relevant_threshold=1.0
    ):
        self.reranker = reranker
        self.relevant_threshold = relevant_threshold

        print(
            "CRAG evaluator initialized successfully!"
        )

    def evaluate(
        self,
        question,
        documents
    ):

        if not documents:

            print(
                "CRAG Evaluation: "
                "No documents retrieved."
            )

            return {
                "is_relevant": False,
                "score": None,
                "action": "correct"
            }

        ranked_documents = (
            self.reranker.rerank(
                question,
                documents,
                top_k=len(documents)
            )
        )

        if not ranked_documents:

            print(
                "CRAG Evaluation: "
                "Unable to evaluate documents."
            )

            return {
                "is_relevant": False,
                "score": None,
                "action": "correct"
            }

        best_document, best_score = (
            ranked_documents[0]
        )

        print(
            f"\nCRAG best relevance score: "
            f"{best_score:.4f}"
        )

        if best_score >= self.relevant_threshold:

            print(
                "CRAG Evaluation: "
                "Retrieved documents are relevant."
            )

            return {
                "is_relevant": True,
                "score": float(best_score),
                "action": "use"
            }

        print(
            "CRAG Evaluation: "
            "Retrieved documents need correction."
        )

        return {
            "is_relevant": False,
            "score": float(best_score),
            "action": "correct"
        }