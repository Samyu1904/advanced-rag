class GraphEvaluator:

    def __init__(self):
        print(
            "Graph RAG evaluator initialized successfully!"
        )

    def evaluate_graph(
        self,
        question,
        graph_context
    ):

        is_relevant = bool(
            graph_context.strip()
        )

        print(
            "\nGraph evaluation: "
            f"{'RELEVANT' if is_relevant else 'NOT_RELEVANT'}"
        )

        return {
            "is_relevant": is_relevant
        }

    def evaluate_answer(
        self,
        question,
        answer,
        graph_context
    ):

        supported = (
            bool(answer.strip())
            and
            bool(graph_context.strip())
        )

        print(
            "\nAnswer evaluation: "
            f"{'SUPPORTED' if supported else 'NOT_SUPPORTED'}"
        )

        return {
            "is_supported": supported
        }