from graph_rag.graph_retriever import GraphRetriever
from graph_rag.graph_answer_generator import GraphAnswerGenerator
from graph_rag.graph_question_handler import GraphQuestionHandler


class GraphRAGPipeline:

    def __init__(
        self,
        graph,
        top_k=10
    ):

        self.retriever = GraphRetriever(
            graph=graph,
            top_k=top_k
        )

        self.answer_generator = (
            GraphAnswerGenerator()
        )

        self.question_handler = (
            GraphQuestionHandler(
                max_history=5
            )
        )

        print(
            "Graph RAG pipeline initialized successfully!"
        )

    def ask(self, question):

        print("\n")
        print("=" * 70)
        print("GRAPH RAG PIPELINE")
        print("=" * 70)

        print(
            f"\nUser Question:\n{question}"
        )

        processed = (
            self.question_handler
            .process_question(question)
        )

        if processed["needs_clarification"]:

            answer = (
                "Could you please provide more "
                "context for your question?"
            )

            print("\n" + answer)

            return {
                "question": question,
                "search_question": "",
                "graph_results": [],
                "answer": answer,
                "source": "Clarification",
                "needs_clarification": True
            }

        search_question = (
            processed["search_question"]
        )

        print(
            f"\nGraph Search Question:\n"
            f"{search_question}"
        )

        graph_results = (
            self.retriever.retrieve(
                search_question
            )
        )

        if not graph_results:

            print("\n")
            print("=" * 70)
            print("GRAPH RETRIEVAL FAILED")
            print("=" * 70)

            print(
                "\nUsing general knowledge..."
            )

            answer = (
                self.answer_generator
                .generate_general_answer(
                    question
                )
            )

            self.question_handler.add_to_memory(
                question,
                answer
            )

            print(
                "\nSource: General Knowledge"
            )

            print(
                "\nAnswer:\n" + answer
            )

            return {
                "question": question,
                "search_question": search_question,
                "graph_results": [],
                "answer": answer,
                "source": "General Knowledge",
                "needs_clarification": False
            }

        graph_context = (
            self.retriever.format_context(
                graph_results
            )
        )

        print("\n")
        print("=" * 70)
        print("GRAPH CONTEXT")
        print("=" * 70)

        print(
            "\n" + graph_context
        )

        print("\n")
        print("=" * 70)
        print("GRAPH INFORMATION FOUND")
        print("=" * 70)

        answer = (
            self.answer_generator
            .generate_graph_answer(
                question,
                graph_context
            )
        )

        self.question_handler.add_to_memory(
            question,
            answer
        )

        print("\n")
        print("=" * 70)
        print("GRAPH RAG FINAL RESULT")
        print("=" * 70)

        print(
            "\nSource: Knowledge Graph"
        )

        print(
            "\nAnswer:\n" + answer
        )

        return {
            "question": question,
            "search_question": search_question,
            "graph_results": graph_results,
            "answer": answer,
            "source": "Knowledge Graph",
            "needs_clarification": False
        }

    def get_history(self):

        return (
            self.question_handler
            .get_history()
        )

    def clear_memory(self):

        self.question_handler.clear_memory()

    def show_history(self):

        history = self.get_history()

        print("\n")
        print("=" * 70)
        print("GRAPH RAG CONVERSATION HISTORY")
        print("=" * 70)

        if not history:

            print(
                "\nNo conversation history."
            )

            return

        for index, item in enumerate(
            history,
            start=1
        ):

            print(
                f"\nConversation {index}"
            )

            print("-" * 70)

            print("Question:")

            print(
                item["question"]
            )

            print("\nAnswer:")

            print(
                item["answer"]
            )