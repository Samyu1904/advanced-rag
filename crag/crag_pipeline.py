from crag.crag_retriever import CRAGRetriever
from crag.crag_answer_generator import CRAGAnswerGenerator
from crag.crag_question_handler import CRAGQuestionHandler


class CRAGPipeline:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        reranker
    ):

        self.retriever = CRAGRetriever(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            top_k=3,
            relevance_threshold=1.0
        )

        self.answer_generator = (
            CRAGAnswerGenerator()
        )

        self.question_handler = (
            CRAGQuestionHandler(
                max_history=5
            )
        )

        print(
            "CRAG pipeline initialized successfully!"
        )

    def ask(self, question):

        print("\n")
        print("=" * 70)
        print("CRAG PIPELINE")
        print("=" * 70)

        print("\nUser Question:")
        print(question)

        processed = (
            self.question_handler
            .process_question(question)
        )

        if processed["needs_clarification"]:

            print("\n")
            print("=" * 70)
            print("CLARIFICATION REQUIRED")
            print("=" * 70)

            answer = (
                "Could you please provide more "
                "context for your question?"
            )

            print(answer)

            return {
                "question": question,
                "search_question": processed[
                    "search_question"
                ],
                "documents": [],
                "answer": answer,
                "source": "Clarification",
                "needs_clarification": True
            }

        search_question = processed[
            "search_question"
        ]

        print("\nSearch Question:")
        print(search_question)

        print("\nRunning CRAG retrieval...")

        documents = self.retriever.retrieve(
            search_question
        )

        if documents:

            print("\n")
            print("=" * 70)
            print("CRAG RESULT")
            print("=" * 70)

            print(
                f"\nRelevant documents found: "
                f"{len(documents)}"
            )

            print(
                "\nGenerating answer from "
                "retrieved documents..."
            )

            answer = (
                self.answer_generator
                .generate_document_answer(
                    question=question,
                    documents=documents
                )
            )

            source = "PDF"

        else:

            print("\n")
            print("=" * 70)
            print("CRAG RESULT")
            print("=" * 70)

            print(
                "\nNo relevant documents found "
                "after correction."
            )

            print(
                "\nGenerating general answer..."
            )

            answer = (
                self.answer_generator
                .generate_general_answer(
                    question
                )
            )

            source = "General Knowledge"

        self.question_handler.add_to_memory(
            question,
            answer
        )

        print("\n")
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(answer)

        print("\nSource:")
        print(source)

        return {
            "question": question,
            "search_question": search_question,
            "documents": documents,
            "answer": answer,
            "source": source,
            "needs_clarification": False
        }

    def get_history(self):

        return self.question_handler.get_history()

    def clear_memory(self):

        self.question_handler.clear_memory()

    def show_history(self):

        history = self.get_history()

        print("\n")
        print("=" * 70)
        print("CRAG CONVERSATION HISTORY")
        print("=" * 70)

        if not history:

            print("\nNo conversation history.")

            return

        for index, item in enumerate(
            history,
            start=1
        ):

            print(
                f"\nConversation {index}"
            )

            print("-" * 70)

            print(
                "Question:"
            )

            print(
                item["question"]
            )

            print(
                "\nAnswer:"
            )

            print(
                item["answer"]
            )