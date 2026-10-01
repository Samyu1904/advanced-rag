from self_rag.self_rag_retriever import SelfRAGRetriever
from self_rag.self_rag_evaluator import SelfRAGEvaluator
from self_rag.self_rag_answer_generator import (
    SelfRAGAnswerGenerator
)
from self_rag.self_rag_question_handler import (
    SelfRAGQuestionHandler
)


class SelfRAGPipeline:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        top_k=3
    ):

        self.retriever = SelfRAGRetriever(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever,
            top_k=top_k
        )

        self.evaluator = SelfRAGEvaluator()

        self.answer_generator = (
            SelfRAGAnswerGenerator()
        )

        self.question_handler = (
            SelfRAGQuestionHandler(
                max_history=5
            )
        )

        print(
            "Self-RAG pipeline initialized successfully!"
        )

    def ask(self, question):

        print("\n")
        print("=" * 70)
        print("SELF-RAG PIPELINE")
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
                "needs_clarification": True,
                "answer_supported": False
            }

        search_question = processed[
            "search_question"
        ]

        print("\nSearch Question:")
        print(search_question)

        print("\nRunning Self-RAG retrieval...")

        retrieval_result = (
            self.retriever.retrieve(
                search_question
            )
        )

        documents = retrieval_result[
            "documents"
        ]

        retrieval_relevant = retrieval_result[
            "is_relevant"
        ]

        # ====================================================
        # NOT RELEVANT -> GENERAL KNOWLEDGE
        # ====================================================

        if not retrieval_relevant:

            print("\n")
            print("=" * 70)
            print("SELF-RAG RETRIEVAL DECISION")
            print("=" * 70)

            print(
                "\nRetrieved documents were not "
                "considered relevant."
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

            print("\n")
            print("=" * 70)
            print("GENERAL KNOWLEDGE ANSWER")
            print("=" * 70)

            print(answer)

            print("\nSource:")
            print("General Knowledge")

            self.question_handler.add_to_memory(
                question,
                answer
            )

            return {
                "question": question,
                "search_question": search_question,
                "documents": [],
                "answer": answer,
                "source": "General Knowledge",
                "needs_clarification": False,
                "answer_supported": False,
                "regeneration_attempted": False
            }

        # ====================================================
        # RELEVANT DOCUMENTS
        # ====================================================

        print("\n")
        print("=" * 70)
        print("SELF-RAG GENERATION")
        print("=" * 70)

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

        print("\nGenerated Answer:")
        print(answer)

        # ====================================================
        # ANSWER EVALUATION
        # ====================================================

        print("\nEvaluating generated answer...")

        answer_evaluation = (
            self.evaluator.evaluate_answer(
                question=question,
                answer=answer,
                documents=documents
            )
        )

        answer_supported = (
            answer_evaluation["is_supported"]
        )

        regeneration_attempted = False

        # ====================================================
        # REGENERATE IF NOT SUPPORTED
        # ====================================================

        if not answer_supported:

            print("\n")
            print("=" * 70)
            print("SELF-RAG REFLECTION")
            print("=" * 70)

            print(
                "\nThe generated answer was not "
                "fully supported."
            )

            print(
                "\nRegenerating a corrected answer..."
            )

            regeneration_attempted = True

            answer = (
                self.answer_generator
                .regenerate_answer(
                    question=question,
                    documents=documents,
                    previous_answer=answer
                )
            )

            print("\nCorrected Answer:")
            print(answer)

            print(
                "\nEvaluating corrected answer..."
            )

            corrected_evaluation = (
                self.evaluator.evaluate_answer(
                    question=question,
                    answer=answer,
                    documents=documents
                )
            )

            answer_supported = (
                corrected_evaluation[
                    "is_supported"
                ]
            )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        print("\n")
        print("=" * 70)
        print("SELF-RAG FINAL RESULT")
        print("=" * 70)

        if answer_supported:

            print(
                "\nFinal answer is supported "
                "by the retrieved documents."
            )

        else:

            print(
                "\nFinal answer could not be "
                "verified against the retrieved documents."
            )

        print("\nAnswer:")
        print(answer)

        print("\nSource:")
        print("PDF")

        self.question_handler.add_to_memory(
            question,
            answer
        )

        return {
            "question": question,
            "search_question": search_question,
            "documents": documents,
            "answer": answer,
            "source": "PDF",
            "needs_clarification": False,
            "answer_supported": answer_supported,
            "regeneration_attempted":
                regeneration_attempted
        }

    def get_history(self):

        return self.question_handler.get_history()

    def clear_memory(self):

        self.question_handler.clear_memory()

    def show_history(self):

        history = self.get_history()

        print("\n")
        print("=" * 70)
        print("SELF-RAG CONVERSATION HISTORY")
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

            print("Question:")

            print(
                item["question"]
            )

            print("\nAnswer:")

            print(
                item["answer"]
            )