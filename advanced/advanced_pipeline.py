from advanced.advanced_question_handler import (
    AdvancedQuestionHandler
)

from advanced.advanced_reranking_retriever import (
    AdvancedRerankingRetriever
)

from advanced.advanced_answer_generator import (
    AdvancedAnswerGenerator
)


class AdvancedRAGPipeline:

    def __init__(
        self,
        faiss_retriever,
        advanced_retriever,
        answer_generator
    ):

        self.faiss_retriever = faiss_retriever
        self.advanced_retriever = advanced_retriever
        self.answer_generator = answer_generator

        self.question_handler = (
            AdvancedQuestionHandler()
        )

        print(
            "Advanced RAG pipeline "
            "initialized successfully!"
        )

    def ask(self, question):

        print("\n")
        print("=" * 70)
        print("USER QUESTION")
        print("=" * 70)

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
                "needs_clarification": True,
                "documents": [],
                "answer": answer
            }

        search_question = processed[
            "search_question"
        ]

        print("\nSearch question:")
        print(search_question)

        print("\nChecking document relevance...")

        relevance_results = (
            self.faiss_retriever
            .retrieve(search_question)
        )

        is_relevant = (
            self.faiss_retriever
            .is_relevant(relevance_results)
        )

        if not is_relevant:

            print("\nQuestion is not related to the document.")

            answer = (
                self.answer_generator
                .generate_general_answer(
                    question
                )
            )

            print("\n")
            print("=" * 70)
            print("GENERAL ANSWER")
            print("=" * 70)

            print(answer)

            self.question_handler.add_to_memory(
                question,
                answer
            )

            return {
                "question": question,
                "search_question": search_question,
                "needs_clarification": False,
                "documents": [],
                "answer": answer
            }

        print("\nQuestion is relevant to the document.")

        print("\nRunning Advanced retrieval...")

        results = (
            self.advanced_retriever
            .retrieve(search_question)
        )

        if not results:

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

            return {
                "question": question,
                "search_question": search_question,
                "needs_clarification": False,
                "documents": [],
                "answer": answer
            }

        documents = [
            document
            for document, score in results
        ]

        print("\n")
        print("=" * 70)
        print("FINAL RERANKED DOCUMENTS")
        print("=" * 70)

        for index, (document, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"\nDocument {index}"
            )

            print(
                f"Reranker score: "
                f"{score:.4f}"
            )

            print("-" * 70)

            print(
                document.page_content[:500]
            )

        print("\nGenerating final answer...")

        answer = (
            self.answer_generator
            .generate_document_answer(
                question=question,
                documents=documents
            )
        )

        print("\n")
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(answer)

        self.question_handler.add_to_memory(
            question,
            answer
        )

        return {
            "question": question,
            "search_question": search_question,
            "needs_clarification": False,
            "documents": documents,
            "answer": answer
        }

    def clear_memory(self):

        self.question_handler.clear_memory()

        print(
            "Advanced RAG conversation "
            "memory cleared."
        )