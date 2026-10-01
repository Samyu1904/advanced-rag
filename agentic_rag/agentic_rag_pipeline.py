from agentic_rag.agentic_rag_agent import (
    AgenticRAGAgent
)

from agentic_rag.agentic_rag_tools import (
    AgenticRAGTools
)

from agentic_rag.agentic_rag_evaluator import (
    AgenticRAGEvaluator
)

from agentic_rag.agentic_rag_answer_generator import (
    AgenticRAGAnswerGenerator
)

from agentic_rag.agentic_rag_question_handler import (
    AgenticRAGQuestionHandler
)


class AgenticRAGPipeline:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        top_k=3,
        max_iterations=3
    ):

        self.answer_generator = (
            AgenticRAGAnswerGenerator()
        )

        self.tools = AgenticRAGTools(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever,
            answer_generator=self.answer_generator
        )

        self.agent = AgenticRAGAgent()

        self.evaluator = AgenticRAGEvaluator()

        self.question_handler = (
            AgenticRAGQuestionHandler(
                max_history=5
            )
        )

        self.top_k = top_k
        self.max_iterations = max_iterations

        print(
            "Agentic RAG pipeline "
            "initialized successfully!"
        )

    def ask(self, question):

        print("\n")
        print("=" * 70)
        print("AGENTIC RAG PIPELINE")
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
                "search_question":
                    processed["search_question"],
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

        # ========================================================
        # INITIAL FAISS RETRIEVAL
        # ========================================================

        print("\nRunning initial FAISS retrieval...")

        initial_faiss_results = (
            self.tools.search_faiss(
                search_question
            )
        )

        print(
            "\nGiving retrieved information "
            "to the Agent..."
        )

        # ========================================================
        # INITIAL AGENT DECISION
        # ========================================================

        action = (
            self.agent
            .decide_initial_action(
                search_question,
                initial_faiss_results
            )
        )

        # ========================================================
        # GENERAL KNOWLEDGE
        # ========================================================

        if action == "GENERAL_KNOWLEDGE":

            result = (
                self.tools
                .general_knowledge(
                    question
                )
            )

            answer = result["answer"]

            print("\n")
            print("=" * 70)
            print("AGENTIC RAG FINAL RESULT")
            print("=" * 70)

            print("\nSource: General Knowledge")

            print("\nAnswer:")
            print(answer)

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
                "iterations": 1
            }

        # ========================================================
        # AGENTIC RETRIEVAL LOOP
        # ========================================================

        documents = []
        answer = ""
        source = "PDF"
        iterations = 0

        while iterations < self.max_iterations:

            iterations += 1

            print("\n")
            print("=" * 70)
            print(
                f"AGENT ITERATION {iterations}"
            )
            print("=" * 70)

            # ====================================================
            # FAISS
            # ====================================================

            if action == "FAISS_SEARCH":

                faiss_results = (
                    initial_faiss_results
                    if iterations == 1
                    else self.tools.search_faiss(
                        search_question
                    )
                )

                faiss_documents = [
                    item["document"]
                    for item in faiss_results
                ]

                faiss_evaluation = (
                    self.evaluator
                    .evaluate_documents(
                        search_question,
                        faiss_documents
                    )
                )

                if faiss_evaluation[
                    "is_relevant"
                ]:

                    bm25_results = []

                    action = (
                        self.agent
                        .decide_after_retrieval(
                            search_question,
                            faiss_results,
                            bm25_results
                        )
                    )

                    if action == "USE_DOCUMENTS":

                        documents = (
                            faiss_documents[
                                :self.top_k
                            ]
                        )

                    elif action == "SEARCH_BM25":

                        continue

                    else:

                        continue

                else:

                    print(
                        "\nFAISS documents were "
                        "not relevant."
                    )

                    action = "SEARCH_BM25"

                    continue

            # ====================================================
            # BM25
            # ====================================================

            if action == "SEARCH_BM25":

                bm25_results = (
                    self.tools.search_bm25(
                        search_question,
                        top_k=self.top_k
                    )
                )

                bm25_documents = [
                    item["document"]
                    for item in bm25_results
                ]

                bm25_evaluation = (
                    self.evaluator
                    .evaluate_documents(
                        search_question,
                        bm25_documents
                    )
                )

                if bm25_evaluation[
                    "is_relevant"
                ]:

                    documents = (
                        bm25_documents[
                            :self.top_k
                        ]
                    )

                    action = "USE_DOCUMENTS"

                else:

                    print(
                        "\nBM25 documents were "
                        "not relevant."
                    )

                    action = "GENERAL_KNOWLEDGE"

                    continue

            # ====================================================
            # USE DOCUMENTS
            # ====================================================

            if action == "USE_DOCUMENTS":

                if not documents:

                    action = "SEARCH_BM25"

                    continue

                print("\n")
                print("=" * 70)
                print(
                    "AGENT DECISION: USE DOCUMENTS"
                )
                print("=" * 70)

                result = (
                    self.tools
                    .generate_document_answer(
                        question,
                        documents
                    )
                )

                answer = result["answer"]

                print("\nGenerated Answer:")
                print(answer)

                answer_evaluation = (
                    self.evaluator
                    .evaluate_answer(
                        question,
                        answer,
                        documents
                    )
                )

                if not answer_evaluation[
                    "is_supported"
                ]:

                    print(
                        "\nAnswer was not "
                        "sufficiently supported."
                    )

                    answer = (
                        self.answer_generator
                        .regenerate_answer(
                            question,
                            documents,
                            answer
                        )
                    )

                    print(
                        "\nRegenerated Answer:"
                    )

                    print(answer)

                    retry_evaluation = (
                        self.evaluator
                        .evaluate_answer(
                            question,
                            answer,
                            documents
                        )
                    )

                    if retry_evaluation[
                        "is_supported"
                    ]:

                        source = "PDF"

                        break

                    action = "SEARCH_BM25"

                    continue

                reflection = (
                    self.agent
                    .decide_after_answer(
                        question,
                        answer,
                        documents
                    )
                )

                if reflection == "FINAL":

                    source = "PDF"

                    break

                if reflection == "RETRY":

                    answer = (
                        self.answer_generator
                        .regenerate_answer(
                            question,
                            documents,
                            answer
                        )
                    )

                    print(
                        "\nRegenerated Answer:"
                    )

                    print(answer)

                    retry_evaluation = (
                        self.evaluator
                        .evaluate_answer(
                            question,
                            answer,
                            documents
                        )
                    )

                    if retry_evaluation[
                        "is_supported"
                    ]:

                        source = "PDF"

                        break

                    action = "SEARCH_BM25"

                    continue

                if reflection == "GENERAL_KNOWLEDGE":

                    action = "GENERAL_KNOWLEDGE"

                    continue

            # ====================================================
            # GENERAL KNOWLEDGE
            # ====================================================

            if action == "GENERAL_KNOWLEDGE":

                result = (
                    self.tools
                    .general_knowledge(
                        question
                    )
                )

                answer = result["answer"]

                source = "General Knowledge"

                break

        # ========================================================
        # MAXIMUM ITERATIONS
        # ========================================================

        if not answer:

            print("\n")
            print("=" * 70)
            print("AGENT MAXIMUM ITERATIONS REACHED")
            print("=" * 70)

            result = (
                self.tools
                .general_knowledge(
                    question
                )
            )

            answer = result["answer"]

            source = "General Knowledge"

        # ========================================================
        # MEMORY
        # ========================================================

        self.question_handler.add_to_memory(
            question,
            answer
        )

        # ========================================================
        # FINAL RESULT
        # ========================================================

        print("\n")
        print("=" * 70)
        print("AGENTIC RAG FINAL RESULT")
        print("=" * 70)

        print(
            f"\nIterations: {iterations}"
        )

        print(
            f"\nSource: {source}"
        )

        print("\nAnswer:")
        print(answer)

        return {
            "question": question,
            "search_question": search_question,
            "documents": documents,
            "answer": answer,
            "source": source,
            "iterations": iterations,
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
        print("AGENTIC RAG CONVERSATION HISTORY")
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