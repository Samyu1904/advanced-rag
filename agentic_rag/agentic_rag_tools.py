class AgenticRAGTools:

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        answer_generator
    ):

        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.answer_generator = answer_generator

        print(
            "Agentic RAG tools initialized successfully!"
        )

    # ============================================================
    # FAISS SEARCH TOOL
    # ============================================================

    def search_faiss(self, question):

        print("\n")
        print("=" * 70)
        print("AGENT TOOL: FAISS SEARCH")
        print("=" * 70)

        print(
            f"\nSearching FAISS for:\n{question}"
        )

        results = (
            self.faiss_retriever.retrieve(
                question
            )
        )

        print(
            f"\nFAISS returned "
            f"{len(results)} documents."
        )

        documents = []

        for document, score in results:

            documents.append({
                "document": document,
                "score": float(score)
            })

        return documents

    # ============================================================
    # BM25 SEARCH TOOL
    # ============================================================

    def search_bm25(
        self,
        question,
        top_k=3
    ):

        print("\n")
        print("=" * 70)
        print("AGENT TOOL: BM25 SEARCH")
        print("=" * 70)

        print(
            f"\nSearching BM25 for:\n{question}"
        )

        results = (
            self.bm25_retriever.retrieve(
                question,
                top_k=top_k
            )
        )

        print(
            f"\nBM25 returned "
            f"{len(results)} documents."
        )

        documents = []

        for document, score in results:

            documents.append({
                "document": document,
                "score": float(score)
            })

        return documents

    # ============================================================
    # GENERAL KNOWLEDGE TOOL
    # ============================================================

    def general_knowledge(self, question):

        print("\n")
        print("=" * 70)
        print("AGENT TOOL: GENERAL KNOWLEDGE")
        print("=" * 70)

        print(
            f"\nGenerating general answer for:\n{question}"
        )

        answer = (
            self.answer_generator
            .generate_general_answer(
                question
            )
        )

        return {
            "answer": answer
        }

    # ============================================================
    # DOCUMENT ANSWER TOOL
    # ============================================================

    def generate_document_answer(
        self,
        question,
        documents
    ):

        print("\n")
        print("=" * 70)
        print("AGENT TOOL: DOCUMENT ANSWER")
        print("=" * 70)

        print(
            "\nGenerating answer from "
            "retrieved documents..."
        )

        answer = (
            self.answer_generator
            .generate_document_answer(
                question,
                documents
            )
        )

        return {
            "answer": answer
        }