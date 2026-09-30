
from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import DocumentReranker

from vectorstore.faiss_store import FAISSVectorStore

from advanced.advanced_reranking_retriever import (
    AdvancedRerankingRetriever
)

from advanced.advanced_answer_generator import (
    AdvancedAnswerGenerator
)

from advanced.advanced_pipeline import (
    AdvancedRAGPipeline
)


class AdvancedRAGApplication:

    def __init__(self):

        self.documents = None
        self.chunks = None

        self.vector_store = None

        self.faiss_retriever = None
        self.bm25_retriever = None
        self.reranker = None

        self.advanced_retriever = None
        self.answer_generator = None
        self.pipeline = None

    def setup(self):

        print("=" * 70)
        print("ADVANCED RAG")
        print("=" * 70)

        # -----------------------------------------------------
        # STEP 1 - DYNAMIC PDF IMPORT
        # -----------------------------------------------------

        print("\nEnter the PDF file path.")

        pdf_path = input(
            "\nPDF path: "
        ).strip()

        print("\nLoading PDF...")

        loader = PDFLoader()

        self.documents = loader.load_pdf(
            pdf_path
        )

        # -----------------------------------------------------
        # STEP 2 - TEXT CHUNKING
        # -----------------------------------------------------

        print("\nSplitting document...")

        splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.chunks = splitter.split_documents(
            self.documents
        )

        # -----------------------------------------------------
        # STEP 3 - FAISS VECTOR STORE
        # -----------------------------------------------------

        print("\nCreating FAISS vector store...")

        faiss_store = FAISSVectorStore()

        self.vector_store = faiss_store.create(
            self.chunks
        )

        self.faiss_retriever = RAGRetriever(
            vector_store=self.vector_store,
            top_k=3,
            threshold=1.2
        )

        # -----------------------------------------------------
        # STEP 4 - BM25
        # -----------------------------------------------------

        print("\nCreating BM25 retriever...")

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        # -----------------------------------------------------
        # STEP 5 - CROSS-ENCODER RERANKER
        # -----------------------------------------------------

        print("\nLoading Cross-Encoder reranker...")

        self.reranker = DocumentReranker()

        # -----------------------------------------------------
        # STEP 6 - ADVANCED RETRIEVER
        # -----------------------------------------------------

        print(
            "\nCreating Advanced Reranking Retriever..."
        )

        self.advanced_retriever = (
            AdvancedRerankingRetriever(
                faiss_retriever=self.faiss_retriever,
                bm25_retriever=self.bm25_retriever,
                reranker=self.reranker,
                top_k=3,
                candidate_k=10
            )
        )

        # -----------------------------------------------------
        # STEP 7 - ANSWER GENERATOR
        # -----------------------------------------------------

        print(
            "\nCreating Advanced Answer Generator..."
        )

        self.answer_generator = (
            AdvancedAnswerGenerator()
        )

        # -----------------------------------------------------
        # STEP 8 - COMPLETE PIPELINE
        # -----------------------------------------------------

        print(
            "\nCreating Advanced RAG Pipeline..."
        )

        self.pipeline = AdvancedRAGPipeline(
            faiss_retriever=self.faiss_retriever,
            advanced_retriever=self.advanced_retriever,
            answer_generator=self.answer_generator
        )

        print("\n" + "=" * 70)
        print("ADVANCED RAG READY")
        print("=" * 70)

        print(
            "\nYour document is ready for questions."
        )

    def chat(self):

        print("\n")
        print("=" * 70)
        print("ADVANCED RAG CHAT")
        print("=" * 70)

        print("\nCommands:")
        print("  clear  -> clear conversation memory")
        print("  exit   -> exit Advanced RAG")

        while True:

            print("\n")

            question = input(
                "Enter your question: "
            ).strip()

            if not question:

                print(
                    "Please enter a question."
                )

                continue

            if question.lower() == "exit":

                print(
                    "\nExiting Advanced RAG..."
                )

                break

            if question.lower() == "clear":

                self.pipeline.clear_memory()

                continue

            try:

                self.pipeline.ask(
                    question
                )

            except Exception as error:

                print("\n")
                print("=" * 70)
                print("ERROR")
                print("=" * 70)

                print(error)

                print(
                    "\nPlease check the error above."
                )


def main():

    application = AdvancedRAGApplication()

    try:

        application.setup()

        application.chat()

    except KeyboardInterrupt:

        print(
            "\n\nAdvanced RAG stopped by user."
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("APPLICATION ERROR")
        print("=" * 70)

        print(error)


if __name__ == "__main__":

    main()
