from pathlib import Path

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever

from agentic_rag.agentic_rag_pipeline import (
    AgenticRAGPipeline
)


class AgenticRAGApplication:

    def __init__(self):

        print("\n========================================")
        print("          AGENTIC RAG SYSTEM")
        print("========================================")

        self.pdf_path = input(
            "\nEnter the full path of your PDF: "
        ).strip()

        pdf_file = Path(
            self.pdf_path
        )

        if not pdf_file.exists():

            print(
                "\nPDF file not found."
            )

            raise SystemExit

        if pdf_file.suffix.lower() != ".pdf":

            print(
                "\nPlease provide a PDF file."
            )

            raise SystemExit

        self.documents = (
            self.load_pdf()
        )

        self.chunks = (
            self.split_documents()
        )

        self.faiss_retriever = (
            self.create_faiss_retriever()
        )

        self.bm25_retriever = (
            self.create_bm25_retriever()
        )

        self.pipeline = (
            self.create_agentic_rag_pipeline()
        )

    # ============================================================
    # LOAD PDF
    # ============================================================

    def load_pdf(self):

        print("\nLoading PDF...")

        loader = PDFLoader()

        documents = loader.load_pdf(
            self.pdf_path
        )

        return documents

    # ============================================================
    # SPLIT DOCUMENT
    # ============================================================

    def split_documents(self):

        print("\nSplitting document...")

        splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = (
            splitter.split_documents(
                self.documents
            )
        )

        return chunks

    # ============================================================
    # CREATE FAISS
    # ============================================================

    def create_faiss_retriever(self):

        print(
            "\nCreating FAISS vector store..."
        )

        faiss_store = (
            FAISSVectorStore()
        )

        vector_store = (
            faiss_store.create(
                self.chunks
            )
        )

        retriever = RAGRetriever(
            vector_store=vector_store,
            top_k=3,
            threshold=1.2
        )

        return retriever

    # ============================================================
    # CREATE BM25
    # ============================================================

    def create_bm25_retriever(self):

        print(
            "\nCreating BM25 retriever..."
        )

        retriever = BM25Retriever(
            documents=self.chunks
        )

        return retriever

    # ============================================================
    # CREATE AGENTIC RAG
    # ============================================================

    def create_agentic_rag_pipeline(self):

        print(
            "\nCreating Agentic RAG pipeline..."
        )

        pipeline = AgenticRAGPipeline(
            faiss_retriever=self.faiss_retriever,
            bm25_retriever=self.bm25_retriever,
            top_k=3,
            max_iterations=3
        )

        return pipeline

    # ============================================================
    # RUN APPLICATION
    # ============================================================

    def run(self):

        print("\n")
        print("=" * 70)
        print("AGENTIC RAG READY")
        print("=" * 70)

        print("\nCommands:")

        print(
            "  history -> show conversation history"
        )

        print(
            "  clear   -> clear conversation memory"
        )

        print(
            "  exit    -> exit Agentic RAG"
        )

        while True:

            question = input(
                "\nQuestion: "
            ).strip()

            if not question:

                print(
                    "Please enter a question."
                )

                continue

            if question.lower() == "exit":

                print(
                    "\nExiting Agentic RAG..."
                )

                break

            if question.lower() == "history":

                self.pipeline.show_history()

                continue

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


if __name__ == "__main__":

    application = (
        AgenticRAGApplication()
    )

    application.run()