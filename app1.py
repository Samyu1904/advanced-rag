from pathlib import Path

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import DocumentReranker

from vectorstore.faiss_store import FAISSVectorStore

from crag.crag_pipeline import CRAGPipeline


class CRAGApplication:

    def __init__(self):

        self.documents = None
        self.chunks = None

        self.vector_store = None

        self.faiss_retriever = None
        self.bm25_retriever = None
        self.reranker = None

        self.pipeline = None

    # =========================================================
    # STEP 1 - LOAD PDF
    # =========================================================

    def load_pdf(self):

        print("\n")
        print("=" * 70)
        print("                         CRAG")
        print("=" * 70)

        pdf_path = input(
            "\nEnter the full path of your PDF: "
        ).strip()

        pdf_file = Path(pdf_path)

        if not pdf_file.exists():

            print("\nPDF file not found.")

            return False

        if pdf_file.suffix.lower() != ".pdf":

            print("\nPlease provide a PDF file.")

            return False

        print("\nLoading PDF...")

        loader = PDFLoader()

        self.documents = loader.load_pdf(
            pdf_path
        )

        return True

    # =========================================================
    # STEP 2 - CHUNK DOCUMENT
    # =========================================================

    def create_chunks(self):

        print("\nSplitting document...")

        splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.chunks = splitter.split_documents(
            self.documents
        )

    # =========================================================
    # STEP 3 - CREATE FAISS
    # =========================================================

    def create_faiss(self):

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

    # =========================================================
    # STEP 4 - CREATE BM25
    # =========================================================

    def create_bm25(self):

        print("\nCreating BM25 retriever...")

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

    # =========================================================
    # STEP 5 - LOAD RERANKER
    # =========================================================

    def create_reranker(self):

        print("\nLoading Cross-Encoder reranker...")

        self.reranker = DocumentReranker()

    # =========================================================
    # STEP 6 - CREATE CRAG PIPELINE
    # =========================================================

    def create_pipeline(self):

        print("\nCreating CRAG pipeline...")

        self.pipeline = CRAGPipeline(
            faiss_retriever=self.faiss_retriever,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker
        )

    # =========================================================
    # COMPLETE SETUP
    # =========================================================

    def setup(self):

        if not self.load_pdf():

            return False

        self.create_chunks()

        self.create_faiss()

        self.create_bm25()

        self.create_reranker()

        self.create_pipeline()

        print("\n")
        print("=" * 70)
        print("                    CRAG READY")
        print("=" * 70)

        print("\nYour document is ready for questions.")

        return True

    # =========================================================
    # CRAG CHAT
    # =========================================================

    def chat(self):

        print("\n")
        print("=" * 70)
        print("                       CRAG CHAT")
        print("=" * 70)

        print("\nCommands:")
        print("  history -> show conversation history")
        print("  clear   -> clear conversation memory")
        print("  exit    -> exit CRAG")

        
        while True:

            print("\n")

            question = input(
                "Question: "
            ).strip()

            if not question:

                print(
                    "Please enter a question."
                )

                continue

            if question.lower() == "exit":

                print(
                    "\nExiting CRAG..."
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

    # =========================================================
    # RUN APPLICATION
    # =========================================================

    def run(self):

        if not self.setup():

            return

        self.chat()


def main():

    application = CRAGApplication()

    try:

        application.run()

    except KeyboardInterrupt:

        print(
            "\n\nCRAG stopped by user."
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("APPLICATION ERROR")
        print("=" * 70)

        print(error)


if __name__ == "__main__":

    main()