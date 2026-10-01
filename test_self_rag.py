from pathlib import Path

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever

from self_rag.self_rag_pipeline import SelfRAGPipeline


# ============================================================
# 1. PDF Path
# ============================================================

PDF_PATH = (
    r"C:\Users\samyu\Downloads\air-quality-and-health.pdf"
)


# ============================================================
# 2. Check PDF
# ============================================================

pdf_file = Path(PDF_PATH)

if not pdf_file.exists():

    print(
        "PDF file not found:"
    )

    print(PDF_PATH)

    raise SystemExit


# ============================================================
# 3. Load PDF
# ============================================================

print("\nLoading PDF...")

loader = PDFLoader()

documents = loader.load_pdf(
    PDF_PATH
)


# ============================================================
# 4. Split PDF
# ============================================================

print("\nSplitting document...")

splitter = TextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)


# ============================================================
# 5. Create FAISS
# ============================================================

print("\nCreating FAISS vector store...")

faiss_store = FAISSVectorStore()

vector_store = faiss_store.create(
    chunks
)


# ============================================================
# 6. Create FAISS Retriever
# ============================================================

faiss_retriever = RAGRetriever(
    vector_store=vector_store,
    top_k=3,
    threshold=1.2
)


# ============================================================
# 7. Create BM25
# ============================================================

print("\nCreating BM25 retriever...")

bm25_retriever = BM25Retriever(
    documents=chunks
)


# ============================================================
# 8. Create Self-RAG Pipeline
# ============================================================

print("\nCreating Self-RAG pipeline...")

pipeline = SelfRAGPipeline(
    faiss_retriever=faiss_retriever,
    bm25_retriever=bm25_retriever,
    top_k=3
)


# ============================================================
# 9. Self-RAG Chat
# ============================================================

print("\n" + "=" * 70)
print("SELF-RAG READY")
print("=" * 70)

print("\nCommands:")
print("  history -> show conversation history")
print("  clear   -> clear conversation memory")
print("  exit    -> exit Self-RAG")


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
            "\nExiting Self-RAG..."
        )

        break

    if question.lower() == "history":

        pipeline.show_history()

        continue

    if question.lower() == "clear":

        pipeline.clear_memory()

        continue

    try:

        pipeline.ask(
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