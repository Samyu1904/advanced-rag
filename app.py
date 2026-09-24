import hashlib
import json
from pathlib import Path

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.answer_generator import AnswerGenerator

from memory.question_handler import QuestionHandler
from memory.conversation_memory import ConversationMemory


VECTORSTORE_PATH = "vectorstore/current_index"
METADATA_PATH = Path(VECTORSTORE_PATH) / "metadata.json"


# ============================================================
# 1. Calculate PDF Hash
# ============================================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# 2. Load Metadata
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():
        return None

    with open(METADATA_PATH, "r") as file:

        return json.load(file)


# ============================================================
# 3. Save Metadata
# ============================================================

def save_metadata(pdf_path, pdf_hash):

    metadata = {
        "pdf_path": str(Path(pdf_path).resolve()),
        "pdf_hash": pdf_hash
    }

    with open(METADATA_PATH, "w") as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


# ============================================================
# 4. Main Application
# ============================================================

def main():

    print("\n========================================")
    print("             RAG ASSISTANT")
    print("========================================")

    # ========================================================
    # 1. Get PDF from User
    # ========================================================

    pdf_path = input(
        "\nEnter the full path of your PDF: "
    ).strip()

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():

        print("\nPDF file not found.")

        return

    if pdf_file.suffix.lower() != ".pdf":

        print("\nPlease provide a PDF file.")

        return

    # ========================================================
    # 2. Calculate PDF Hash
    # ========================================================

    print("\nChecking PDF...")

    pdf_hash = get_file_hash(pdf_path)

    metadata = load_metadata()

    # ========================================================
    # 3. Create or Load FAISS Vector Store
    # ========================================================

    vector_store = FAISSVectorStore()

    chunks = None

    if (
        metadata
        and metadata.get("pdf_hash") == pdf_hash
        and Path(VECTORSTORE_PATH).exists()
    ):

        print("Existing FAISS vector store found.")
        print("Loading vector store...")

        db = vector_store.load(
            VECTORSTORE_PATH
        )

        # ----------------------------------------------------
        # Load PDF chunks for BM25
        # ----------------------------------------------------

        print("Loading PDF chunks for BM25...")

        pdf_loader = PDFLoader()

        documents = pdf_loader.load_pdf(
            pdf_path
        )

        text_splitter = TextSplitter()

        chunks = text_splitter.split_documents(
            documents
        )

    else:

        print("New PDF detected.")
        print("Creating FAISS vector store...")

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        pdf_loader = PDFLoader()

        documents = pdf_loader.load_pdf(
            pdf_path
        )

        # ----------------------------------------------------
        # Split PDF into chunks
        # ----------------------------------------------------

        text_splitter = TextSplitter()

        chunks = text_splitter.split_documents(
            documents
        )

        # ----------------------------------------------------
        # Create FAISS
        # ----------------------------------------------------

        db = vector_store.create(
            chunks
        )

        # ----------------------------------------------------
        # Save FAISS
        # ----------------------------------------------------

        vector_store.save(
            VECTORSTORE_PATH
        )

        save_metadata(
            pdf_path,
            pdf_hash
        )

        print("FAISS vector store saved.")

    # ========================================================
    # 4. Create FAISS Retriever
    # ========================================================

    retriever = RAGRetriever(
        vector_store=db,
        top_k=3,
        threshold=1.2
    )

    # ========================================================
    # 5. Create BM25 Retriever
    # ========================================================

    bm25_retriever = BM25Retriever(
        documents=chunks
    )

    # ========================================================
    # 6. Create Hybrid Retriever
    # ========================================================

    hybrid_retriever = HybridRetriever(
        faiss_retriever=retriever,
        bm25_retriever=bm25_retriever,
        top_k=3
    )

    # ========================================================
    # 7. Create Answer Generator
    # ========================================================

    answer_generator = AnswerGenerator()

    # ========================================================
    # 8. Create Conversation Memory
    # ========================================================

    conversation_memory = ConversationMemory(
        max_history=5
    )

    # ========================================================
    # 9. Create Question Handler
    # ========================================================

    question_handler = QuestionHandler(
        conversation_memory
    )

    print("\n========================================")
    print("PDF is ready!")
    print("FAISS + BM25 Hybrid Retrieval is ready!")
    print("Type 'exit' to stop.")
    print("========================================")

    # ========================================================
    # 10. Question Loop
    # ========================================================

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() == "exit":

            print("\nExiting RAG Assistant...")

            break

        # ----------------------------------------------------
        # Empty Question
        # ----------------------------------------------------

        if not question:

            print("Please enter a question.")

            continue

        # ====================================================
        # 11. Check Question Completeness
        # ====================================================

        complete_question, needs_clarification = (
            question_handler.process_question(
                question
            )
        )

        if needs_clarification:

            print("\nI need more information.")
            print("Please complete your question.")

            continue

        question = complete_question

        # ====================================================
        # 12. Hybrid Retrieval
        # ====================================================

        print("\nSearching...")

        results = hybrid_retriever.retrieve(
            question
        )

        relevant = len(results) > 0

        # ====================================================
        # 13. Generate Answer
        # ====================================================

        if relevant:

            print("Source: PDF")

            answer = answer_generator.generate_pdf_answer(
                question,
                results
            )

        else:

            print("Source: General Knowledge")

            answer = answer_generator.generate_general_answer(
                question
            )

        # ====================================================
        # 14. Save Conversation
        # ====================================================

        conversation_memory.add_conversation(
            question,
            answer
        )

        # ====================================================
        # 15. Display Answer
        # ====================================================

        print("\nAnswer:")
        print(answer)

        print("\n----------------------------------------")


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":
    main()