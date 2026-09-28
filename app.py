import hashlib
import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.answer_generator import AnswerGenerator

from retrieval.reranker import DocumentReranker
from retrieval.reranking_retriever import RerankingRetriever

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

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
# 4. LangGraph State
# ============================================================

class RAGState(TypedDict):

    question: str

    processed_question: str

    parent_documents: list

    answer: str

    is_relevant: bool

    waiting_for_clarification: bool


# ============================================================
# 5. Build Parent-Child LangGraph
# ============================================================

def build_parent_child_graph(
    parent_child_retriever,
    answer_generator,
    question_handler,
    conversation_memory
):

    def process_question_node(state):

        question = state["question"]

        processed_question, needs_clarification = (
            question_handler.process_question(
                question
            )
        )

        return {
            "processed_question":
                processed_question
                if processed_question is not None
                else "",

            "waiting_for_clarification":
                needs_clarification
        }

    def question_status_decision(state):

        if state["waiting_for_clarification"]:
            return "clarification"

        return "retrieve"

    def clarification_node(state):

        return {
            "answer":
                "Could you please provide more context for your question?"
        }

    def retrieve_node(state):

        question = state["processed_question"]

        print("\nSearching Parent-Child chunks...")

        parents = parent_child_retriever.retrieve(
            question
        )

        return {
            "parent_documents": parents,
            "is_relevant": len(parents) > 0
        }

    def relevance_decision(state):

        if state["is_relevant"]:
            return "pdf_answer"

        return "general_answer"

    def pdf_answer_node(state):

        print("Source: Parent Document")

        answer = answer_generator.generate_answer(
            state["processed_question"],
            state["parent_documents"]
        )

        return {
            "answer": answer
        }

    def general_answer_node(state):

        print("Source: General Knowledge")

        answer = answer_generator.generate_general_answer(
            state["processed_question"]
        )

        return {
            "answer": answer
        }

    def save_memory_node(state):

        if state["processed_question"]:

            conversation_memory.add_conversation(
                state["processed_question"],
                state["answer"]
            )

        return {}

    graph = StateGraph(RAGState)

    graph.add_node(
        "process_question",
        process_question_node
    )

    graph.add_node(
        "clarification",
        clarification_node
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "pdf_answer",
        pdf_answer_node
    )

    graph.add_node(
        "general_answer",
        general_answer_node
    )

    graph.add_node(
        "save_memory",
        save_memory_node
    )

    graph.set_entry_point(
        "process_question"
    )

    graph.add_conditional_edges(
        "process_question",
        question_status_decision,
        {
            "clarification": "clarification",
            "retrieve": "retrieve"
        }
    )

    graph.add_edge(
        "clarification",
        END
    )

    graph.add_conditional_edges(
        "retrieve",
        relevance_decision,
        {
            "pdf_answer": "pdf_answer",
            "general_answer": "general_answer"
        }
    )

    graph.add_edge(
        "pdf_answer",
        "save_memory"
    )

    graph.add_edge(
        "general_answer",
        "save_memory"
    )

    graph.add_edge(
        "save_memory",
        END
    )

    return graph.compile()


# ============================================================
# 6. Run Basic RAG
# ============================================================

def run_basic_rag(
    db,
    question_handler,
    conversation_memory
):

    retriever = RAGRetriever(
        vector_store=db,
        top_k=3,
        threshold=1.2
    )

    answer_generator = AnswerGenerator()

    print("\n========================================")
    print("             BASIC RAG")
    print("========================================")

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        complete_question, needs_clarification = (
            question_handler.process_question(
                question
            )
        )

        if needs_clarification:

            print(
                "\nI need more information."
            )

            print(
                "Please complete your question."
            )

            continue

        question = complete_question

        print("\nSearching...")

        results = retriever.retrieve(
            question
        )

        relevant = retriever.is_relevant(
            results
        )

        if relevant:

            print("Source: PDF")

            answer = (
                answer_generator.generate_pdf_answer(
                    question,
                    results
                )
            )

        else:

            print("Source: General Knowledge")

            answer = (
                answer_generator.generate_general_answer(
                    question
                )
            )

        conversation_memory.add_conversation(
            question,
            answer
        )

        print("\nAnswer:")
        print(answer)

        print(
            "\n----------------------------------------"
        )


# ============================================================
# 7. Run Hybrid RAG
# ============================================================

def run_hybrid_rag(
    hybrid_retriever,
    answer_generator,
    question_handler,
    conversation_memory
):

    print("\n========================================")
    print("             HYBRID RAG")
    print("========================================")

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        complete_question, needs_clarification = (
            question_handler.process_question(
                question
            )
        )

        if needs_clarification:

            print(
                "\nI need more information."
            )

            print(
                "Please complete your question."
            )

            continue

        question = complete_question

        print("\nSearching...")

        results = hybrid_retriever.retrieve(
            question
        )

        relevant = len(results) > 0

        if relevant:

            print("Source: PDF")

            answer = (
                answer_generator.generate_pdf_answer(
                    question,
                    results
                )
            )

        else:

            print("Source: General Knowledge")

            answer = (
                answer_generator.generate_general_answer(
                    question
                )
            )

        conversation_memory.add_conversation(
            question,
            answer
        )

        print("\nAnswer:")
        print(answer)

        print(
            "\n----------------------------------------"
        )


# ============================================================
# 8. Run Reranking RAG
# ============================================================

def run_reranking_rag(
    reranking_retriever,
    answer_generator,
    question_handler,
    conversation_memory
):

    print("\n========================================")
    print("            RERANKING RAG")
    print("========================================")

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            print(
                "\nReturning to main menu..."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        complete_question, needs_clarification = (
            question_handler.process_question(
                question
            )
        )

        if needs_clarification:

            print(
                "\nI need more information."
            )

            print(
                "Please complete your question."
            )

            continue

        question = complete_question

        results = reranking_retriever.retrieve(
            question
        )

        if results:

            print(
                "\nSource: PDF"
            )

            answer = (
                answer_generator.generate_pdf_answer(
                    question,
                    results
                )
            )

        else:

            print(
                "\nSource: General Knowledge"
            )

            answer = (
                answer_generator.generate_general_answer(
                    question
                )
            )

        conversation_memory.add_conversation(
            question,
            answer
        )

        print("\nAnswer:")
        print(answer)

        print(
            "\n----------------------------------------"
        )


# ============================================================
# 9. Run Parent-Child RAG + LangGraph
# ============================================================

def run_parent_child_rag(
    documents
):

    # IMPORTANT:
    # Parent-Child imports are kept inside this function.
    # Therefore Basic, Hybrid and Reranking RAG do not
    # depend on Parent-Child files during application startup.

    from ingestion.parent_child_splitter import (
        ParentChildSplitter
    )

    from vectorstore.child_vectorstore import (
        ChildVectorStore
    )

    from retrieval.parent_store import (
        ParentStore
    )

    from retrieval.parent_child_retriever import (
        ParentChildRetriever
    )

    from retrieval.parent_child_answer_generator import (
        ParentChildAnswerGenerator
    )

    print("\n========================================")
    print("      PARENT-CHILD RAG + LANGGRAPH")
    print("========================================")

    splitter = ParentChildSplitter(
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=500,
        child_chunk_overlap=100
    )

    parents, children = (
        splitter.split_documents(
            documents
        )
    )

    parent_store = ParentStore(
        parents
    )

    child_vector_store = ChildVectorStore()

    child_vector_store.create(
        children
    )

    parent_child_retriever = ParentChildRetriever(
        child_vector_store=child_vector_store,
        parent_store=parent_store,
        top_k=3,
        threshold=1.2
    )

    answer_generator = (
        ParentChildAnswerGenerator()
    )

    conversation_memory = ConversationMemory(
        max_history=5
    )

    question_handler = QuestionHandler(
        conversation_memory
    )

    graph = build_parent_child_graph(
        parent_child_retriever,
        answer_generator,
        question_handler,
        conversation_memory
    )

    print("\nLangGraph workflow is ready!")

    print(
        "\nType 'exit' to return to the main menu."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":

            print(
                "\nReturning to main menu..."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        print(
            "\nRunning LangGraph..."
        )

        initial_state = {

            "question": question,

            "processed_question": "",

            "parent_documents": [],

            "answer": "",

            "is_relevant": False,

            "waiting_for_clarification": False
        }

        final_state = graph.invoke(
            initial_state
        )

        print("\nProcessed Question:")

        print(
            final_state["processed_question"]
        )

        print(
            "\nWaiting for Clarification:"
        )

        print(
            final_state[
                "waiting_for_clarification"
            ]
        )

        print(
            "\nRelevant to PDF:"
        )

        print(
            final_state["is_relevant"]
        )

        print("\nAnswer:")

        print(
            final_state["answer"]
        )

        print(
            "\n----------------------------------------"
        )


# ============================================================
# 10. Main Application
# ============================================================

def main():

    print("\n========================================")
    print("           ADVANCED RAG SYSTEM")
    print("========================================")

    pdf_path = input(
        "\nEnter the full path of your PDF: "
    ).strip()

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():

        print(
            "\nPDF file not found."
        )

        return

    if pdf_file.suffix.lower() != ".pdf":

        print(
            "\nPlease provide a PDF file."
        )

        return

    print(
        "\nChecking PDF..."
    )

    pdf_hash = get_file_hash(
        pdf_path
    )

    metadata = load_metadata()

    pdf_loader = PDFLoader()

    documents = pdf_loader.load_pdf(
        pdf_path
    )

    vector_store = FAISSVectorStore()

    chunks = None

    if (
        metadata
        and metadata.get("pdf_hash") == pdf_hash
        and Path(VECTORSTORE_PATH).exists()
    ):

        print(
            "\nExisting FAISS vector store found."
        )

        print(
            "Loading vector store..."
        )

        db = vector_store.load(
            VECTORSTORE_PATH
        )

        print(
            "Loading PDF chunks for BM25..."
        )

        text_splitter = TextSplitter()

        chunks = (
            text_splitter.split_documents(
                documents
            )
        )

    else:

        print(
            "\nNew PDF detected."
        )

        print(
            "Creating FAISS vector store..."
        )

        text_splitter = TextSplitter()

        chunks = (
            text_splitter.split_documents(
                documents
            )
        )

        db = vector_store.create(
            chunks
        )

        vector_store.save(
            VECTORSTORE_PATH
        )

        save_metadata(
            pdf_path,
            pdf_hash
        )

        print(
            "FAISS vector store saved."
        )

    # ========================================================
    # Basic + Hybrid Components
    # ========================================================

    retriever = RAGRetriever(
        vector_store=db,
        top_k=3,
        threshold=1.2
    )

    bm25_retriever = BM25Retriever(
        documents=chunks
    )

    hybrid_retriever = HybridRetriever(
        faiss_retriever=retriever,
        bm25_retriever=bm25_retriever,
        top_k=3
    )

    answer_generator = AnswerGenerator()

    # ========================================================
    # Reranking Components
    # ========================================================

    print(
        "\nLoading Reranker..."
    )

    reranker = DocumentReranker()

    reranking_retriever = RerankingRetriever(
        faiss_retriever=retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        candidate_k=10,
        top_k=3
    )

    print(
        "Reranking RAG components ready!"
    )

    # ========================================================
    # Main Menu
    # ========================================================

    while True:

        print("\n========================================")
        print("              SELECT MODE")
        print("========================================")

        print(
            "1. Basic RAG"
        )

        print(
            "2. Hybrid RAG"
        )

        print(
            "3. Parent-Child RAG + LangGraph"
        )

        print(
            "4. Reranking RAG"
        )

        print(
            "5. Exit"
        )

        choice = input(
            "\nEnter your choice: "
        ).strip()

        # ====================================================
        # Basic RAG
        # ====================================================

        if choice == "1":

            conversation_memory = (
                ConversationMemory(
                    max_history=5
                )
            )

            question_handler = (
                QuestionHandler(
                    conversation_memory
                )
            )

            run_basic_rag(
                db,
                question_handler,
                conversation_memory
            )

        # ====================================================
        # Hybrid RAG
        # ====================================================

        elif choice == "2":

            conversation_memory = (
                ConversationMemory(
                    max_history=5
                )
            )

            question_handler = (
                QuestionHandler(
                    conversation_memory
                )
            )

            run_hybrid_rag(
                hybrid_retriever,
                answer_generator,
                question_handler,
                conversation_memory
            )

        # ====================================================
        # Parent-Child RAG + LangGraph
        # ====================================================

        elif choice == "3":

            run_parent_child_rag(
                documents
            )

        # ====================================================
        # Reranking RAG
        # ====================================================

        elif choice == "4":

            conversation_memory = (
                ConversationMemory(
                    max_history=5
                )
            )

            question_handler = (
                QuestionHandler(
                    conversation_memory
                )
            )

            run_reranking_rag(
                reranking_retriever,
                answer_generator,
                question_handler,
                conversation_memory
            )

        # ====================================================
        # Exit
        # ====================================================

        elif choice == "5":

            print(
                "\nExiting Advanced RAG System..."
            )

            break

        else:

            print(
                "\nInvalid choice."
            )

            print(
                "Please select 1, 2, 3, 4, or 5."
            )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    main()