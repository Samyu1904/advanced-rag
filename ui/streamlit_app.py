import sys
import io
import contextlib
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



# ============================================================
# QUERY CONTEXT / FOLLOW-UP PREPARATION
# ============================================================

def get_previous_user_question(current_question=None):
    """Return the user question immediately before the current one."""
    user_questions = [
        message.get("content", "").strip()
        for message in st.session_state.get("messages", [])
        if message.get("role") == "user"
    ]

    if current_question and user_questions:
        # The current question is already appended to Streamlit history
        # before get_answer() is called.
        if user_questions[-1] == current_question.strip():
            user_questions = user_questions[:-1]

    return user_questions[-1] if user_questions else None


def prepare_ui_question(question):
    """
    Add the previous topic only for likely follow-up questions and
    add lightweight retrieval synonyms for common paraphrases.
    """
    previous_question = get_previous_user_question(question)

    return prepare_retrieval_question(
        question,
        previous_question=previous_question
    )

# ============================================================
# EXISTING RAG COMPONENTS
# ============================================================

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.answer_generator import AnswerGenerator

from retrieval.reranker import DocumentReranker
from retrieval.reranking_retriever import RerankingRetriever

from memory.conversation_memory import ConversationMemory
from memory.question_handler import QuestionHandler

from advanced.advanced_reranking_retriever import (
    AdvancedRerankingRetriever
)
from advanced.advanced_answer_generator import (
    AdvancedAnswerGenerator
)
from advanced.advanced_pipeline import (
    AdvancedRAGPipeline
)

from crag.crag_pipeline import CRAGPipeline

from self_rag.self_rag_pipeline import SelfRAGPipeline

from agentic_rag.agentic_rag_pipeline import (
    AgenticRAGPipeline
)

from graph_rag.graph_builder import GraphBuilder
from graph_rag.graph_pipeline import GraphRAGPipeline


# Parent-Child LangGraph builder already exists in app.py.
from app import (
    build_parent_child_graph,
    build_rag_components,
    build_parent_child_components,
    build_advanced_pipeline,
    build_crag_pipeline,
    build_self_rag_pipeline,
    build_agentic_rag_pipeline,
    build_graph_rag_pipeline,
    prepare_retrieval_question
)

from ingestion.parent_child_splitter import ParentChildSplitter
from vectorstore.child_vectorstore import ChildVectorStore
from retrieval.parent_store import ParentStore
from retrieval.parent_child_retriever import ParentChildRetriever
from retrieval.parent_child_answer_generator import (
    ParentChildAnswerGenerator
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .rag-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "documents": None,
    "chunks": None,
    "vector_store": None,
    "faiss_retriever": None,
    "bm25_retriever": None,
    "hybrid_retriever": None,
    "answer_generator": None,
    "reranker": None,
    "reranking_retriever": None,
    "conversation_memory": None,
    "question_handler": None,
    "loaded_file_name": None,
    "loaded_file_path": None,
    "loaded_file_bytes": None,
    "pipelines": {},
    "parent_child_graph": None,
    "graph_pipeline": None,
    "graph_file_name": None,
    "parent_child_file_name": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# RAG OPTIONS
# ============================================================

RAG_OPTIONS = {
    "Basic RAG": "basic",
    "Hybrid RAG": "hybrid",
    "Parent-Child RAG + LangGraph": "parent_child",
    "Reranking RAG": "reranking",
    "Advanced RAG": "advanced",
    "CRAG": "crag",
    "Self-RAG": "self_rag",
    "Agentic RAG": "agentic",
    "Graph RAG": "graph"
}


# ============================================================
# HELPER: RESET RAG STATE
# ============================================================

def reset_rag_state():

    st.session_state.documents = None
    st.session_state.chunks = None
    st.session_state.vector_store = None

    st.session_state.faiss_retriever = None
    st.session_state.bm25_retriever = None
    st.session_state.hybrid_retriever = None

    st.session_state.answer_generator = None
    st.session_state.reranker = None
    st.session_state.reranking_retriever = None

    st.session_state.conversation_memory = None
    st.session_state.question_handler = None

    st.session_state.pipelines = {}

    st.session_state.parent_child_graph = None
    st.session_state.graph_pipeline = None
    st.session_state.graph_file_name = None
    st.session_state.parent_child_file_name = None


# ============================================================
# HELPER: EXTRACT ANSWER FROM PIPELINE OUTPUT
# ============================================================

def extract_answer_from_output(output):

    if not output:
        return "No answer was generated."

    output = output.strip()

    # Most of the existing pipelines print:
    #
    # Answer:
    # actual answer
    #
    if "Answer:" in output:

        answer = output.rsplit(
            "Answer:",
            1
        )[1].strip()

        # Remove trailing separators
        if "----------------------------------------" in answer:

            answer = answer.split(
                "----------------------------------------",
                1
            )[0].strip()

        if answer:
            return answer

    return output


# ============================================================
# HELPER: RUN PIPELINE AND CAPTURE EXISTING TERMINAL OUTPUT
# ============================================================

def run_existing_pipeline(pipeline, question):

    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):

        result = pipeline.ask(question)

    captured_output = output_buffer.getvalue()

    # Some pipelines may return the answer directly.
    if isinstance(result, str) and result.strip():

        return result.strip()

    return extract_answer_from_output(
        captured_output
    )


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(uploaded_file):

    if uploaded_file is None:
        return False

    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    if (
        st.session_state.loaded_file_name == file_name
        and st.session_state.loaded_file_bytes == file_bytes
        and st.session_state.documents is not None
    ):
        return True

    reset_rag_state()

    documents_folder = PROJECT_ROOT / "data" / "documents"
    documents_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = documents_folder / file_name

    with open(pdf_path, "wb") as file:
        file.write(file_bytes)

    st.session_state.loaded_file_name = file_name
    st.session_state.loaded_file_path = str(pdf_path)
    st.session_state.loaded_file_bytes = file_bytes

    components = build_rag_components(
        str(pdf_path)
    )

    st.session_state.documents = components["documents"]
    st.session_state.chunks = components["chunks"]
    st.session_state.vector_store = components["vector_store"]
    st.session_state.faiss_retriever = components["faiss_retriever"]
    st.session_state.bm25_retriever = components["bm25_retriever"]
    st.session_state.hybrid_retriever = components["hybrid_retriever"]
    st.session_state.answer_generator = components["answer_generator"]
    st.session_state.reranker = components["reranker"]
    st.session_state.reranking_retriever = components["reranking_retriever"]

    conversation_memory = ConversationMemory(
        max_history=5
    )

    question_handler = QuestionHandler(
        conversation_memory
    )

    st.session_state.conversation_memory = (
        conversation_memory
    )

    st.session_state.question_handler = (
        question_handler
    )

    return True


# ============================================================
# BASIC RAG
# ============================================================

def answer_basic(question):

    retriever = st.session_state.faiss_retriever
    answer_generator = st.session_state.answer_generator
    question_handler = st.session_state.question_handler
    conversation_memory = st.session_state.conversation_memory

    complete_question, needs_clarification = (
        question_handler.process_question(question)
    )

    if needs_clarification:

        return (
            "Could you please provide more context "
            "for your question?"
        )

    question = complete_question
    retrieval_question = prepare_ui_question(question)

    results = retriever.retrieve(
        retrieval_question
    )

    if retriever.is_relevant(results):

        answer = answer_generator.generate_pdf_answer(
            question,
            results
        )

    else:

        answer = answer_generator.generate_general_answer(
            question
        )

    conversation_memory.add_conversation(
        question,
        answer
    )

    return answer


# ============================================================
# HYBRID RAG
# ============================================================

def answer_hybrid(question):

    retriever = st.session_state.hybrid_retriever
    answer_generator = st.session_state.answer_generator
    question_handler = st.session_state.question_handler
    conversation_memory = st.session_state.conversation_memory

    complete_question, needs_clarification = (
        question_handler.process_question(question)
    )

    if needs_clarification:

        return (
            "Could you please provide more context "
            "for your question?"
        )

    question = complete_question
    retrieval_question = prepare_ui_question(question)

    results = retriever.retrieve(
        retrieval_question
    )

    if results:

        answer = answer_generator.generate_pdf_answer(
            question,
            results
        )

    else:

        answer = answer_generator.generate_general_answer(
            question
        )

    conversation_memory.add_conversation(
        question,
        answer
    )

    return answer


# ============================================================
# RERANKING RAG
# ============================================================

def answer_reranking(question):

    retriever = st.session_state.reranking_retriever
    answer_generator = st.session_state.answer_generator
    question_handler = st.session_state.question_handler
    conversation_memory = st.session_state.conversation_memory

    complete_question, needs_clarification = (
        question_handler.process_question(question)
    )

    if needs_clarification:

        return (
            "Could you please provide more context "
            "for your question?"
        )

    question = complete_question
    retrieval_question = prepare_ui_question(question)

    results = retriever.retrieve(
        retrieval_question
    )

    if results:

        answer = answer_generator.generate_pdf_answer(
            question,
            results
        )

    else:

        answer = answer_generator.generate_general_answer(
            question
        )

    conversation_memory.add_conversation(
        question,
        answer
    )

    return answer


# ============================================================
# PARENT-CHILD RAG + LANGGRAPH
# ============================================================

def get_parent_child_graph():

    file_name = st.session_state.loaded_file_name

    if (
        st.session_state.parent_child_graph is not None
        and st.session_state.parent_child_file_name == file_name
    ):
        return st.session_state.parent_child_graph

    components = build_parent_child_components(
        st.session_state.documents
    )

    st.session_state.parent_child_graph = components["graph"]
    st.session_state.parent_child_file_name = file_name

    return components["graph"]


def answer_parent_child(question):

    graph = get_parent_child_graph()
    retrieval_question = prepare_ui_question(question)

    initial_state = {
        "question": retrieval_question,
        "processed_question": "",
        "parent_documents": [],
        "answer": "",
        "is_relevant": False,
        "waiting_for_clarification": False
    }

    final_state = graph.invoke(
        initial_state
    )

    return final_state.get(
        "answer",
        "No answer was generated."
    )


# ============================================================
# ADVANCED RAG
# ============================================================

def get_advanced_pipeline():

    key = "advanced"

    if key in st.session_state.pipelines:
        return st.session_state.pipelines[key]

    faiss_retriever = st.session_state.faiss_retriever
    bm25_retriever = st.session_state.bm25_retriever
    reranker = st.session_state.reranker

    pipeline = build_advanced_pipeline(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker
    )

    st.session_state.pipelines[key] = pipeline

    return pipeline


def answer_advanced(question):

    pipeline = get_advanced_pipeline()

    retrieval_question = prepare_ui_question(question)

    return run_existing_pipeline(
        pipeline,
        retrieval_question
    )


# ============================================================
# CRAG
# ============================================================

def get_crag_pipeline():

    key = "crag"

    if key in st.session_state.pipelines:
        return st.session_state.pipelines[key]

    pipeline = build_crag_pipeline(
        faiss_retriever=st.session_state.faiss_retriever,
        bm25_retriever=st.session_state.bm25_retriever,
        reranker=st.session_state.reranker
    )

    st.session_state.pipelines[key] = pipeline

    return pipeline


def answer_crag(question):

    pipeline = get_crag_pipeline()

    retrieval_question = prepare_ui_question(question)

    return run_existing_pipeline(
        pipeline,
        retrieval_question
    )


# ============================================================
# SELF-RAG
# ============================================================

def get_self_rag_pipeline():

    key = "self_rag"

    if key in st.session_state.pipelines:
        return st.session_state.pipelines[key]

    pipeline = build_self_rag_pipeline(
        faiss_retriever=st.session_state.faiss_retriever,
        bm25_retriever=st.session_state.bm25_retriever
    )

    st.session_state.pipelines[key] = pipeline

    return pipeline


def answer_self_rag(question):

    pipeline = get_self_rag_pipeline()

    retrieval_question = prepare_ui_question(question)

    return run_existing_pipeline(
        pipeline,
        retrieval_question
    )


# ============================================================
# AGENTIC RAG
# ============================================================

def get_agentic_pipeline():

    key = "agentic"

    if key in st.session_state.pipelines:
        return st.session_state.pipelines[key]

    pipeline = build_agentic_rag_pipeline(
        faiss_retriever=st.session_state.faiss_retriever,
        bm25_retriever=st.session_state.bm25_retriever
    )

    st.session_state.pipelines[key] = pipeline

    return pipeline


def answer_agentic(question):

    pipeline = get_agentic_pipeline()

    retrieval_question = prepare_ui_question(question)

    return run_existing_pipeline(
        pipeline,
        retrieval_question
    )


# ============================================================
# GRAPH RAG
# ============================================================

def get_graph_pipeline():

    file_name = st.session_state.loaded_file_name

    if (
        st.session_state.graph_pipeline is not None
        and st.session_state.graph_file_name == file_name
    ):
        return st.session_state.graph_pipeline

    documents = st.session_state.documents

    pipeline = build_graph_rag_pipeline(
        documents
    )

    st.session_state.graph_pipeline = pipeline
    st.session_state.graph_file_name = file_name

    return pipeline


def answer_graph(question):

    pipeline = get_graph_pipeline()

    retrieval_question = prepare_ui_question(question)

    return run_existing_pipeline(
        pipeline,
        retrieval_question
    )


# ============================================================
# MAIN ANSWER ROUTER
# ============================================================

def get_answer(rag_type, question):

    if rag_type == "Basic RAG":

        return answer_basic(question)

    if rag_type == "Hybrid RAG":

        return answer_hybrid(question)

    if rag_type == "Parent-Child RAG + LangGraph":

        return answer_parent_child(question)

    if rag_type == "Reranking RAG":

        return answer_reranking(question)

    if rag_type == "Advanced RAG":

        return answer_advanced(question)

    if rag_type == "CRAG":

        return answer_crag(question)

    if rag_type == "Self-RAG":

        return answer_self_rag(question)

    if rag_type == "Agentic RAG":

        return answer_agentic(question)

    if rag_type == "Graph RAG":

        return answer_graph(question)

    return "Unknown RAG architecture."


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Advanced RAG System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore multiple Retrieval-Augmented Generation architectures '
    'using the same document.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    selected_rag = st.selectbox(
        "Choose RAG Architecture",
        list(RAG_OPTIONS.keys())
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown(
        "### Architectures"
    )

    for index, name in enumerate(
        RAG_OPTIONS.keys(),
        start=1
    ):

        st.write(
            f"{index}. {name}"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        if st.session_state.conversation_memory:

            try:
                st.session_state.conversation_memory.clear()
            except Exception:
                pass

        st.session_state.pipelines = {}

        st.rerun()


# ============================================================
# PDF STATUS
# ============================================================

if uploaded_file is not None:

    try:

        with st.spinner(
            "Loading and preparing your PDF..."
        ):

            load_pdf(uploaded_file)

        st.success(
            f"PDF loaded: {uploaded_file.name}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Pages",
                len(st.session_state.documents)
            )

        with col2:

            st.metric(
                "Chunks",
                len(st.session_state.chunks)
            )

        with col3:

            st.metric(
                "RAG",
                selected_rag
            )

    except Exception as error:

        st.error(
            f"PDF loading failed: {error}"
        )

        st.stop()

else:

    st.info(
        "👈 Upload a PDF from the sidebar to start."
    )


# ============================================================
# CURRENT RAG
# ============================================================

st.markdown(
    f"### 🔎 {selected_rag}"
)

if selected_rag == "Basic RAG":

    st.caption(
        "Vector similarity search using FAISS."
    )

elif selected_rag == "Hybrid RAG":

    st.caption(
        "Combines semantic FAISS retrieval with BM25 keyword retrieval."
    )

elif selected_rag == "Parent-Child RAG + LangGraph":

    st.caption(
        "Retrieves small child chunks and returns their larger parent context through LangGraph."
    )

elif selected_rag == "Reranking RAG":

    st.caption(
        "Retrieves candidates and reranks them using a cross-encoder."
    )

elif selected_rag == "Advanced RAG":

    st.caption(
        "Combines advanced retrieval, reranking and answer generation."
    )

elif selected_rag == "CRAG":

    st.caption(
        "Corrective RAG evaluates retrieval quality and can correct poor retrieval."
    )

elif selected_rag == "Self-RAG":

    st.caption(
        "Self-RAG evaluates and improves its own retrieval/answer process."
    )

elif selected_rag == "Agentic RAG":

    st.caption(
        "Agentic workflow can reason over retrieval and use multiple retrieval strategies."
    )

elif selected_rag == "Graph RAG":

    st.caption(
        "Uses a knowledge graph for entity and relationship-based retrieval."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF..."
)


if question:

    if st.session_state.documents is None:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            f"Running {selected_rag}..."
        ):

            try:

                answer = get_answer(
                    selected_rag,
                    question
                )

                if not answer:

                    answer = (
                        "I could not generate an answer."
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                error_message = str(error)

                if "429" in error_message:

                    st.error(
                        "Gemini API quota/rate limit was reached. "
                        "Please wait and try again."
                    )

                else:

                    st.error(
                        f"Error while running {selected_rag}: "
                        f"{error}"
                    )