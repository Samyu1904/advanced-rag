import sys
import json
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from ingestion.pdf_loader import PDFLoader
from ingestion.parent_child_splitter import ParentChildSplitter

from vectorstore.child_vectorstore import ChildVectorStore

from retrieval.parent_store import ParentStore
from retrieval.parent_child_retriever import ParentChildRetriever

from graph_rag.graph_retriever import GraphRetriever


# ============================================================
# PATHS
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "air-quality-and-health.pdf"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_dataset.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_results_phase3.json"
)


# ============================================================
# METRICS
# ============================================================

def precision_at_k(retrieved_ids, relevant_ids, k=3):

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant = set(relevant_ids)

    hits = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant
    )

    return hits / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k=3):

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    hits = len(
        retrieved.intersection(relevant)
    )

    return hits / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):

    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def document_text(document):

    if hasattr(document, "page_content"):
        return document.page_content

    if isinstance(document, dict):

        if "page_content" in document:
            return document["page_content"]

        if "text" in document:
            return document["text"]

    return str(document)


def document_metadata(document):

    if hasattr(document, "metadata"):
        return document.metadata

    if isinstance(document, dict):
        return document.get("metadata", {})

    return {}


def get_chunk_id(document, chunks):

    text = document_text(document)

    for index, chunk in enumerate(chunks):

        if document_text(chunk) == text:
            return f"chunk_{index}"

    return None


# ============================================================
# PARENT-CHILD HELPERS
# ============================================================

def extract_parent_child_output(split_result):

    """
    Supports the common Parent-Child return formats:

        (parents, children)

        {
            "parents": ...,
            "children": ...
        }
    """

    if isinstance(split_result, tuple):

        if len(split_result) >= 2:

            parents = split_result[0]
            children = split_result[1]

            return parents, children

    if isinstance(split_result, dict):

        parents = split_result.get(
            "parents",
            []
        )

        children = split_result.get(
            "children",
            []
        )

        return parents, children

    raise ValueError(
        "Unexpected ParentChildSplitter output. "
        f"Received type: {type(split_result)}"
    )


def extract_parent_id(document):

    metadata = document_metadata(document)

    possible_keys = [
        "parent_id",
        "parent",
        "parent_doc_id",
        "parent_document_id"
    ]

    for key in possible_keys:

        if key in metadata:
            return metadata[key]

    return None


def map_parent_to_original_chunks(
    parent_document,
    original_chunks
):

    parent_text = document_text(
        parent_document
    )

    matches = []

    for index, chunk in enumerate(
        original_chunks
    ):

        chunk_text = document_text(chunk)

        if chunk_text in parent_text:

            matches.append(
                f"chunk_{index}"
            )

    return matches


# ============================================================
# PARENT-CHILD EVALUATION
# ============================================================

def evaluate_parent_child(
    original_chunks,
    dataset
):

    print("\n")
    print("=" * 70)
    print("PARENT-CHILD RAG EVALUATION")
    print("=" * 70)

    splitter = ParentChildSplitter()

    split_result = splitter.split_documents(
        original_chunks
    )

    parents, children = extract_parent_child_output(
        split_result
    )

    print(
        f"\nNumber of parents: {len(parents)}"
    )

    print(
        f"Number of children: {len(children)}"
    )

    # --------------------------------------------------------
    # Parent IDs
    # --------------------------------------------------------

    for index, parent in enumerate(parents):

        metadata = document_metadata(parent)

        if not any(
            key in metadata
            for key in [
                "parent_id",
                "parent",
                "parent_doc_id",
                "parent_document_id"
            ]
        ):

            metadata["parent_id"] = (
                f"parent_{index}"
            )

            try:
                parent.metadata = metadata
            except Exception:
                pass

    # --------------------------------------------------------
    # Child IDs
    # --------------------------------------------------------

    for index, child in enumerate(children):

        metadata = document_metadata(child)

        if not any(
            key in metadata
            for key in [
                "parent_id",
                "parent",
                "parent_doc_id",
                "parent_document_id"
            ]
        ):

            print(
                "\nWarning:"
                f" child_{index} has no parent_id."
            )

    # --------------------------------------------------------
    # Parent store
    # --------------------------------------------------------

    parent_store = ParentStore(
        parents
    )

    # --------------------------------------------------------
    # Child vector store
    # --------------------------------------------------------

    child_vector_store = ChildVectorStore()

    child_vector_store.create(
        children
    )

    # --------------------------------------------------------
    # Parent-child retriever
    # --------------------------------------------------------

    retriever = ParentChildRetriever(
        child_vector_store=child_vector_store,
        parent_store=parent_store,
        top_k=3,
        threshold=1.2
    )

    results = []

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    for item in dataset:

        question = item["question"]

        relevant_ids = item.get(
            "relevant_chunk_ids",
            []
        )

        print("\n" + "-" * 70)
        print(question)

        try:

            retrieved = retriever.retrieve(
                question
            )

            print(
                "\nRetrieved parent documents:"
            )

            print(
                f"Type: {type(retrieved)}"
            )

            # ------------------------------------------------
            # Normalize result
            # ------------------------------------------------

            documents = []

            if isinstance(
                retrieved,
                list
            ):

                for item_result in retrieved:

                    if isinstance(
                        item_result,
                        tuple
                    ):

                        documents.append(
                            item_result[0]
                        )

                    else:

                        documents.append(
                            item_result
                        )

            elif retrieved is not None:

                documents = [retrieved]

            # ------------------------------------------------
            # Convert parents back to original chunk IDs
            # ------------------------------------------------

            retrieved_ids = []

            for parent in documents:

                matched_chunks = (
                    map_parent_to_original_chunks(
                        parent,
                        original_chunks
                    )
                )

                retrieved_ids.extend(
                    matched_chunks
                )

            # Remove duplicates while preserving order

            retrieved_ids = list(
                dict.fromkeys(
                    retrieved_ids
                )
            )

            retrieved_ids = retrieved_ids[:3]

            print(
                f"Retrieved: {retrieved_ids}"
            )

            p = precision_at_k(
                retrieved_ids,
                relevant_ids,
                k=3
            )

            r = recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=3
            )

            mrr = reciprocal_rank(
                retrieved_ids,
                relevant_ids
            )

            print(
                f"Precision@3: {p:.3f}"
            )

            print(
                f"Recall@3:    {r:.3f}"
            )

            print(
                f"MRR:         {mrr:.3f}"
            )

            results.append(
                {
                    "question": question,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant_ids,
                    "precision_at_3": p,
                    "recall_at_3": r,
                    "mrr": mrr
                }
            )

        except Exception as error:

            print(
                "\nEvaluation failed:"
            )

            print(error)

            results.append(
                {
                    "question": question,
                    "retrieved_ids": [],
                    "relevant_ids": relevant_ids,
                    "precision_at_3": 0.0,
                    "recall_at_3": 0.0,
                    "mrr": 0.0,
                    "error": str(error)
                }
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    if results:

        precision = sum(
            item["precision_at_3"]
            for item in results
        ) / len(results)

        recall = sum(
            item["recall_at_3"]
            for item in results
        ) / len(results)

        mrr = sum(
            item["mrr"]
            for item in results
        ) / len(results)

    else:

        precision = 0.0
        recall = 0.0
        mrr = 0.0

    print("\n")
    print("=" * 70)
    print("Parent-Child Overall Results")
    print("=" * 70)

    print(
        f"Precision@3: {precision:.3f}"
    )

    print(
        f"Recall@3:    {recall:.3f}"
    )

    print(
        f"MRR:         {mrr:.3f}"
    )

    return {
        "precision_at_3": precision,
        "recall_at_3": recall,
        "mrr": mrr,
        "questions": results
    }


# ============================================================
# GRAPH RAG
# ============================================================

def evaluate_existing_graph(
    original_chunks,
    dataset,
    graph
):

    print("\n")
    print("=" * 70)
    print("GRAPH RAG EVALUATION")
    print("=" * 70)

    retriever = GraphRetriever(
        graph=graph,
        top_k=10
    )

    results = []

    for item in dataset:

        question = item["question"]

        relevant_ids = item.get(
            "relevant_chunk_ids",
            []
        )

        print("\n" + "-" * 70)
        print(question)

        try:

            retrieved = retriever.retrieve(
                question
            )

            # Graph results are nodes rather than
            # original document chunks.
            #
            # Therefore we report the graph results
            # separately instead of pretending that
            # graph nodes are chunk IDs.

            print(
                f"Graph results: {retrieved}"
            )

            results.append(
                {
                    "question": question,
                    "graph_results": str(
                        retrieved
                    ),
                    "relevant_chunk_ids": (
                        relevant_ids
                    )
                }
            )

        except Exception as error:

            print(
                f"Graph evaluation failed: {error}"
            )

            results.append(
                {
                    "question": question,
                    "graph_results": [],
                    "relevant_chunk_ids": (
                        relevant_ids
                    ),
                    "error": str(error)
                }
            )

    return {
        "note": (
            "Graph RAG retrieval was inspected "
            "using an existing graph. "
            "Standard chunk Precision@3, Recall@3 "
            "and MRR were not calculated because "
            "GraphRetriever returns graph nodes, "
            "not original chunk IDs."
        ),
        "questions": results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 3 RAG EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        raise FileNotFoundError(
            f"PDF not found:\n{PDF_PATH}"
        )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    print(
        f"\nLoading PDF:\n{PDF_PATH}"
    )

    loader = PDFLoader()

    documents = loader.load_pdf(
        str(PDF_PATH)
    )

    # --------------------------------------------------------
    # Original chunks
    # --------------------------------------------------------

    from ingestion.text_splitter import TextSplitter

    splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"\nOriginal chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Parent-Child
    # --------------------------------------------------------

    parent_child_results = (
        evaluate_parent_child(
            chunks,
            dataset
        )
    )

    # --------------------------------------------------------
    # Graph RAG
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # GraphBuilder.build_graph() calls Gemini.
    #
    # Your Gemini quota is currently exhausted,
    # so we deliberately DO NOT build a new graph.
    #
    # Graph RAG will therefore be recorded as
    # "not evaluated" in this phase.
    # --------------------------------------------------------

    graph_results = {
        "status": "not_evaluated",
        "reason": (
            "GraphBuilder requires Gemini to "
            "construct the knowledge graph. "
            "Gemini quota is currently exhausted. "
            "No Gemini call was made by this evaluator."
        )
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    final_results = {

        "phase": 3,

        "gemini_used": False,

        "parent_child": parent_child_results,

        "graph_rag": graph_results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)

    print(
        "\nParent-Child RAG"
    )

    print(
        f"Precision@3: "
        f"{parent_child_results['precision_at_3']:.3f}"
    )

    print(
        f"Recall@3:    "
        f"{parent_child_results['recall_at_3']:.3f}"
    )

    print(
        f"MRR:         "
        f"{parent_child_results['mrr']:.3f}"
    )

    print(
        "\nGraph RAG:"
    )

    print(
        "Not evaluated because building the "
        "graph requires Gemini."
    )

    print(
        "\nGemini used: False"
    )

    print(
        f"\nResults saved to:\n{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()