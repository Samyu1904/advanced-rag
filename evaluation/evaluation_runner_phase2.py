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
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker import DocumentReranker


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_results_phase2.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    return dataset


# ============================================================
# CREATE CHUNK ID MAP
# ============================================================

def create_chunk_id_map(chunks):

    return {
        chunk.page_content: f"chunk_{index}"
        for index, chunk in enumerate(chunks)
    }


# ============================================================
# GET DOCUMENTS FROM RESULTS
# ============================================================

def extract_documents(results):

    documents = []

    if results is None:
        return documents

    if isinstance(results, tuple):

        results = results[0]

    if not isinstance(results, list):

        return documents

    for item in results:

        # (Document, score)
        if isinstance(item, tuple):

            if len(item) > 0:
                documents.append(item[0])

        # Document
        elif hasattr(item, "page_content"):

            documents.append(item)

    return documents


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k=3
):

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant = set(relevant_ids)

    hits = sum(
        1
        for chunk_id in retrieved
        if chunk_id in relevant
    )

    return hits / len(retrieved)


def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k=3
):

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]

    hits = sum(
        1
        for chunk_id in retrieved
        if chunk_id in relevant
    )

    return hits / len(relevant)


def reciprocal_rank(
    retrieved_ids,
    relevant_ids
):

    relevant = set(relevant_ids)

    for position, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in relevant:

            return 1.0 / position

    return 0.0


# ============================================================
# EVALUATE ONE RETRIEVER
# ============================================================

def evaluate_retriever(
    name,
    retriever_function,
    dataset,
    chunk_id_map
):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    results = []

    for item in dataset:

        question = item["question"]

        relevant_ids = item[
            "relevant_chunk_ids"
        ]

        print(
            f"\nQuestion: {question}"
        )

        try:

            raw_results = retriever_function(
                question
            )

            documents = extract_documents(
                raw_results
            )

            retrieved_ids = []

            for document in documents:

                chunk_id = chunk_id_map.get(
                    document.page_content
                )

                if chunk_id is not None:

                    retrieved_ids.append(
                        chunk_id
                    )

            precision = precision_at_k(
                retrieved_ids,
                relevant_ids,
                k=3
            )

            recall = recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=3
            )

            mrr = reciprocal_rank(
                retrieved_ids,
                relevant_ids
            )

            print(
                f"Retrieved: {retrieved_ids[:3]}"
            )

            print(
                f"Precision@3: {precision:.3f}"
            )

            print(
                f"Recall@3:    {recall:.3f}"
            )

            print(
                f"MRR:         {mrr:.3f}"
            )

            results.append(
                {
                    "question": question,
                    "retrieved_chunk_ids": retrieved_ids[:3],
                    "relevant_chunk_ids": relevant_ids,
                    "precision_at_3": precision,
                    "recall_at_3": recall,
                    "mrr": mrr
                }
            )

        except Exception as error:

            print(
                f"Evaluation failed: {error}"
            )

            results.append(
                {
                    "question": question,
                    "error": str(error)
                }
            )

    # ========================================================
    # AVERAGES
    # ========================================================

    valid_results = [
        result
        for result in results
        if "precision_at_3" in result
    ]

    if valid_results:

        avg_precision = sum(
            result["precision_at_3"]
            for result in valid_results
        ) / len(valid_results)

        avg_recall = sum(
            result["recall_at_3"]
            for result in valid_results
        ) / len(valid_results)

        avg_mrr = sum(
            result["mrr"]
            for result in valid_results
        ) / len(valid_results)

    else:

        avg_precision = 0.0
        avg_recall = 0.0
        avg_mrr = 0.0

    print("\nOverall Results")

    print(
        f"Precision@3: {avg_precision:.3f}"
    )

    print(
        f"Recall@3:    {avg_recall:.3f}"
    )

    print(
        f"MRR:         {avg_mrr:.3f}"
    )

    return {
        "rag": name,
        "precision_at_3": round(
            avg_precision,
            3
        ),
        "recall_at_3": round(
            avg_recall,
            3
        ),
        "mrr": round(
            avg_mrr,
            3
        ),
        "questions": results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PHASE 2 RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    print("\nPDF:")
    print(PDF_PATH)

    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------

    print("\nLoading PDF...")

    pdf_loader = PDFLoader()

    documents = pdf_loader.load_pdf(
        str(PDF_PATH)
    )

    # --------------------------------------------------------
    # SPLIT DOCUMENT
    # --------------------------------------------------------

    splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = load_dataset()

    print(
        f"Evaluation questions: {len(dataset)}"
    )

    chunk_id_map = create_chunk_id_map(
        chunks
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print(
        "\nBuilding FAISS vector store..."
    )

    vector_store = FAISSVectorStore()

    db = vector_store.create(
        chunks
    )

    faiss_retriever = RAGRetriever(
        vector_store=db,
        top_k=3,
        threshold=1.2
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 retriever..."
    )

    bm25_retriever = BM25Retriever(
        documents=chunks
    )

    # --------------------------------------------------------
    # RERANKER
    # --------------------------------------------------------

    print(
        "\nLoading reranker..."
    )

    reranker = DocumentReranker()

    # --------------------------------------------------------
    # ADVANCED RETRIEVAL
    #
    # IMPORTANT:
    # We intentionally do NOT use
    # AdvancedRerankingRetriever here because
    # it performs Gemini multi-query generation.
    #
    # Instead, we reproduce its retrieval stage:
    #
    # FAISS + BM25 -> candidate documents -> reranking
    # --------------------------------------------------------

    def advanced_retrieval(question):

        faiss_results = faiss_retriever.retrieve(
            question
        )

        bm25_results = bm25_retriever.retrieve(
            question
        )

        faiss_documents = extract_documents(
            faiss_results
        )

        bm25_documents = extract_documents(
            bm25_results
        )

        combined_documents = []

        seen = set()

        for document in (
            faiss_documents
            + bm25_documents
        ):

            content = document.page_content

            if content not in seen:

                seen.add(content)

                combined_documents.append(
                    document
                )

        # Rerank candidates

        reranked_documents = reranker.rerank(
            question,
            combined_documents
        )

        return reranked_documents[:3]

    # --------------------------------------------------------
    # CRAG RETRIEVAL
    #
    # We evaluate the retrieval component without
    # running the CRAG LLM evaluator/corrector.
    # --------------------------------------------------------

    def crag_retrieval(question):

        faiss_results = faiss_retriever.retrieve(
            question
        )

        bm25_results = bm25_retriever.retrieve(
            question
        )

        faiss_documents = extract_documents(
            faiss_results
        )

        bm25_documents = extract_documents(
            bm25_results
        )

        combined_documents = []

        seen = set()

        for document in (
            faiss_documents
            + bm25_documents
        ):

            content = document.page_content

            if content not in seen:

                seen.add(content)

                combined_documents.append(
                    document
                )

        reranked_documents = reranker.rerank(
            question,
            combined_documents
        )

        return reranked_documents[:3]

    # --------------------------------------------------------
    # SELF-RAG RETRIEVAL
    #
    # Self-RAG combines FAISS and BM25.
    # We evaluate that retrieval stage directly.
    # --------------------------------------------------------

    def self_rag_retrieval(question):

        faiss_results = faiss_retriever.retrieve(
            question
        )

        bm25_results = bm25_retriever.retrieve(
            question
        )

        faiss_documents = extract_documents(
            faiss_results
        )

        bm25_documents = extract_documents(
            bm25_results
        )

        combined_documents = []

        seen = set()

        for document in (
            faiss_documents
            + bm25_documents
        ):

            content = document.page_content

            if content not in seen:

                seen.add(content)

                combined_documents.append(
                    document
                )

        return combined_documents[:3]

    # ========================================================
    # RUN EVALUATIONS
    # ========================================================

    all_results = []

    all_results.append(
        evaluate_retriever(
            "Advanced RAG",
            advanced_retrieval,
            dataset,
            chunk_id_map
        )
    )

    all_results.append(
        evaluate_retriever(
            "CRAG",
            crag_retrieval,
            dataset,
            chunk_id_map
        )
    )

    all_results.append(
        evaluate_retriever(
            "Self-RAG",
            self_rag_retrieval,
            dataset,
            chunk_id_map
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "evaluation": "Phase 2 Retrieval Evaluation",
        "metrics": [
            "Precision@3",
            "Recall@3",
            "MRR"
        ],
        "gemini_used": False,
        "results": all_results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 60)
    print("PHASE 2 FINAL COMPARISON")
    print("=" * 60)

    print(
        f"\n{'RAG':<20}"
        f"{'Precision@3':<15}"
        f"{'Recall@3':<15}"
        f"{'MRR':<10}"
    )

    print("-" * 60)

    for result in all_results:

        print(
            f"{result['rag']:<20}"
            f"{result['precision_at_3']:<15.3f}"
            f"{result['recall_at_3']:<15.3f}"
            f"{result['mrr']:<10.3f}"
        )

    print("\nResults saved to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()