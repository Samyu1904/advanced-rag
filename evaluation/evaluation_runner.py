import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import DocumentReranker
from retrieval.reranking_retriever import RerankingRetriever

from evaluation.retrieval_evaluator import RetrievalEvaluator, find_chunk_id

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "air-quality-and-health.pdf"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "evaluation_dataset.json"


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        raise SystemExit(1)

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if any(not item["relevant_chunk_ids"] for item in dataset):
        print("Ground-truth chunk IDs are empty.")
        print("First run: python evaluation\\inspect_chunks.py")
        print("Then fill evaluation\\evaluation_dataset.json.")
        raise SystemExit(1)

    documents = PDFLoader().load_pdf(str(PDF_PATH))
    chunks = TextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

    vector_store = FAISSVectorStore().create(chunks)

    faiss_retriever = RAGRetriever(vector_store=vector_store, top_k=3, threshold=1.2)
    bm25_retriever = BM25Retriever(documents=chunks)
    hybrid_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        top_k=3,
    )
    reranker = DocumentReranker()
    reranking_retriever = RerankingRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        candidate_k=10,
        top_k=3,
    )

    retrievers = {
        "Basic RAG": faiss_retriever,
        "Hybrid RAG": hybrid_retriever,
        "Reranking RAG": reranking_retriever,
    }

    all_results = {}

    for rag_name, retriever in retrievers.items():
        question_results = []

        for item in dataset:
            raw_results = retriever.retrieve(item["question"])
            if raw_results is None:
                raw_results = []
            if isinstance(raw_results, tuple):
                raw_results = list(raw_results)

            retrieved_ids = []
            for result in raw_results:
                chunk_id = find_chunk_id(result, chunks)
                if chunk_id is not None:
                    retrieved_ids.append(chunk_id)

            metrics = RetrievalEvaluator.evaluate_question(
                retrieved_ids,
                item["relevant_chunk_ids"],
                k=3,
            )

            question_results.append({
                "question": item["question"],
                "retrieved_ids": retrieved_ids,
                "relevant_ids": item["relevant_chunk_ids"],
                **metrics,
            })

        all_results[rag_name] = {
            "questions": question_results,
            "aggregate": RetrievalEvaluator.aggregate(question_results),
        }

        aggregate = all_results[rag_name]["aggregate"]
        print("\n" + "=" * 60)
        print(rag_name)
        print("=" * 60)
        print(f"Precision@3: {aggregate['precision_at_k']:.3f}")
        print(f"Recall@3:    {aggregate['recall_at_k']:.3f}")
        print(f"MRR:         {aggregate['mrr']:.3f}")

    output = PROJECT_ROOT / "evaluation" / "retrieval_results.json"
    output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
