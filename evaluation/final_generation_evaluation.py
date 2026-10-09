import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from vectorstore.faiss_store import FAISSVectorStore
from vectorstore.child_vectorstore import ChildVectorStore

from retrieval.retriever import RAGRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import DocumentReranker
from retrieval.reranking_retriever import RerankingRetriever

from retrieval.parent_child_retriever import ParentChildRetriever
from retrieval.parent_store import ParentStore
from ingestion.parent_child_splitter import ParentChildSplitter

from retrieval.answer_generator import AnswerGenerator

from generation_evaluator import GenerationEvaluator


PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "air-quality-and-health.pdf"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_evaluation_dataset.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "final_generation_results.json"
)


def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def document_text(document):

    if hasattr(document, "page_content"):
        return document.page_content

    return str(document)


def extract_documents(results):

    documents = []

    if not results:
        return documents

    for item in results:

        if isinstance(item, tuple):

            documents.append(item[0])

        else:

            documents.append(item)

    return documents


def unique_documents(documents):

    unique = []
    seen = set()

    for document in documents:

        text = document_text(document)

        if text not in seen:

            seen.add(text)
            unique.append(document)

    return unique


def generate_answer(
    answer_generator,
    question,
    documents
):

    context = "\n\n".join(
        document_text(document)
        for document in documents
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY
the information provided in the retrieved
document context.

Do not use outside knowledge.

If the answer cannot be found in the
provided context, say:

"I could not find the answer in the
provided document."

Retrieved Context:

{context}

Question:

{question}

Answer:
"""

    response = answer_generator.llm.invoke(
        prompt
    )

    return answer_generator.extract_text(
        response
    )


def retrieve_basic(
    retriever,
    question
):

    results, relevant = retriever.search(
        question
    )

    return extract_documents(results)


def retrieve_hybrid(
    hybrid_retriever,
    question
):

    results = hybrid_retriever.retrieve(
        question
    )

    return extract_documents(results)


def retrieve_reranking(
    reranking_retriever,
    question
):

    results = reranking_retriever.retrieve(
        question
    )

    return extract_documents(results)


def retrieve_advanced(
    faiss_retriever,
    bm25_retriever,
    reranker,
    question
):

    faiss_results = faiss_retriever.retrieve(
        question
    )

    bm25_results = bm25_retriever.retrieve(
        question
    )

    candidates = []

    for document, score in faiss_results:
        candidates.append(document)

    for document in bm25_results:
        if isinstance(document, tuple):
            candidates.append(document[0])
        else:
            candidates.append(document)

    candidates = unique_documents(
        candidates
    )

    if not candidates:
        return []

    reranked = reranker.rerank(
        question,
        candidates
    )

    if isinstance(reranked, list):

        documents = []

        for item in reranked[:3]:

            if isinstance(item, tuple):
                documents.append(item[0])
            else:
                documents.append(item)

        return documents

    return candidates[:3]


def retrieve_crag(
    faiss_retriever,
    bm25_retriever,
    reranker,
    question
):

    faiss_results = faiss_retriever.retrieve(
        question
    )

    bm25_results = bm25_retriever.retrieve(
        question
    )

    candidates = []

    for document, score in faiss_results:
        candidates.append(document)

    for item in bm25_results:

        if isinstance(item, tuple):
            candidates.append(item[0])
        else:
            candidates.append(item)

    candidates = unique_documents(
        candidates
    )

    if not candidates:
        return []

    reranked = reranker.rerank(
        question,
        candidates
    )

    if isinstance(reranked, list):

        documents = []

        for item in reranked[:3]:

            if isinstance(item, tuple):
                documents.append(item[0])
            else:
                documents.append(item)

        return documents

    return candidates[:3]


def retrieve_self_rag(
    faiss_retriever,
    bm25_retriever,
    question
):

    faiss_results = faiss_retriever.retrieve(
        question
    )

    bm25_results = bm25_retriever.retrieve(
        question
    )

    candidates = []

    for document, score in faiss_results:
        candidates.append(document)

    for item in bm25_results:

        if isinstance(item, tuple):
            candidates.append(item[0])
        else:
            candidates.append(item)

    candidates = unique_documents(
        candidates
    )

    return candidates[:3]


def retrieve_parent_child(
    parent_child_retriever,
    question
):

    results = parent_child_retriever.retrieve(
        question
    )

    return extract_documents(results)


def evaluate_rag(
    rag_name,
    retrieve_function,
    dataset,
    answer_generator,
    evaluator
):

    print("\n")
    print("=" * 70)
    print(rag_name.upper())
    print("=" * 70)

    rag_results = []

    for item in dataset:

        question = item["question"]
        expected_answer = item[
            "expected_answer"
        ]

        print("\n" + "-" * 70)
        print(question)

        start_time = time.perf_counter()

        try:

            documents = retrieve_function(
                question
            )

            retrieval_time = (
                time.perf_counter()
                - start_time
            )

            answer_start = time.perf_counter()

            answer = generate_answer(
                answer_generator,
                question,
                documents
            )

            generation_time = (
                time.perf_counter()
                - answer_start
            )

            total_time = (
                time.perf_counter()
                - start_time
            )

            context = "\n\n".join(
                document_text(document)
                for document in documents
            )

            metrics = evaluator.evaluate(
                question=question,
                answer=answer,
                context=context,
                expected_answer=expected_answer
            )

            result = {
                "rag": rag_name,
                "question": question,
                "answer": answer,
                "retrieved_documents": len(
                    documents
                ),
                "faithfulness": metrics[
                    "faithfulness"
                ],
                "answer_relevance": metrics[
                    "answer_relevance"
                ],
                "answer_correctness": metrics[
                    "answer_correctness"
                ],
                "retrieval_latency_seconds":
                    round(
                        retrieval_time,
                        3
                    ),
                "generation_latency_seconds":
                    round(
                        generation_time,
                        3
                    ),
                "total_latency_seconds":
                    round(
                        total_time,
                        3
                    )
            }

            rag_results.append(result)

            print(
                f"Faithfulness: "
                f"{metrics['faithfulness']:.3f}"
            )

            print(
                f"Answer Relevance: "
                f"{metrics['answer_relevance']:.3f}"
            )

            print(
                f"Answer Correctness: "
                f"{metrics['answer_correctness']:.3f}"
            )

            print(
                f"Total Latency: "
                f"{total_time:.3f}s"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            rag_results.append({
                "rag": rag_name,
                "question": question,
                "error": str(error)
            })

    valid_results = [
        item
        for item in rag_results
        if "error" not in item
    ]

    if valid_results:

        overall = {
            "faithfulness": round(
                sum(
                    item["faithfulness"]
                    for item in valid_results
                )
                / len(valid_results),
                3
            ),
            "answer_relevance": round(
                sum(
                    item["answer_relevance"]
                    for item in valid_results
                )
                / len(valid_results),
                3
            ),
            "answer_correctness": round(
                sum(
                    item["answer_correctness"]
                    for item in valid_results
                )
                / len(valid_results),
                3
            ),
            "average_latency_seconds": round(
                sum(
                    item["total_latency_seconds"]
                    for item in valid_results
                )
                / len(valid_results),
                3
            )
        }

    else:

        overall = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "answer_correctness": 0.0,
            "average_latency_seconds": 0.0
        }

    print("\nOverall:")
    print(
        f"Faithfulness: "
        f"{overall['faithfulness']:.3f}"
    )
    print(
        f"Answer Relevance: "
        f"{overall['answer_relevance']:.3f}"
    )
    print(
        f"Answer Correctness: "
        f"{overall['answer_correctness']:.3f}"
    )
    print(
        f"Average Latency: "
        f"{overall['average_latency_seconds']:.3f}s"
    )

    return {
        "results": rag_results,
        "overall": overall
    }


def main():

    print("=" * 70)
    print("FINAL RAG GENERATION EVALUATION")
    print("=" * 70)

    print("\nPDF:")
    print(PDF_PATH)

    dataset = load_dataset()

    print(
        f"\nEvaluation questions: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------

    pdf_loader = PDFLoader()

    documents = pdf_loader.load_pdf(
        str(PDF_PATH)
    )

    text_splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"\nOriginal chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    vector_store = FAISSVectorStore()

    faiss_db = vector_store.create(
        chunks
    )

    faiss_retriever = RAGRetriever(
        vector_store=faiss_db,
        top_k=3,
        threshold=1.2
    )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(
        documents=chunks
    )

    # --------------------------------------------------
    # HYBRID
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        top_k=3
    )

    # --------------------------------------------------
    # RERANKING
    # --------------------------------------------------

    reranker = DocumentReranker()

    reranking_retriever = RerankingRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        candidate_k=10,
        top_k=3
    )

    # --------------------------------------------------
    # PARENT-CHILD
    # --------------------------------------------------

    parent_child_splitter = ParentChildSplitter()

    parents, children = (
        parent_child_splitter.split_documents(
            chunks
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

    # --------------------------------------------------
    # GEMINI
    # --------------------------------------------------

    answer_generator = AnswerGenerator()

    evaluator = GenerationEvaluator()

    all_results = {}

    # --------------------------------------------------
    # BASIC
    # --------------------------------------------------

    all_results["Basic RAG"] = evaluate_rag(
        "Basic RAG",
        lambda question:
            retrieve_basic(
                faiss_retriever,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # HYBRID
    # --------------------------------------------------

    all_results["Hybrid RAG"] = evaluate_rag(
        "Hybrid RAG",
        lambda question:
            retrieve_hybrid(
                hybrid_retriever,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # RERANKING
    # --------------------------------------------------

    all_results["Reranking RAG"] = evaluate_rag(
        "Reranking RAG",
        lambda question:
            retrieve_reranking(
                reranking_retriever,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # ADVANCED
    # --------------------------------------------------

    all_results["Advanced RAG"] = evaluate_rag(
        "Advanced RAG",
        lambda question:
            retrieve_advanced(
                faiss_retriever,
                bm25_retriever,
                reranker,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # CRAG
    # --------------------------------------------------

    all_results["CRAG"] = evaluate_rag(
        "CRAG",
        lambda question:
            retrieve_crag(
                faiss_retriever,
                bm25_retriever,
                reranker,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # SELF-RAG
    # --------------------------------------------------

    all_results["Self-RAG"] = evaluate_rag(
        "Self-RAG",
        lambda question:
            retrieve_self_rag(
                faiss_retriever,
                bm25_retriever,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # PARENT-CHILD
    # --------------------------------------------------

    all_results["Parent-Child RAG"] = evaluate_rag(
        "Parent-Child RAG",
        lambda question:
            retrieve_parent_child(
                parent_child_retriever,
                question
            ),
        dataset,
        answer_generator,
        evaluator
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    output = {
        "gemini_used": True,
        "evaluation_questions": len(dataset),
        "results": all_results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n")
    print("=" * 90)
    print("FINAL GENERATION EVALUATION SUMMARY")
    print("=" * 90)

    print(
        f"{'RAG':<22}"
        f"{'Faithfulness':<16}"
        f"{'Relevance':<14}"
        f"{'Correctness':<15}"
        f"{'Latency':<12}"
    )

    print("-" * 90)

    for rag_name, data in all_results.items():

        overall = data["overall"]

        print(
            f"{rag_name:<22}"
            f"{overall['faithfulness']:<16.3f}"
            f"{overall['answer_relevance']:<14.3f}"
            f"{overall['answer_correctness']:<15.3f}"
            f"{overall['average_latency_seconds']:<12.3f}"
        )

    print("\nGemini used: True")

    print("\nResults saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()