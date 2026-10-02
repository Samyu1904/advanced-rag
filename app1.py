from pathlib import Path

from ingestion.pdf_loader import PDFLoader
from ingestion.text_splitter import TextSplitter

from graph_rag.graph_builder import GraphBuilder
from graph_rag.graph_pipeline import GraphRAGPipeline


def load_and_build_graph():

    print("\n")
    print("=" * 70)
    print("GRAPH RAG INITIALIZATION")
    print("=" * 70)

    pdf_path = input(
        "\nEnter PDF path: "
    ).strip()

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():

        print(
            "\nPDF file not found."
        )

        return None

    if pdf_path.suffix.lower() != ".pdf":

        print(
            "\nPlease provide a PDF file."
        )

        return None

    print("\nLoading PDF...")

    loader = PDFLoader()

    documents = loader.load_pdf(
        str(pdf_path)
    )

    print("\nSplitting PDF text...")

    splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"\nChunks available for graph construction: "
        f"{len(chunks)}"
    )

    builder = GraphBuilder(
        batch_size=3
    )

    graph = builder.build_graph(
        chunks
    )

    builder.print_graph()

    pipeline = GraphRAGPipeline(
        graph=graph,
        top_k=10
    )

    return pipeline


def main():

    pipeline = load_and_build_graph()

    if pipeline is None:
        return

    print("\n")
    print("=" * 70)
    print("GRAPH RAG READY")
    print("=" * 70)

    print("\nCommands:")
    print("history  -> show conversation history")
    print("clear    -> clear conversation memory")
    print("exit     -> exit Graph RAG")

    while True:

        print("\n")

        question = input(
            "Ask your question: "
        ).strip()

        if not question:

            print(
                "\nPlease enter a question."
            )

            continue

        command = question.lower()

        if command == "exit":

            print(
                "\nExiting Graph RAG..."
            )

            break

        if command == "history":

            pipeline.show_history()

            continue

        if command == "clear":

            pipeline.clear_memory()

            continue

        pipeline.ask(question)


if __name__ == "__main__":
    main()