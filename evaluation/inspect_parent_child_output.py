import sys
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
from ingestion.parent_child_splitter import ParentChildSplitter

from vectorstore.child_vectorstore import ChildVectorStore

from retrieval.parent_store import ParentStore
from retrieval.parent_child_retriever import ParentChildRetriever


# ============================================================
# PATH
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "air-quality-and-health.pdf"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INSPECTING PARENT-CHILD RETRIEVAL OUTPUT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    loader = PDFLoader()

    documents = loader.load_pdf(
        str(PDF_PATH)
    )

    # --------------------------------------------------------
    # Original chunks
    # --------------------------------------------------------

    splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    # --------------------------------------------------------
    # Parent-child splitting
    # --------------------------------------------------------

    pc_splitter = ParentChildSplitter()

    parents, children = (
        pc_splitter.split_documents(
            chunks
        )
    )

    print(
        f"\nParents: {len(parents)}"
    )

    print(
        f"Children: {len(children)}"
    )

    # --------------------------------------------------------
    # Print parent metadata
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PARENT METADATA")
    print("=" * 70)

    for index, parent in enumerate(parents):

        print(
            f"\nParent {index}"
        )

        print(
            "Type:",
            type(parent)
        )

        print(
            "Metadata:",
            getattr(
                parent,
                "metadata",
                None
            )
        )

        print(
            "Text:",
            parent.page_content[:300]
        )

    # --------------------------------------------------------
    # Print child metadata
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CHILD METADATA")
    print("=" * 70)

    for index, child in enumerate(children[:10]):

        print(
            f"\nChild {index}"
        )

        print(
            "Type:",
            type(child)
        )

        print(
            "Metadata:",
            getattr(
                child,
                "metadata",
                None
            )
        )

        print(
            "Text:",
            child.page_content[:300]
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
    # Retriever
    # --------------------------------------------------------

    retriever = ParentChildRetriever(
        child_vector_store=child_vector_store,
        parent_store=parent_store,
        top_k=3,
        threshold=1.2
    )

    # --------------------------------------------------------
    # Test question
    # --------------------------------------------------------

    question = (
        "What are the health effects "
        "of air pollution?"
    )

    print("\n")
    print("=" * 70)
    print("RETRIEVAL OUTPUT")
    print("=" * 70)

    print(
        f"\nQuestion: {question}"
    )

    result = retriever.retrieve(
        question
    )

    print(
        "\nResult type:",
        type(result)
    )

    print(
        "Number of results:",
        len(result)
        if isinstance(result, list)
        else "N/A"
    )

    # --------------------------------------------------------
    # Inspect every returned object
    # --------------------------------------------------------

    for index, item in enumerate(result):

        print("\n")
        print("-" * 70)

        print(
            f"RESULT {index}"
        )

        print(
            "Python type:",
            type(item)
        )

        print(
            "Raw object:"
        )

        print(
            repr(item)
        )

        # ----------------------------------------------------
        # Tuple
        # ----------------------------------------------------

        if isinstance(item, tuple):

            print(
                "\nTuple length:",
                len(item)
            )

            for tuple_index, value in enumerate(item):

                print(
                    f"\nTuple item {tuple_index}"
                )

                print(
                    "Type:",
                    type(value)
                )

                print(
                    "Value:",
                    repr(value)
                )

                if hasattr(
                    value,
                    "metadata"
                ):

                    print(
                        "Metadata:",
                        value.metadata
                    )

                if hasattr(
                    value,
                    "page_content"
                ):

                    print(
                        "Text:",
                        value.page_content[:500]
                    )

        # ----------------------------------------------------
        # Document
        # ----------------------------------------------------

        elif hasattr(
            item,
            "page_content"
        ):

            print(
                "\nMetadata:",
                item.metadata
            )

            print(
                "\nText:"
            )

            print(
                item.page_content[:1000]
            )

        # ----------------------------------------------------
        # Other object
        # ----------------------------------------------------

        else:

            print(
                "\nAttributes:"
            )

            print(
                dir(item)
            )


if __name__ == "__main__":
    main()