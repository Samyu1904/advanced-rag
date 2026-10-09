import sys
import inspect
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# HELPER
# ============================================================

def inspect_class(name, class_object):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\nConstructor:")

    try:
        print(inspect.signature(class_object))
    except Exception:
        print("Could not inspect constructor.")

    print("\nMethods:")

    for method_name, method in inspect.getmembers(
        class_object,
        predicate=inspect.isfunction
    ):

        if not method_name.startswith("_"):

            try:
                print(
                    f"  {method_name}"
                    f"{inspect.signature(method)}"
                )

            except Exception:

                print(
                    f"  {method_name}"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Parent-Child
    # --------------------------------------------------------

    from ingestion.parent_child_splitter import (
        ParentChildSplitter
    )

    from retrieval.parent_store import (
        ParentStore
    )

    from retrieval.parent_child_retriever import (
        ParentChildRetriever
    )

    # --------------------------------------------------------
    # Graph RAG
    # --------------------------------------------------------

    from graph_rag.graph_builder import (
        GraphBuilder
    )

    from graph_rag.graph_retriever import (
        GraphRetriever
    )

    from graph_rag.graph_pipeline import (
        GraphRAGPipeline
    )

    # --------------------------------------------------------
    # Inspect Parent-Child
    # --------------------------------------------------------

    inspect_class(
        "ParentChildSplitter",
        ParentChildSplitter
    )

    inspect_class(
        "ParentStore",
        ParentStore
    )

    inspect_class(
        "ParentChildRetriever",
        ParentChildRetriever
    )

    # --------------------------------------------------------
    # Inspect Graph RAG
    # --------------------------------------------------------

    inspect_class(
        "GraphBuilder",
        GraphBuilder
    )

    inspect_class(
        "GraphRetriever",
        GraphRetriever
    )

    inspect_class(
        "GraphRAGPipeline",
        GraphRAGPipeline
    )


if __name__ == "__main__":
    main()