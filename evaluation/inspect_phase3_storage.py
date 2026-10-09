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
                print(f"  {method_name}")


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Child Vector Store
    # --------------------------------------------------------

    from vectorstore.child_vectorstore import (
        ChildVectorStore
    )

    inspect_class(
        "ChildVectorStore",
        ChildVectorStore
    )

    # --------------------------------------------------------
    # Also inspect GraphBuilder source location
    # --------------------------------------------------------

    from graph_rag.graph_builder import GraphBuilder

    print("\n" + "=" * 70)
    print("GraphBuilder Source File")
    print("=" * 70)

    print(inspect.getfile(GraphBuilder))

    print("\n" + "=" * 70)
    print("GraphBuilder build_graph Source")
    print("=" * 70)

    try:
        print(
            inspect.getsource(
                GraphBuilder.build_graph
            )
        )
    except Exception as error:
        print(
            f"Could not read source: {error}"
        )

    print("\n" + "=" * 70)
    print("GraphBuilder extract_graph_from_batch Source")
    print("=" * 70)

    try:
        print(
            inspect.getsource(
                GraphBuilder.extract_graph_from_batch
            )
        )
    except Exception as error:
        print(
            f"Could not read source: {error}"
        )


if __name__ == "__main__":
    main()