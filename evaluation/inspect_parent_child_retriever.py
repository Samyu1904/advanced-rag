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
# IMPORT
# ============================================================

from retrieval.parent_child_retriever import (
    ParentChildRetriever
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PARENT-CHILD RETRIEVER SOURCE")
    print("=" * 70)

    print("\nSource file:")

    print(
        inspect.getfile(
            ParentChildRetriever
        )
    )

    print("\n")
    print("=" * 70)
    print("FULL CLASS SOURCE")
    print("=" * 70)

    print(
        inspect.getsource(
            ParentChildRetriever
        )
    )


if __name__ == "__main__":
    main()