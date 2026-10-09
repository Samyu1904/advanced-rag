import sys
import inspect
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.parent_store import ParentStore


def main():

    print("=" * 70)
    print("PARENT STORE SOURCE")
    print("=" * 70)

    print("\nSource file:")

    print(
        inspect.getfile(
            ParentStore
        )
    )

    print("\n")
    print("=" * 70)
    print("FULL CLASS SOURCE")
    print("=" * 70)

    print(
        inspect.getsource(
            ParentStore
        )
    )


if __name__ == "__main__":
    main()