import sys
import inspect
from pathlib import Path


# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def inspect_class(class_name, class_object):
    print("\n" + "=" * 70)
    print(class_name)
    print("=" * 70)

    print("\nConstructor:")
    print(inspect.signature(class_object))

    print("\nMethods:")

    for name, method in inspect.getmembers(
        class_object,
        predicate=inspect.isfunction
    ):
        if not name.startswith("_"):
            try:
                print(f"  {name}{inspect.signature(method)}")
            except (ValueError, TypeError):
                print(f"  {name}")


def main():

    from advanced.advanced_reranking_retriever import (
        AdvancedRerankingRetriever
    )

    from advanced.advanced_pipeline import (
        AdvancedRAGPipeline
    )

    from crag.crag_retriever import CRAGRetriever

    from crag.crag_pipeline import CRAGPipeline

    from self_rag.self_rag_retriever import SelfRAGRetriever

    from self_rag.self_rag_pipeline import SelfRAGPipeline

    from agentic_rag.agentic_rag_agent import AgenticRAGAgent

    from agentic_rag.agentic_rag_pipeline import AgenticRAGPipeline

    inspect_class(
        "AdvancedRerankingRetriever",
        AdvancedRerankingRetriever
    )

    inspect_class(
        "AdvancedRAGPipeline",
        AdvancedRAGPipeline
    )

    inspect_class(
        "CRAGRetriever",
        CRAGRetriever
    )

    inspect_class(
        "CRAGPipeline",
        CRAGPipeline
    )

    inspect_class(
        "SelfRAGRetriever",
        SelfRAGRetriever
    )

    inspect_class(
        "SelfRAGPipeline",
        SelfRAGPipeline
    )

    inspect_class(
        "AgenticRAGAgent",
        AgenticRAGAgent
    )

    inspect_class(
        "AgenticRAGPipeline",
        AgenticRAGPipeline
    )


if __name__ == "__main__":
    main()