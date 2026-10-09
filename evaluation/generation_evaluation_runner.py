import json
import sys
from pathlib import Path

from generation_evaluator import GenerationEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_evaluation_dataset.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results.json"
)


SAMPLE_ANSWERS = {
    "What are the health effects of air pollution?":
        "Air pollution can cause respiratory diseases, cardiovascular diseases, lung cancer, and other health problems.",

    "What are the causes of air pollution?":
        "Air pollution is caused by vehicles, industries, power plants, burning of fossil fuels, and other human activities.",

    "What measures can reduce exposure to air pollution?":
        "Exposure can be reduced by avoiding polluted areas, reducing outdoor activity when pollution is high, using protective measures, and reducing emissions.",

    "What is particulate matter?":
        "Particulate matter is a mixture of very small solid particles and liquid droplets suspended in the air.",

    "How does air pollution affect children?":
        "Air pollution can affect children's respiratory health, impair lung development, and increase the risk of respiratory illnesses."
}


def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("=" * 70)
    print("GENERATION EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    evaluator = GenerationEvaluator()

    results = []

    for item in dataset:

        question = item["question"]

        expected_answer = item[
            "expected_answer"
        ]

        answer = SAMPLE_ANSWERS.get(
            question,
            ""
        )

        evaluation = evaluator.evaluate(
            question=question,
            answer=answer,
            context=expected_answer,
            expected_answer=expected_answer
        )

        result = {
            "question": question,
            "answer": answer,
            "expected_answer": expected_answer,
            **evaluation
        }

        results.append(result)

        print("\n" + "-" * 70)
        print(question)

        print(
            f"Faithfulness: "
            f"{evaluation['faithfulness']:.3f}"
        )

        print(
            f"Answer Relevance: "
            f"{evaluation['answer_relevance']:.3f}"
        )

        print(
            f"Answer Correctness: "
            f"{evaluation['answer_correctness']:.3f}"
        )

    if results:

        average_faithfulness = sum(
            item["faithfulness"]
            for item in results
        ) / len(results)

        average_relevance = sum(
            item["answer_relevance"]
            for item in results
        ) / len(results)

        average_correctness = sum(
            item["answer_correctness"]
            for item in results
        ) / len(results)

    else:

        average_faithfulness = 0.0
        average_relevance = 0.0
        average_correctness = 0.0

    summary = {
        "faithfulness": round(
            average_faithfulness,
            3
        ),
        "answer_relevance": round(
            average_relevance,
            3
        ),
        "answer_correctness": round(
            average_correctness,
            3
        )
    }

    output = {
        "gemini_used": False,
        "results": results,
        "overall": summary
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

    print("\n")
    print("=" * 70)
    print("OVERALL GENERATION RESULTS")
    print("=" * 70)

    print(
        f"Faithfulness: "
        f"{summary['faithfulness']:.3f}"
    )

    print(
        f"Answer Relevance: "
        f"{summary['answer_relevance']:.3f}"
    )

    print(
        f"Answer Correctness: "
        f"{summary['answer_correctness']:.3f}"
    )

    print("\nGemini used: False")

    print("\nResults saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()