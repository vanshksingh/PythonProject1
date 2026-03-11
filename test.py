import os
import json
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from datasets import Dataset


def load_ragas_dataset(file_path: str) -> Dataset:
    with open(file_path, "r") as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    return dataset


def run_ragas_evaluation(dataset: Dataset):
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )

    return result


if __name__ == "__main__":
    load_dotenv()

    dataset = load_ragas_dataset("test_ragas_dataset.json")

    results = run_ragas_evaluation(dataset)

    print(results)