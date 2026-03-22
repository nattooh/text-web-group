from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_split_data(train_csv_path: str, validation_csv_path: str, test_csv_path: str):
    train_dataframe = pd.read_csv(train_csv_path)
    validation_dataframe = pd.read_csv(validation_csv_path)
    test_dataframe = pd.read_csv(test_csv_path)

    required_columns = ["text", "generated"]

    for dataframe_name, dataframe in [
        ("train", train_dataframe),
        ("validation", validation_dataframe),
        ("test", test_dataframe),
    ]:
        for column_name in required_columns:
            if column_name not in dataframe.columns:
                raise ValueError(
                    f"Column '{column_name}' not found in {dataframe_name} dataframe. "
                    f"Available columns: {list(dataframe.columns)}"
                )

    train_dataframe = train_dataframe[required_columns].dropna().copy()
    validation_dataframe = validation_dataframe[required_columns].dropna().copy()
    test_dataframe = test_dataframe[required_columns].dropna().copy()

    for dataframe in [train_dataframe, validation_dataframe, test_dataframe]:
        dataframe["text"] = dataframe["text"].astype(str)
        dataframe["generated"] = dataframe["generated"].astype(int)

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


def evaluate_predictions(true_labels, predicted_labels, predicted_probabilities=None) -> dict:
    evaluation_results = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels).tolist(),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            target_names=["Human", "AI"],
            zero_division=0,
        ),
    }

    if predicted_probabilities is not None:
        evaluation_results["average_ai_probability"] = float(
            predicted_probabilities.mean()
        )

    return evaluation_results


def print_evaluation_results(model_name: str, evaluation_results: dict) -> None:
    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy : {evaluation_results['accuracy']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f}")
    print(f"Recall   : {evaluation_results['recall']:.4f}")
    print(f"F1-score : {evaluation_results['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(evaluation_results["confusion_matrix"])
    print("\nClassification Report:")
    print(evaluation_results["classification_report"])


def save_metrics(evaluation_results: dict, output_directory: str, file_name: str = "metrics.json") -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_output_path = output_path / file_name

    with open(metrics_output_path, "w", encoding="utf-8") as metrics_file:
        json.dump(evaluation_results, metrics_file, indent=4)

    print(f"Saved metrics to: {metrics_output_path}")


def save_sklearn_model(model_object, output_directory: str, file_name: str) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    model_output_path = output_path / file_name
    joblib.dump(model_object, model_output_path)

    print(f"Saved model to: {model_output_path}")