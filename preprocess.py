from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def clean_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return ""

    cleaned_text = raw_text.strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.replace("\x00", "")

    return cleaned_text


def load_large_csv(csv_path: str, chunk_size: int = 50000) -> pd.DataFrame:
    processed_chunks = []
    total_rows_processed = 0

    print("Starting to load dataset in chunks...\n")

    chunk_iterator = pd.read_csv(
        csv_path,
        usecols=["text", "generated"],
        chunksize=chunk_size,
    )

    for chunk_index, chunk_dataframe in enumerate(tqdm(chunk_iterator, desc="Processing chunks")):
        original_size = len(chunk_dataframe)

        chunk_dataframe = chunk_dataframe.dropna(subset=["text", "generated"]).copy()

        chunk_dataframe["text"] = chunk_dataframe["text"].astype(str).map(clean_text)
        chunk_dataframe["generated"] = pd.to_numeric(
            chunk_dataframe["generated"],
            errors="coerce",
        )

        chunk_dataframe = chunk_dataframe.dropna(subset=["generated"])
        chunk_dataframe["generated"] = chunk_dataframe["generated"].astype(int)

        chunk_dataframe = chunk_dataframe[
            chunk_dataframe["generated"].isin([0, 1])
        ].copy()

        chunk_dataframe = chunk_dataframe[
            chunk_dataframe["text"].str.len() > 0
        ].copy()

        processed_size = len(chunk_dataframe)
        total_rows_processed += processed_size

        print(
            f"Chunk {chunk_index + 1}: "
            f"{original_size} → {processed_size} rows | "
            f"Total processed: {total_rows_processed}"
        )

        processed_chunks.append(chunk_dataframe)

    print(f"\nFinished loading all chunks. Total rows: {total_rows_processed}")

    full_dataframe = pd.concat(processed_chunks, ignore_index=True)
    return full_dataframe


def preprocess_dataset(
    input_csv_path: str,
    output_directory: str = "dataset/processed_data",
    sample_size: int | None = None,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = load_large_csv(input_csv_path)

    print("\nFinished loading dataset.")
    print(f"Rows before duplicate removal: {len(dataframe)}")

    dataframe = dataframe.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"Rows after duplicate removal: {len(dataframe)}")
    print("\nClass distribution:")
    print(dataframe["generated"].value_counts())

    if sample_size is not None and sample_size < len(dataframe):
        dataframe = dataframe.groupby("generated", group_keys=False).apply(
            lambda group: group.sample(
                n=min(len(group), sample_size // 2),
                random_state=42,
            )
        ).reset_index(drop=True)

        print(f"\nApplied balanced sampling. New size: {len(dataframe)}")
        print(dataframe["generated"].value_counts())

    print("\nSplitting dataset...")

    train_dataframe, temp_dataframe = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        stratify=dataframe["generated"],
    )

    validation_dataframe, test_dataframe = train_test_split(
        temp_dataframe,
        test_size=0.5,
        random_state=42,
        stratify=temp_dataframe["generated"],
    )

    print("Saving files...")

    train_output_path = output_path / "train.csv"
    validation_output_path = output_path / "validation.csv"
    test_output_path = output_path / "test.csv"
    full_output_path = output_path / "full_cleaned.csv"

    dataframe.to_csv(full_output_path, index=False)
    train_dataframe.to_csv(train_output_path, index=False)
    validation_dataframe.to_csv(validation_output_path, index=False)
    test_dataframe.to_csv(test_output_path, index=False)

    print("\nDone ")
    print(f"- Full cleaned dataset: {full_output_path}")
    print(f"- Train set: {train_output_path}")
    print(f"- Validation set: {validation_output_path}")
    print(f"- Test set: {test_output_path}")


def create_subset_splits(
    dataframe: pd.DataFrame,
    subset_size: int,
    output_directory: str,
) -> None:
    print(f"\n=== Creating dataset with size: {subset_size} ===")

    # Ensure balanced sampling
    samples_per_class = subset_size // 2

    subset_dataframe = dataframe.groupby("generated", group_keys=False).apply(
        lambda group: group.sample(
            n=min(len(group), samples_per_class),
            random_state=42,
        )
    ).reset_index(drop=True)

    print("Subset class distribution:")
    print(subset_dataframe["generated"].value_counts())

    # Split: 80 train, 10 val, 10 test
    train_df, temp_df = train_test_split(
        subset_dataframe,
        test_size=0.2,
        random_state=42,
        stratify=subset_dataframe["generated"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["generated"],
    )

    # Create directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save files
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "validation.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print(f"Saved to {output_directory}")


if __name__ == "__main__":
    full_dataframe = load_large_csv("dataset/AI_Human.csv")

    print("\nRemoving duplicates...")
    full_dataframe = full_dataframe.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("\nFinal dataset distribution:")
    print(full_dataframe["generated"].value_counts())

    # Create multiple dataset sizes
    create_subset_splits(full_dataframe, 5000, "processed_data/dataset_5k")
    create_subset_splits(full_dataframe, 20000, "processed_data/dataset_20k")
    create_subset_splits(full_dataframe, 100000, "processed_data/dataset_100k")