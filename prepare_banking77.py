from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"


def prepare_banking77(tokenizer=None, *, raw_train=None) -> DatasetDict:
    raw = raw_train
    if raw is None:
        raw = load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            data_files={"train": "data/train-00000-of-00001.parquet"},
            split="train",
            verification_mode="no_checks",
        )

    # Remove duplicates before splitting so identical questions cannot cross splits.
    seen = set()
    keep = []
    for index, text in enumerate(raw["text"]):
        key = " ".join(text.lower().split())
        if key not in seen:
            seen.add(key)
            keep.append(index)

    split = raw.select(keep).train_test_split(
        test_size=0.10, stratify_by_column="label", seed=42
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    label_names = raw.features["label"].names
    candidate_labels = ", ".join(label_names)

    def make_example(row):
        question = (
            "Classify the banking customer message. "
            "Reply with exactly one candidate label.\n\n"
            f"Candidate labels: {candidate_labels}\n\n"
            f"Customer message: {row['text']}"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": prompt, "completion": label_names[row["label"]]}

    partitions = DatasetDict({"train": split["train"], "validation": split["test"]})
    prepared = partitions.map(make_example, remove_columns=raw.column_names)
    prepared["validation"] = prepared["validation"].add_column("question", list(split["test"]["text"]))
    return prepared


def main():
    partitions = prepare_banking77()
    train: Dataset = partitions["train"]
    print(f"Train: {len(train)}")
    print(f"Validation: {len(partitions['validation'])}")
    print(f"First training answer: {train[0]['completion']}")


if __name__ == "__main__":
    main()
