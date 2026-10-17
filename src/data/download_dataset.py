from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_NAME = (
    "hellotayssir/"
    "FinQA_TAT-QA_financial_finetuning_dataset"
)

RAW_DIR = Path("data/raw/financial_dataset")


def main():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading dataset...")

    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        local_dir=str(RAW_DIR)
    )

    print(
        f"Dataset downloaded to: {RAW_DIR}"
    )


if __name__ == "__main__":
    main()