from huggingface_hub import HfApi  # type: ignore
from pathlib import Path


def upload_dataset_to_hf(local_directory: Path, repo_name: str):
    api = HfApi()

    api.upload_large_folder(
        repo_id=repo_name,
        folder_path=local_directory,
        repo_type="dataset",
        allow_patterns="*.parquet",
    )

    print(f"Dataset uploaded successfully to {repo_name}")


upload_dataset_to_hf(
    local_directory=Path("../anki-revlogs-10k"),
    repo_name="open-spaced-repetition/anki-revlogs-10k",
)
