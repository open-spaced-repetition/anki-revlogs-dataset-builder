from huggingface_hub import snapshot_download  # type: ignore
from pathlib import Path

download_dir = Path("../anki-revlogs-10k")
download_dir.mkdir(exist_ok=True)

snapshot_download(
    repo_id="open-spaced-repetition/anki-revlogs-10k",
    repo_type="dataset",
    allow_patterns=[f"revlogs/user_id={i}/data.parquet" for i in range(1, 11)],
    local_dir=download_dir,
)
