# Anki Revlogs Dataset Builder

A tool for processing Anki spaced repetition review logs using Protocol Buffers and Parquet files.

## Overview

This project provides utilities to:
- Convert Anki review logs from Protocol Buffer format to Parquet files
- Upload/download datasets to/from Hugging Face Hub
- Show example of how to use the dataset

The converted dataset is available on Hugging Face Hub: [Anki Revlogs 10K](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k)

## Installation

Required dependencies:
```
pandas
pyarrow
protobuf
huggingface_hub
tqdm
```

## Usage

### Converting Review Logs to Parquet

Use the build_parquet.py script to convert Protocol Buffer files to Parquet format:

```python
python build_parquet.py
```

### Downloading Dataset from Hugging Face

To download the processed dataset:

```python
python download_from_hf.py
```

### Processing Individual User Data

To analyze data for a specific user:

```python
python process_dataset.py
```

## Data Processing Pipeline

1. Raw review logs are read from .revlog files (Protocol Buffer format)
2. Data is processed and transformed using pandas DataFrames
3. Results are saved as Parquet files partitioned by user_id
4. Upload to Hugging Face Hub for sharing

## File Structure

```
.
├── README.md
├── build_parquet.py    # Main conversion script
├── stats.proto         # Protocol Buffer definition
├── download_from_hf.py # Dataset download utility
├── upload_to_hf.py     # Dataset upload utility
└── process_dataset.py  # Individual data processing
```

## License

GNU AGPL, version 3 or later
