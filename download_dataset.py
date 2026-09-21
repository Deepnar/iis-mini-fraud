"""Download dataset (run with: uv run python download_dataset.py)."""
import kagglehub

path = kagglehub.dataset_download("ealtman2019/credit-card-transactions")
print("Path to dataset files:", path)
