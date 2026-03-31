"""Load and combine datasets from CSV files."""

from pathlib import Path
import pandas as pd


def load_all_csv(csv_path: str = "data/All.csv", **kwargs) -> pd.DataFrame:
	"""Load the combined data file and return a DataFrame."""
	path = Path(csv_path)
	df = pd.read_csv(path, **kwargs)
	return df


def save_csv(df: pd.DataFrame, csv_path: str, index: bool = False) -> None:
	"""Save a DataFrame to a CSV path."""
	path = Path(csv_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(path, index=index)
