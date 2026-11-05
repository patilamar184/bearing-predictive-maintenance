"""
Feature extraction utilities for the NASA IMS bearing vibration dataset.

Expected input:
- A whitespace-separated recording file.
- One column per vibration channel (the first IMS experiment normally has 8).
- One row per vibration sample.

The output feature names and order must match the feature_columns.json
exported from the training notebook.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def extract_features_from_dataframe(data: pd.DataFrame) -> dict:
    """Extract 8 statistical features from each vibration channel."""
    if data.empty:
        raise ValueError("The vibration recording is empty.")

    features = {}

    for channel_index in range(data.shape[1]):
        signal = pd.to_numeric(
            data.iloc[:, channel_index], errors="coerce"
        ).dropna().to_numpy(dtype=np.float64)

        if signal.size == 0:
            raise ValueError(
                f"Channel {channel_index + 1} contains no valid numeric samples."
            )

        rms = float(np.sqrt(np.mean(signal ** 2)))
        abs_peak = float(np.max(np.abs(signal)))

        values = {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "variance": float(np.var(signal)),
            "rms": rms,
            "skewness": float(skew(signal, bias=True)),
            "kurtosis": float(kurtosis(signal, bias=True)),
            "peak_to_peak": float(np.ptp(signal)),
            "crest_factor": float(abs_peak / (rms + 1e-10)),
        }

        for feature_name, value in values.items():
            features[f"ch{channel_index + 1}_{feature_name}"] = value

    return features


def extract_features_from_file(file_path: str | Path) -> dict:
    """Read one whitespace-separated vibration recording and extract features."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Recording file not found: {file_path}")

    data = pd.read_csv(file_path, sep=r"\s+", header=None)

    return extract_features_from_dataframe(data)
