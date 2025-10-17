"""Visualization helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_distribution(frame: pd.DataFrame, column: str, output_path: Path) -> Path:
    """Create a simple histogram for *column* and save it to *output_path*."""

    fig, ax = plt.subplots(figsize=(8, 5))
    frame[column].hist(ax=ax, bins=30, edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
