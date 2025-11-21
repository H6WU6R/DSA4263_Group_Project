"""Visualization helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")


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


def _strip_urls(text: str | float | None) -> str:
    """Remove URL-like substrings and coerce non-strings to empty strings."""

    if not isinstance(text, str):
        return ""
    return URL_PATTERN.sub("", text).strip()


def _plot_density_by_label(
    series: pd.Series,
    labels: pd.Series,
    label_names: Dict[int, str],
    title: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    """Render a density plot for the supplied series grouped by *labels*."""

    fig, ax = plt.subplots(figsize=(8, 5))
    for label_value, name in label_names.items():
        mask = labels == label_value
        data = series[mask].dropna()
        if data.empty:
            continue
        sns.kdeplot(
            data,
            ax=ax,
            label=name,
            linewidth=2,
            fill=True,
            alpha=0.25,
            common_norm=False,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_word_count_density_plots(
    csv_path: Path,
    output_dir: Path,
    *,
    label_column: str = "label",
    subject_column: str = "subject",
    body_column: str = "body",
    label_names: Iterable[tuple[int, str]] | None = None,
) -> Dict[str, Path]:
    """Generate density plots for subject/body word counts with URLs removed."""

    label_names = (
        dict(label_names)
        if label_names is not None
        else {0: "Legitimate", 1: "Phishing"}
    )

    df = pd.read_csv(csv_path, usecols=[subject_column, body_column, label_column])
    df["subject_clean"] = df[subject_column].map(_strip_urls)
    df["body_clean"] = df[body_column].map(_strip_urls)
    df["subject_word_count"] = df["subject_clean"].str.split().str.len()
    df["body_word_count"] = df["body_clean"].str.split().str.len()

    output_dir.mkdir(parents=True, exist_ok=True)

    subject_plot = _plot_density_by_label(
        df["subject_word_count"],
        df[label_column],
        label_names,
        "Subject Word Count Density (URLs Removed)",
        "Word Count",
        output_dir / "subject_word_count_density.png",
    )
    body_plot = _plot_density_by_label(
        df["body_word_count"],
        df[label_column],
        label_names,
        "Body Word Count Density (URLs Removed)",
        "Word Count",
        output_dir / "body_word_count_density.png",
    )

    return {"subject": subject_plot, "body": body_plot}
