"""
Heterogeneous GNN for Email Spam Detection
===========================================

This script implements a heterogeneous Graph Neural Network (R-GCN/HAN) 
for spam email detection based on network structure and text features.

Graph Schema:
- Nodes: Email, Sender, Receiver
- Edges: Sender --sent--> Email, Email --to--> Receiver
- Optional: Email --similar_subject--> Email

Node Features:
- Email: TF-IDF of subject, url_count, body_len
- Sender/Receiver: degree, unique partners, first_seen_days, send_rate_7d/30d, last_seen_days

Model: R-GCN or HAN
Target: Classify Email nodes (spam/ham)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BANNER = """
============================================================
HETEROGENEOUS GNN FOR EMAIL SPAM DETECTION
============================================================
"""


# -----------------------------
# Path utilities
# -----------------------------
def detect_project_root(script_path: Path) -> Path:
    """
    Heuristic: project root is parent of 'notebooks' folder that contains this script.
    Fallback to script's parent if structure is different.
    """
    if script_path.parent.name.lower() == "notebooks":
        return script_path.parent.parent
    return script_path.parent


def build_default_paths(script_path: Path) -> Tuple[Path, Path, Path]:
    root = detect_project_root(script_path)
    data_csv = root / "data" / "processed" / "combined_phishing_data.csv"
    graph_pkl = root / "data" / "processed" / "graph_data.pkl"
    out_dir = root / "outputs"
    return data_csv, graph_pkl, out_dir


# -----------------------------
# IO helpers
# -----------------------------
def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\n"
            f"CWD: {Path.cwd()}\n"
            "Tip: run `python notebooks/test.py` from repo root, or pass --data-path."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def try_load_graph(graph_path: Path) -> Optional[object]:
    if graph_path.exists():
        import pickle

        with open(graph_path, "rb") as f:
            return pickle.load(f)
    return None


# -----------------------------
# Sanity checks
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.encode("utf-8", "ignore").str.decode("utf-8")
    )
    df.columns = df.columns.str.strip()
    return df


def infer_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Try to guess a reasonable text column and label column.
    You can also specify them via CLI flags if your names differ.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    # Common text fields
    for k in ("text", "body", "message", "content", "email_text", "clean_text"):
        if k in cols_lower:
            text_col = cols_lower[k]
            break
    else:
        # Fallback: pick the longest average string column
        object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not object_cols:
            raise ValueError(
                "Could not find a text-like column. Pass --text-col explicitly."
            )
        avg_len = (
            df[object_cols]
            .astype("string")
            .apply(lambda s: s.str.len().fillna(0).mean())
            .sort_values(ascending=False)
        )
        text_col = avg_len.index[0]

    # Common labels
    label_candidates = (
        "label",
        "is_spam",
        "spam",
        "target",
        "y",
        "class",
    )
    label_col = None
    for k in label_candidates:
        if k in cols_lower:
            label_col = cols_lower[k]
            break

    return text_col, label_col


def basic_report(df: pd.DataFrame, text_col: str, label_col: Optional[str]) -> str:
    rep = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "text_col": text_col,
        "label_col": label_col,
        "null_text_pct": float(df[text_col].isna().mean()) if text_col in df else None,
        "columns": df.columns.tolist(),
    }
    return json.dumps(rep, indent=2)


# -----------------------------
# Baseline (TF-IDF + LogisticRegression)
# -----------------------------
def run_text_baseline(
    df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str],
    limit: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Quick, lightweight baseline to confirm data is readable and roughly classifiable.
    If no labels are present, it only vectorizes and reports shape.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    data = df.copy()
    if limit:
        data = data.iloc[:limit].copy()

    X_text = data[text_col].astype("string").fillna("")

    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        strip_accents="unicode",
    )
    X = vect.fit_transform(X_text)

    print(f"[Baseline] TF-IDF matrix shape: {X.shape}")

    if label_col is None:
        print(
            "[Baseline] No label column detected; skipping classification. "
            "Pass --label-col if you have one."
        )
        return

    y = data[label_col].astype("category").cat.codes.values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        class_weight="balanced",
        solver="liblinear",
    )
    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_te)
    print("\n[Baseline] Classification report:")
    print(classification_report(y_te, y_pr, digits=4))


# -----------------------------
# CLI & main
# -----------------------------
def parse_args(script_path: Path) -> argparse.Namespace:
    default_csv, default_graph, default_out = build_default_paths(script_path)
    p = argparse.ArgumentParser(description="Spam detection pipeline bootstrap")
    p.add_argument(
        "--data-path",
        type=Path,
        default=default_csv,
        help="Path to CSV (default: data/processed/combined_phishing_data.csv relative to project root).",
    )
    p.add_argument(
        "--graph-path",
        type=Path,
        default=default_graph,
        help="Optional graph pickle path (default: data/processed/graph_data.pkl).",
    )
    p.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Name of the text column. If omitted, the script will try to infer it.",
    )
    p.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Name of the label column. If omitted, the script will try to infer it.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows for a quick run (e.g., 20000).",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the TF-IDF baseline (useful when you only want IO checks).",
    )
    return p.parse_args()


def main():
    print(BANNER)
    print("Loading data...")

    # Resolve script path (robust to symlinks)
    script_path = Path(__file__).resolve()
    args = parse_args(script_path)

    # Load CSV
    df = load_dataframe(args.data_path)
    df = normalize_columns(df)

    # Detect columns if not provided
    text_col, guessed_label = infer_text_and_label_columns(df)
    if args.text_col:
        text_col = args.text_col
        if text_col not in df.columns:
            raise KeyError(f"--text-col '{text_col}' not in columns: {df.columns.tolist()}")

    label_col = args.label_col or guessed_label
    if args.label_col and label_col not in df.columns:
        raise KeyError(f"--label-col '{label_col}' not in columns: {df.columns.tolist()}")

    # Basic report
    print("\n[Data Report]")
    print(basic_report(df, text_col, label_col))

    # Try to load graph (optional)
    graph_obj = try_load_graph(args.graph_path)
    if graph_obj is not None:
        print(f"\n[Graph] Loaded graph object from: {args.graph_path}")
    else:
        print(f"\n[Graph] No graph pickle found at: {args.graph_path} (this is fine for now)")

    # Quick baseline
    if not args.no_baseline:
        print("\nRunning quick text baseline (TF-IDF + LogisticRegression)...")
        run_text_baseline(
            df, text_col=text_col, label_col=label_col, limit=args.limit
        )
    else:
        print("\nSkipping baseline as requested (--no-baseline).")

    print("\nDone. You can now plug your GNN code after the baseline section.")


if __name__ == "__main__":
    main()
