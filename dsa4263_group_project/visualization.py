"""
Visualization utilities for EDA, feature analysis, and model comparison.

This module centralizes plotting helpers so that notebook visuals can be reused
as callable methods throughout the project.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from dsa4263_group_project.config import PROCESSED_DATA_DIR


class Visualizer:
    """Collection of reusable visualization routines."""

    COLOR_THEME = {
        "primary": "steelblue",
        "secondary": "coral",
        "accent": "mediumpurple",
        "success": "mediumseagreen",
        "danger": "indianred",
        "muted": "gray",
        "highlight": "gold",
    }

    PALETTES = {
        "bar": [
            "steelblue",
            "coral",
            "mediumseagreen",
            "mediumpurple",
            "gold",
            "gray",
            "lightsteelblue",
        ],
        "pie": [
            "lightblue",
            "lightcoral",
            "coral",
            "steelblue",
            "mediumpurple",
            "mediumseagreen",
            "gold",
            "gray",
        ],
        "line": ["navy", "darkorange", "forestgreen", "indigo", "crimson"],
        "label": {0: "mediumseagreen", 1: "indianred"},
        "feature_type": {
            "temporal": "steelblue",
            "url_domain": "mediumseagreen",
            "text_meta": "mediumpurple",
            "graph": "coral",
            "other": "gray",
        },
    }

    def __init__(self, output_dir: Optional[Path] = None, verbose: bool = True):
        """
        Args:
            output_dir: Directory to save plots. Defaults to PROCESSED_DATA_DIR.
            verbose: Print progress messages.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Provide a consistent base style for all plots
        sns.set_style("whitegrid")

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _palette(self, key: str, n: Optional[int] = None) -> List[str]:
        colors = self.PALETTES.get(key, [])
        if isinstance(colors, dict):
            palette = list(colors.values())
        else:
            palette = list(colors)

        if n is None or len(palette) >= n:
            return palette

        repeats = int(np.ceil(n / len(palette)))
        return (palette * repeats)[:n]

    def _label_color(self, label_value: int) -> str:
        return self.PALETTES["label"].get(label_value, self.COLOR_THEME["muted"])

    def _require_columns(self, df: pd.DataFrame, required: Sequence[str]) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self._log(f"  ✓ Saved to: {output_path}")
        return str(output_path)

    # --------------------------------------------------------------------- #
    # Core visualizations
    # --------------------------------------------------------------------- #
    def plot_label_distribution(
        self, df: pd.DataFrame, label_col: str = "label", filename: str = "label_distribution.png"
    ) -> str:
        """Bar + pie plot of label distribution (ham vs spam)."""
        self._require_columns(df, [label_col])
        self._log("\n[Visualization] Plotting label distribution...")

        label_counts = df[label_col].value_counts().sort_index()
        labels = label_counts.index.tolist()
        colors = [self._label_color(lbl) for lbl in labels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Email Class Distribution", fontsize=16, fontweight="bold")

        axes[0].bar([str(l) for l in labels], label_counts.values, color=colors, alpha=0.8)
        axes[0].set_ylabel("Count")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].pie(
            label_counts.values,
            labels=[f"{l}" for l in labels],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[1].set_title("Percentage")

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_text_length_distribution(
        self,
        df: pd.DataFrame,
        text_col: str = "email_text",
        label_col: str = "label",
        filename: str = "text_length_distribution.png",
    ) -> str:
        """Histogram of text lengths plus label count plot."""
        self._require_columns(df, [text_col, label_col])
        self._log("\n[Visualization] Plotting text length distribution...")

        lengths = df[text_col].fillna("").apply(lambda x: len(str(x).split()))
        colors = [self._label_color(0), self._label_color(1)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Email Text Length Analysis", fontsize=16, fontweight="bold")

        sns.countplot(x=label_col, data=df, palette=colors, ax=axes[0])
        axes[0].set_title("Label Distribution")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Count")
        axes[0].grid(axis="y", alpha=0.3)

        sns.histplot(lengths, bins=50, kde=True, color=self.COLOR_THEME["primary"], ax=axes[1])
        axes[1].set_title("Distribution of Email Text Length")
        axes[1].set_xlabel("Text Length (words)")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_top_token_frequency(
        self,
        legit_tokens: Sequence[Tuple[str, int]],
        spam_tokens: Sequence[Tuple[str, int]],
        level: str = "subject",
        filename: Optional[str] = None,
    ) -> str:
        """
        Visualize top token frequencies for legit vs spam segments.

        Args:
            legit_tokens: Sequence of (token, count) tuples for label 0.
            spam_tokens: Sequence of (token, count) tuples for label 1.
            level: Descriptor used in subplot titles (e.g., 'subject' or 'body').
            filename: Optional custom filename.
        """
        self._log(f"\n[Visualization] Plotting top {level} tokens...")

        if filename is None:
            filename = f"top_{level}_tokens.png"

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"Top {level.capitalize()} Tokens", fontsize=16, fontweight="bold")

        for ax, tokens, title, color in [
            (axes[0], legit_tokens, "Legit (0)", self._label_color(0)),
            (axes[1], spam_tokens, "Spam (1)", self._label_color(1)),
        ]:
            if not tokens:
                ax.set_visible(False)
                continue
            words, counts = zip(*tokens)
            ax.barh(range(len(words)), counts, color=color, alpha=0.8)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.invert_yaxis()
            ax.set_xlabel("Frequency")
            ax.set_title(f"Top 15 Words in {title}")
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_url_analysis(
        self,
        df: pd.DataFrame,
        url_col: str = "urls",
        label_col: str = "label",
        filename: str = "url_analysis.png",
    ) -> str:
        """Box plot + mean comparison of URL counts by label."""
        self._require_columns(df, [url_col, label_col])
        self._log("\n[Visualization] Plotting URL analysis by label...")

        url_stats = df.groupby(label_col)[url_col].agg(["mean", "median", "std", "min", "max"])
        colors = [self._label_color(0), self._label_color(1)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("URL Count by Email Type", fontsize=16, fontweight="bold")

        df.boxplot(column=url_col, by=label_col, ax=axes[0], grid=False, patch_artist=True)
        for patch, color in zip(axes[0].artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_xlabel("Label (0=Legit, 1=Spam)")
        axes[0].set_ylabel("Number of URLs")
        axes[0].set_title("URL Count Distribution")

        axes[1].bar(["Legit (0)", "Spam (1)"], url_stats["mean"].values, color=colors, alpha=0.8)
        axes[1].set_ylabel("Average URLs")
        axes[1].set_title("Average URL Count by Email Type")
        axes[1].grid(axis="y", alpha=0.3)

        plt.suptitle("")  # remove automatic pandas title
        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_time_distribution(
        self,
        df: pd.DataFrame,
        hour_col: str = "hour",
        day_name_col: str = "day_name",
        label_col: str = "label",
        filename: str = "time_distribution.png",
    ) -> str:
        """Grouped bar plots for hour-of-day and day-of-week distributions."""
        self._require_columns(df, [hour_col, day_name_col, label_col])
        self._log("\n[Visualization] Plotting hour/day distributions...")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle("Email Distribution Over Time", fontsize=16, fontweight="bold")
        colors = [self._label_color(0), self._label_color(1)]

        hour_df = df.groupby([hour_col, label_col]).size().unstack(fill_value=0)
        hour_df.plot(kind="bar", ax=axes[0], color=colors, alpha=0.8)
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel("Number of Emails")
        axes[0].set_title("Email Distribution by Hour")
        axes[0].legend(["Legit", "Spam"])
        axes[0].grid(axis="y", alpha=0.3)

        day_df = df.groupby([day_name_col, label_col]).size().unstack(fill_value=0)
        day_df.plot(kind="bar", ax=axes[1], color=colors, alpha=0.8)
        axes[1].set_xlabel("Day of Week")
        axes[1].set_ylabel("Number of Emails")
        axes[1].set_title("Email Distribution by Day of Week")
        axes[1].legend(["Legit", "Spam"])
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].set_xticklabels(day_df.index, rotation=45)

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_capital_ratio_distribution(
        self,
        df: pd.DataFrame,
        subject_col: str = "subject_capital_ratio",
        body_col: str = "body_capital_ratio",
        label_col: str = "label",
        filename: str = "capital_ratio_distribution.png",
    ) -> str:
        """Overlayed histograms for capital letter ratios in subject and body."""
        self._require_columns(df, [subject_col, body_col, label_col])
        self._log("\n[Visualization] Plotting capital ratio distributions...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Capital Letter Ratio Analysis", fontsize=16, fontweight="bold")

        for ax, col, title in [
            (axes[0], subject_col, "Subject"),
            (axes[1], body_col, "Body"),
        ]:
            ax.hist(
                df[df[label_col] == 0][col],
                bins=30,
                alpha=0.7,
                label="Legit",
                color=self._label_color(0),
            )
            ax.hist(
                df[df[label_col] == 1][col],
                bins=30,
                alpha=0.7,
                label="Spam",
                color=self._label_color(1),
            )
            ax.set_xlabel("Capital Letter Ratio")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{title} Capital Ratio Distribution")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_region_overview(
        self,
        df: pd.DataFrame,
        region_col: str = "timezone_region",
        label_col: str = "label",
        filename: str = "region_overview.png",
    ) -> str:
        """Region-level counts, rates, and stacked distribution."""
        self._require_columns(df, [region_col, label_col])
        self._log("\n[Visualization] Plotting region overview...")

        regional_stats = (
            df.groupby(region_col)[label_col]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "total", "sum": "spam_count", "mean": "spam_rate"})
            .sort_values("spam_rate", ascending=False)
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Region-Level Analysis", fontsize=16, fontweight="bold")

        regional_stats["total"].plot(kind="bar", ax=axes[0], color=self.COLOR_THEME["primary"])
        axes[0].set_title("Total Emails by Region")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Email Count")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(axis="y", alpha=0.3)

        regional_stats["spam_rate"].plot(kind="bar", ax=axes[1], color=self.COLOR_THEME["secondary"])
        axes[1].set_title("Spam Rate by Region")
        axes[1].set_xlabel("Region")
        axes[1].set_ylabel("Spam Rate")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].axhline(y=df[label_col].mean(), color="red", linestyle="--", label="Overall Average")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        region_label = df.groupby([region_col, label_col]).size().unstack(fill_value=0)
        region_label.plot(kind="bar", stacked=True, ax=axes[2], color=[self._label_color(0), self._label_color(1)])
        axes[2].set_title("Legit vs Spam by Region")
        axes[2].set_xlabel("Region")
        axes[2].set_ylabel("Email Count")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].legend(["Legit (0)", "Spam (1)"])
        axes[2].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_weekday_hour_heatmaps(
        self,
        df: pd.DataFrame,
        weekday_col: str = "day_of_week",
        hour_col: str = "hour",
        label_col: str = "label",
        filename: str = "weekday_hour_heatmaps.png",
    ) -> str:
        """Heatmaps for email volume and spam rate across weekday × hour."""
        self._require_columns(df, [weekday_col, hour_col, label_col])
        self._log("\n[Visualization] Plotting weekday × hour heatmaps...")

        weekday_hour_volume = df.pivot_table(values=label_col, index=weekday_col, columns=hour_col, aggfunc="count", fill_value=0)
        weekday_hour_rate = df.pivot_table(values=label_col, index=weekday_col, columns=hour_col, aggfunc="mean", fill_value=0)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        def _map_day(idx: object) -> str:
            try:
                return day_names[int(idx)]
            except (ValueError, TypeError, IndexError):
                return str(idx)

        weekday_hour_volume.index = [_map_day(i) for i in weekday_hour_volume.index]
        weekday_hour_rate.index = [_map_day(i) for i in weekday_hour_rate.index]

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle("Weekday × Hour Analysis", fontsize=16, fontweight="bold")

        sns.heatmap(weekday_hour_volume, annot=False, cmap="Blues", ax=axes[0], cbar_kws={"label": "Email Count"})
        axes[0].set_title("Email Volume by Weekday and Hour")
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel("Day of Week")

        sns.heatmap(
            weekday_hour_rate,
            annot=False,
            cmap="RdYlGn_r",
            ax=axes[1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Spam Rate"},
        )
        axes[1].set_title("Spam Rate by Weekday and Hour")
        axes[1].set_xlabel("Hour of Day")
        axes[1].set_ylabel("Day of Week")

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_month_region_heatmaps(
        self,
        df: pd.DataFrame,
        month_col: str = "month",
        region_col: str = "timezone_region",
        label_col: str = "label",
        filename: str = "month_region_heatmaps.png",
    ) -> str:
        """Heatmaps for month × region volume and spam rates."""
        self._require_columns(df, [month_col, region_col, label_col])
        self._log("\n[Visualization] Plotting month × region heatmaps...")

        month_region_volume = df.pivot_table(values=label_col, index=month_col, columns=region_col, aggfunc="count", fill_value=0)
        month_region_rate = df.pivot_table(values=label_col, index=month_col, columns=region_col, aggfunc="mean", fill_value=0)

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        def _map_month(idx: object) -> str:
            try:
                return month_names[int(idx) - 1]
            except (ValueError, TypeError, IndexError):
                return str(idx)

        month_region_volume.index = [_map_month(i) for i in month_region_volume.index]
        month_region_rate.index = [_map_month(i) for i in month_region_rate.index]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle("Month × Region Analysis", fontsize=16, fontweight="bold")

        sns.heatmap(
            month_region_volume,
            annot=True,
            fmt="g",
            cmap="Purples",
            ax=axes[0],
            cbar_kws={"label": "Email Count"},
        )
        axes[0].set_title("Email Volume by Month and Region")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Month")

        sns.heatmap(
            month_region_rate,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            ax=axes[1],
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Spam Rate"},
        )
        axes[1].set_title("Spam Rate by Month and Region")
        axes[1].set_xlabel("Region")
        axes[1].set_ylabel("Month")

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def plot_temporal_comparison(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        filename: str = "temporal_comparison.png",
    ) -> str:
        """Side-by-side temporal comparisons (hour/day/month hist + hour boxplot)."""
        self._require_columns(df, ["hour", "day_of_week", "month", label_col])
        self._log("\n[Visualization] Plotting temporal comparisons...")

        normal = df[df[label_col] == 0]
        spam = df[df[label_col] == 1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Temporal Comparison: Legit vs Spam", fontsize=16, fontweight="bold")
        colors = [self._label_color(0), self._label_color(1)]

        axes[0, 0].hist([normal["hour"], spam["hour"]], bins=24, label=["Legit", "Spam"], color=colors, alpha=0.8)
        axes[0, 0].set_title("Hour Distribution")
        axes[0, 0].set_xlabel("Hour of Day")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(
            [normal["day_of_week"], spam["day_of_week"]],
            bins=7,
            label=["Legit", "Spam"],
            color=colors,
            alpha=0.8,
        )
        axes[0, 1].set_title("Weekday Distribution")
        axes[0, 1].set_xlabel("Day of Week")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_xticks(range(0, 7))
        axes[0, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].hist(
            [normal["month"], spam["month"]],
            bins=12,
            label=["Legit", "Spam"],
            color=colors,
            alpha=0.8,
        )
        axes[1, 0].set_title("Month Distribution")
        axes[1, 0].set_xlabel("Month")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].boxplot([normal["hour"], spam["hour"]], labels=["Legit", "Spam"])
        axes[1, 1].set_title("Hour Distribution (Box Plot)")
        axes[1, 1].set_ylabel("Hour of Day")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        return self._save_fig(fig, filename)

    # --------------------------------------------------------------------- #
    # Existing analysis routines (refreshed with shared palettes)
    # --------------------------------------------------------------------- #
    def create_temporal_eda_plots(
        self, df: pd.DataFrame, filename: str = "temporal_eda_analysis.png"
    ) -> str:
        """Create comprehensive temporal EDA visualizations and save to output_dir."""
        self._require_columns(df, ["timezone_region", "label", "hour", "day_of_week", "sender", "is_weekend"])
        self._log("\n[Visualization] Creating temporal EDA plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Temporal Patterns in Email Spam Detection", fontsize=16, fontweight="bold")

        # Plot 1: Regional distribution
        ax1 = axes[0, 0]
        region_counts = df["timezone_region"].value_counts()
        ax1.bar(range(len(region_counts)), region_counts.values, color=self.COLOR_THEME["primary"], alpha=0.7)
        ax1.set_xticks(range(len(region_counts)))
        ax1.set_xticklabels(region_counts.index, rotation=45, ha="right")
        ax1.set_ylabel("Email Count")
        ax1.set_title("Email Distribution by Region")
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Spam rate by region
        ax2 = axes[0, 1]
        region_spam = df.groupby("timezone_region")["label"].mean().sort_values(ascending=False)
        ax2.bar(range(len(region_spam)), region_spam.values, color=self.COLOR_THEME["secondary"], alpha=0.7)
        ax2.set_xticks(range(len(region_spam)))
        ax2.set_xticklabels(region_spam.index, rotation=45, ha="right")
        ax2.set_ylabel("Spam Rate")
        ax2.set_title("Spam Rate by Region")
        ax2.axhline(y=df["label"].mean(), color="red", linestyle="--", alpha=0.5, label="Overall")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # Plot 3: Hourly patterns
        ax3 = axes[0, 2]
        hour_spam = df.groupby("hour")["label"].mean()
        ax3.plot(hour_spam.index, hour_spam.values, marker="o", linewidth=2, markersize=6, color=self.COLOR_THEME["primary"])
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("Spam Rate")
        ax3.set_title("Spam Rate by Hour")
        ax3.set_xticks(range(0, 24, 3))
        ax3.grid(alpha=0.3)

        # Plot 4: Weekday patterns
        ax4 = axes[1, 0]
        weekday_spam = df.groupby("day_of_week")["label"].mean()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ax4.bar(range(7), weekday_spam.values, color=self.COLOR_THEME["success"], alpha=0.7)
        ax4.set_xticks(range(7))
        ax4.set_xticklabels(day_names)
        ax4.set_ylabel("Spam Rate")
        ax4.set_title("Spam Rate by Day of Week")
        ax4.axhline(y=df["label"].mean(), color="red", linestyle="--", alpha=0.5)
        ax4.grid(axis="y", alpha=0.3)

        # Plot 5: Sender volume distribution
        ax5 = axes[1, 1]
        sender_stats = df.groupby("sender")["label"].count()
        sender_volumes = sender_stats.value_counts().sort_index()
        ax5.bar(sender_volumes.index[:20], sender_volumes.values[:20], color=self.COLOR_THEME["accent"], alpha=0.6)
        ax5.set_xlabel("Emails per Sender")
        ax5.set_ylabel("Number of Senders")
        ax5.set_title("Sender Volume Distribution (Top 20)")
        ax5.set_yscale("log")
        ax5.grid(alpha=0.3)

        # Plot 6: Weekend vs Weekday
        ax6 = axes[1, 2]
        weekend_data = df.groupby("is_weekend")["label"].mean()
        labels = ["Weekday", "Weekend"]
        colors_pie = [self.COLOR_THEME["primary"], self.COLOR_THEME["secondary"]]
        ax6.pie(
            [weekend_data.get(0, 0), weekend_data.get(1, 0)],
            labels=labels,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax6.set_title("Spam Rate: Weekday vs Weekend")

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def create_feature_importance_plots(
        self, importance_df: pd.DataFrame, filename: str = "feature_importance_analysis.png"
    ) -> str:
        """Create feature importance visualizations and save to output_dir."""
        self._require_columns(importance_df, ["feature", "importance", "type"])
        self._log("\n[Visualization] Creating feature importance plots...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Top 20 features
        ax1 = axes[0]
        top_20 = importance_df.head(20)
        type_palette = self.PALETTES["feature_type"]
        bar_colors = [type_palette.get(t, self.COLOR_THEME["muted"]) for t in top_20["type"]]

        bars = ax1.barh(range(len(top_20)), top_20["importance"].values, color=bar_colors, alpha=0.7)
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20["feature"].values)
        ax1.set_xlabel("Importance", fontweight="bold")
        ax1.set_title("Top 20 Most Important Features", fontweight="bold")
        ax1.invert_yaxis()
        ax1.grid(axis="x", alpha=0.3)

        legend_elements = [
            Patch(facecolor=type_palette["temporal"], label="Temporal"),
            Patch(facecolor=type_palette["url_domain"], label="URL/Domain"),
            Patch(facecolor=type_palette["text_meta"], label="Text Meta"),
            Patch(facecolor=type_palette["graph"], label="Graph"),
            Patch(facecolor=type_palette["other"], label="Other"),
        ]
        ax1.legend(handles=legend_elements, loc="lower right")

        # Plot 2: Feature type comparison
        ax2 = axes[1]
        type_importance = importance_df.groupby("type")["importance"].sum()
        colors_pie = [type_palette.get(t, self.COLOR_THEME["muted"]) for t in type_importance.index]
        ax2.pie(
            type_importance.values,
            labels=type_importance.index,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        ax2.set_title("Feature Type Contribution", fontweight="bold")

        plt.tight_layout()
        return self._save_fig(fig, filename)

    def create_model_comparison_plots(
        self, model_results: List[Dict], filename: str = "model_comparison_analysis.png"
    ) -> str:
        """Create model comparison visualizations and save to output_dir."""
        required_keys = ["Model", "Accuracy", "F1-Score", "Precision", "Recall", "Training Time (s)"]
        for record in model_results:
            missing = [k for k in required_keys if k not in record]
            if missing:
                raise ValueError(f"Model result missing keys: {missing}")

        self._log("\n[Visualization] Creating model comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Comparison Analysis", fontsize=16, fontweight="bold")

        models = [r["Model"] for r in model_results]
        accuracies = [r["Accuracy"] for r in model_results]
        f1_scores = [r["F1-Score"] for r in model_results]
        precisions = [r["Precision"] for r in model_results]
        recalls = [r["Recall"] for r in model_results]
        times = [r["Training Time (s)"] for r in model_results]

        colors_acc = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

        # Plot 1: Accuracy Comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(models)), accuracies, color=colors_acc, alpha=0.7)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy by Model")
        ax1.set_ylim([min(accuracies) * 0.95, 1.0])
        ax1.grid(axis="y", alpha=0.3)
        best_acc_idx = accuracies.index(max(accuracies))
        bars[best_acc_idx].set_edgecolor("red")
        bars[best_acc_idx].set_linewidth(3)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Plot 2: F1-Score Comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(models)), f1_scores, color=colors_acc, alpha=0.7)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha="right")
        ax2.set_ylabel("F1-Score")
        ax2.set_title("F1-Score by Model")
        ax2.set_ylim([min(f1_scores) * 0.95, 1.0])
        ax2.grid(axis="y", alpha=0.3)
        best_f1_idx = f1_scores.index(max(f1_scores))
        bars[best_f1_idx].set_edgecolor("red")
        bars[best_f1_idx].set_linewidth(3)
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Plot 3: Precision vs Recall
        ax3 = axes[1, 0]
        for i, (p, r, model) in enumerate(zip(precisions, recalls, models)):
            ax3.scatter(r, p, s=200, alpha=0.6, c=[colors_acc[i]], label=model)
            ax3.annotate(model, (r, p), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.set_title("Precision vs Recall Trade-off")
        ax3.grid(alpha=0.3)
        ax3.set_xlim([min(recalls) * 0.95, 1.0])
        ax3.set_ylim([min(precisions) * 0.95, 1.0])

        # Plot 4: Training Time Comparison
        ax4 = axes[1, 1]
        bars = ax4.bar(range(len(models)), times, color=colors_acc, alpha=0.7)
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=45, ha="right")
        ax4.set_ylabel("Training Time (seconds)")
        ax4.set_title("Training Time by Model")
        ax4.grid(axis="y", alpha=0.3)
        for i, v in enumerate(times):
            ax4.text(i, v + max(times) * 0.02, f"{v:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        return self._save_fig(fig, filename)
