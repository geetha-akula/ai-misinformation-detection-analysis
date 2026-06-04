#!/usr/bin/env python
"""
Comprehensive analytics pipeline for the generative AI misinformation dataset.

The script performs:
  * Exploratory data analysis and descriptive statistics.
  * Training / evaluation of baseline (logistic regression) and tree-based (random forest)
    classifiers to estimate misinformation risk.
  * Risk-based prioritisation analysis to approximate optimal allocation of manual
    fact-checking capacity.
  * Generation of publication-quality figures and structured summary metrics for reporting.

All outputs are written to the `reports/` directory so that they can be embedded directly
in project documentation (e.g., README.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler


DATA_PATH = Path("genai-dataset/generative_ai_misinformation_dataset.csv")
FIGURES_DIR = Path("reports/figures")
SUMMARIES_PATH = Path("reports/metrics_summary.json")
RISK_COVERAGE_CSV = Path("reports/risk_coverage_curve.csv")
CATEGORY_COLORS = {"Legitimate": "#4C72B0", "Misinformation": "#DD8452"}
DEFAULT_COLOR = "#4C72B0"
TEXT_COLUMN = "text"
TFIDF_MAX_FEATURES = 500
TEXT_SVD_COMPONENTS = 50
NUMERIC_DISTRIBUTION_SPECS = [
    ("author_followers_log10", "Author followers (log10 scale)", 30),
    ("engagement_log10", "Engagement (log10 scale)", 30),
    ("text_length", "Text length (characters)", 30),
    ("token_count", "Token count", 30),
    ("readability_score", "Readability score", 30),
    ("num_urls", "Number of URLs", 15),
    ("num_mentions", "Number of mentions", 15),
    ("num_hashtags", "Number of hashtags", 15),
    ("sentiment_score", "Sentiment score", 30),
    ("toxicity_score", "Toxicity score", 30),
    ("detected_synthetic_score", "Detected synthetic score", 30),
    ("embedding_sim_to_facts", "Embedding similarity to facts", 30),
    ("external_factchecks_count", "External fact-check count", 15),
    ("source_domain_reliability", "Source domain reliability", 30),
]
BOXPLOT_CATEGORY_SPECS = [
    {
        "feature": "sentiment_score",
        "category": "platform",
        "pretty_feature": "Sentiment score",
        "pretty_category": "Platform",
        "filename": "sentiment_by_platform.png",
        "order": ["Facebook", "Reddit", "Telegram", "Twitter"],
    },
    {
        "feature": "toxicity_score",
        "category": "platform",
        "pretty_feature": "Toxicity score",
        "pretty_category": "Platform",
        "filename": "toxicity_by_platform.png",
        "order": ["Facebook", "Reddit", "Telegram", "Twitter"],
    },
    {
        "feature": "engagement_log10",
        "category": "platform",
        "pretty_feature": "Engagement (log10 scale)",
        "pretty_category": "Platform",
        "filename": "engagement_by_platform.png",
        "order": ["Facebook", "Reddit", "Telegram", "Twitter"],
    },
    {
        "feature": "text_length",
        "category": "platform",
        "pretty_feature": "Text length (characters)",
        "pretty_category": "Platform",
        "filename": "text_length_by_platform.png",
        "order": ["Facebook", "Reddit", "Telegram", "Twitter"],
    },
    {
        "feature": "author_followers_log10",
        "category": "platform",
        "pretty_feature": "Author followers (log10 scale)",
        "pretty_category": "Platform",
        "filename": "author_followers_by_platform.png",
        "order": ["Facebook", "Reddit", "Telegram", "Twitter"],
    },
]


@dataclass
class ModelArtifacts:
    """Container for model outputs."""

    pipeline: Pipeline
    metrics: Dict[str, Dict[str, float]]
    figure_paths: List[str]
    feature_importances: pd.DataFrame


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and minimally preprocess the dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["weekday"] = pd.Categorical(df["weekday"], categories=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ], ordered=True)
    df["month"] = pd.Categorical(df["month"], categories=[
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ], ordered=True)
    df["hour"] = df["timestamp"].dt.hour
    df["label_name"] = pd.Categorical(
        df["is_misinformation"].map({0: "Legitimate", 1: "Misinformation"}),
        categories=["Legitimate", "Misinformation"],
        ordered=True,
    )
    df["author_followers_log10"] = np.log10(df["author_followers"] + 1)
    df["engagement_log10"] = np.log10(df["engagement"] + 1)
    return df


def ensure_output_dirs() -> None:
    """Create required directories for outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Legitimate", "Misinformation"]
    counts = df["label_name"].value_counts().reindex(order)
    sns.countplot(
        data=df,
        x="label_name",
        order=order,
        color=DEFAULT_COLOR,
        ax=ax,
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of posts")
    ax.set_title("Class distribution")
    for idx, (label, count) in enumerate(counts.items()):
        ax.text(idx, count + 5, f"{count}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "class_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_numeric_histogram(
    df: pd.DataFrame,
    feature: str,
    pretty_name: str,
    filename: str,
    bins: int = 30,
    kde: bool = True,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        df[feature],
        bins=bins,
        kde=kde,
        color="#4C72B0",
        ax=ax,
    )
    ax.set_xlabel(pretty_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {pretty_name.lower()}")
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_numeric_histogram_by_label(
    df: pd.DataFrame,
    feature: str,
    pretty_name: str,
    filename: str,
    bins: int = 30,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=feature,
        hue="label_name",
        bins=bins,
        stat="density",
        common_norm=False,
        element="step",
        palette=CATEGORY_COLORS,
        ax=ax,
    )
    ax.set_xlabel(pretty_name)
    ax.set_ylabel("Density")
    ax.set_title(f"{pretty_name} by misinformation label")
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_numeric_box_by_category(
    df: pd.DataFrame,
    feature: str,
    category: str,
    pretty_feature: str,
    pretty_category: str,
    filename: str,
    order: List[str] | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if order is not None:
        n_colors = len(order)
    else:
        n_colors = df[category].nunique()
    palette = sns.color_palette("Set2", n_colors)
    sns.boxplot(
        data=df,
        x=category,
        y=feature,
        order=order,
        hue=category,
        palette=palette,
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel(pretty_category)
    ax.set_ylabel(pretty_feature)
    ax.set_title(f"{pretty_feature} by {pretty_category.lower()}")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_sentiment_toxicity_scatter(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="sentiment_score",
        y="toxicity_score",
        hue="label_name",
        palette=CATEGORY_COLORS,
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Sentiment score")
    ax.set_ylabel("Toxicity score")
    ax.set_title("Sentiment vs toxicity by label")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    fig.tight_layout()
    path = FIGURES_DIR / "sentiment_vs_toxicity_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_followers_engagement_scatter(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="author_followers_log10",
        y="engagement_log10",
        hue="label_name",
        palette=CATEGORY_COLORS,
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Author followers (log10 scale)")
    ax.set_ylabel("Engagement (log10 scale)")
    ax.set_title("Followers vs engagement by label")
    fig.tight_layout()
    path = FIGURES_DIR / "followers_vs_engagement_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_platform_misinfo_rate(df: pd.DataFrame) -> str:
    platform_rates = (
        df.groupby("platform", observed=True)["is_misinformation"]
        .mean()
        .sort_values(ascending=False)
    )
    rate_df = platform_rates.reset_index()
    rate_df.columns = ["platform", "misinformation_rate"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=rate_df,
        x="platform",
        y="misinformation_rate",
        hue="platform",
        palette=sns.color_palette("deep", len(rate_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_ylabel("Share of posts flagged as misinformation")
    ax.set_xlabel("Platform")
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Misinformation rate by platform")
    for idx, value in enumerate(rate_df["misinformation_rate"]):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "misinformation_rate_by_platform.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_platform_post_counts(df: pd.DataFrame) -> str:
    counts = df["platform"].value_counts().sort_values(ascending=False)
    count_df = counts.reset_index()
    count_df.columns = ["platform", "count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=count_df,
        x="platform",
        y="count",
        hue="platform",
        palette=sns.color_palette("deep", len(count_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Platform")
    ax.set_ylabel("Number of posts")
    ax.set_title("Post volume by platform")
    for idx, value in enumerate(count_df["count"]):
        ax.text(idx, value + 5, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "platform_counts.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_monthly_misinfo_rate(df: pd.DataFrame) -> str:
    rates = (
        df.groupby("month", observed=True)["is_misinformation"].mean().dropna()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        x=rates.index.astype(str),
        y=rates.values,
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Misinformation share")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Monthly misinformation prevalence")
    for idx, value in enumerate(rates.values):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    path = FIGURES_DIR / "monthly_misinformation_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_weekday_misinfo_rate(df: pd.DataFrame) -> str:
    rates = (
        df.groupby("weekday", observed=True)["is_misinformation"].mean().dropna()
    )
    rate_df = rates.reset_index()
    rate_df.columns = ["weekday", "misinformation_rate"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=rate_df,
        x="weekday",
        y="misinformation_rate",
        hue="weekday",
        palette=sns.color_palette("deep", len(rate_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Misinformation share")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Weekday misinformation prevalence")
    for idx, value in enumerate(rate_df["misinformation_rate"]):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    path = FIGURES_DIR / "weekday_misinformation_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_hourly_misinfo_rate(df: pd.DataFrame) -> str:
    rates = df.groupby("hour")["is_misinformation"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=rates.index, y=rates.values, marker="o", ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Misinformation share")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Hourly misinformation prevalence")
    fig.tight_layout()
    path = FIGURES_DIR / "hourly_misinformation_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_country_misinfo_rate(df: pd.DataFrame, top_n: int = 15) -> str:
    rates = (
        df.groupby("country")["is_misinformation"]
        .agg(["mean", "count"])
        .sort_values(by="count", ascending=False)
        .head(top_n)
    )
    rate_df = rates.reset_index()
    rate_df.columns = ["country", "misinformation_rate", "count"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=rate_df,
        y="country",
        x="misinformation_rate",
        hue="country",
        palette=sns.color_palette("viridis", len(rate_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Misinformation share")
    ax.set_ylabel("Country")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"Top {top_n} countries by misinformation prevalence")
    for idx, value in enumerate(rate_df["misinformation_rate"]):
        ax.text(value + 0.01, idx, f"{value:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "country_misinformation_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_factcheck_verdict_counts(df: pd.DataFrame) -> str:
    counts = df["factcheck_verdict"].value_counts().sort_values(ascending=False)
    count_df = counts.reset_index()
    count_df.columns = ["factcheck_verdict", "count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=count_df,
        x="factcheck_verdict",
        y="count",
        hue="factcheck_verdict",
        palette=sns.color_palette("deep", len(count_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Fact-check verdict")
    ax.set_ylabel("Number of posts")
    ax.set_title("Distribution of fact-check verdicts")
    for idx, value in enumerate(count_df["count"]):
        ax.text(idx, value + 5, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "factcheck_verdict_counts.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_factcheck_verdict_misinfo_rate(df: pd.DataFrame) -> str:
    rates = (
        df.groupby("factcheck_verdict", observed=True)["is_misinformation"]
        .mean()
        .sort_values(ascending=False)
    )
    rate_df = rates.reset_index()
    rate_df.columns = ["factcheck_verdict", "misinformation_rate"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=rate_df,
        x="factcheck_verdict",
        y="misinformation_rate",
        hue="factcheck_verdict",
        palette=sns.color_palette("deep", len(rate_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Fact-check verdict")
    ax.set_ylabel("Misinformation share")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Misinformation rate by fact-check verdict")
    for idx, value in enumerate(rate_df["misinformation_rate"]):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "factcheck_verdict_misinfo_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_author_verified_misinfo_rate(df: pd.DataFrame) -> str:
    rates = (
        df.groupby("author_verified", observed=True)["is_misinformation"]
        .mean()
        .rename(index={0: "Not verified", 1: "Verified"})
    )
    rate_df = rates.reset_index()
    rate_df.columns = ["author_verified", "misinformation_rate"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=rate_df,
        x="author_verified",
        y="misinformation_rate",
        hue="author_verified",
        palette=sns.color_palette("deep", len(rate_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Author verification status")
    ax.set_ylabel("Misinformation share")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Misinformation rate by author verification")
    for idx, value in enumerate(rate_df["misinformation_rate"]):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "author_verification_misinfo_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_platform_country_heatmap(df: pd.DataFrame, min_count: int = 5) -> str:
    pivot = (
        df.groupby(["country", "platform"], observed=True)["is_misinformation"]
        .agg(["mean", "count"])
        .reset_index()
    )
    pivot = pivot[pivot["count"] >= min_count]
    heatmap_data = pivot.pivot(index="country", columns="platform", values="mean")
    if heatmap_data.dropna(how="all").dropna(axis=1, how="all").empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        center=0.5,
        annot=True,
        fmt=".0%",
        cbar_kws={"format": PercentFormatter(1)},
        ax=ax,
    )
    ax.set_xlabel("Platform")
    ax.set_ylabel("Country")
    ax.set_title("Misinformation rate by country-platform combination\n(filtered to cells with ≥10 samples)")
    fig.tight_layout()
    path = FIGURES_DIR / "platform_country_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_platform_month_heatmap(df: pd.DataFrame) -> str:
    heatmap_data = (
        df.pivot_table(
            index="month",
            columns="platform",
            values="is_misinformation",
            aggfunc="mean",
            observed=True,
        )
        .reindex(df["month"].cat.categories)
    )
    if heatmap_data.dropna(how="all").dropna(axis=1, how="all").empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        center=0.5,
        annot=True,
        fmt=".0%",
        cbar_kws={"format": PercentFormatter(1)},
        ax=ax,
    )
    ax.set_xlabel("Platform")
    ax.set_ylabel("Month")
    ax.set_title("Misinformation rate by month and platform")
    fig.tight_layout()
    path = FIGURES_DIR / "platform_month_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_model_signature_distribution(df: pd.DataFrame) -> str:
    counts = df["model_signature"].value_counts().sort_values(ascending=False)
    count_df = counts.reset_index()
    count_df.columns = ["model_signature", "count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=count_df,
        x="model_signature",
        y="count",
        hue="model_signature",
        palette=sns.color_palette("deep", len(count_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Model signature")
    ax.set_ylabel("Number of posts")
    ax.set_title("Distribution of detected model signatures")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    path = FIGURES_DIR / "model_signature_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_model_signature_misinfo_rate(df: pd.DataFrame, top_n: int = 10) -> str:
    rates = (
        df.groupby("model_signature", observed=True)["is_misinformation"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    rate_df = rates.reset_index()
    rate_df.columns = ["model_signature", "misinformation_rate"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=rate_df,
        x="model_signature",
        y="misinformation_rate",
        hue="model_signature",
        palette=sns.color_palette("deep", len(rate_df)),
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Model signature")
    ax.set_ylabel("Misinformation share")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(f"Misinformation rate by model signature (top {top_n})")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    for idx, value in enumerate(rates.values):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "model_signature_misinfo_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)

def plot_feature_boxplot(
    df: pd.DataFrame,
    feature: str,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = [CATEGORY_COLORS["Legitimate"], CATEGORY_COLORS["Misinformation"]]
    sns.boxplot(
        data=df,
        x="label_name",
        y=feature,
        hue="label_name",
        palette=palette,
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("Label")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    corr = df[numeric_cols + ["is_misinformation"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature correlations with misinformation label")
    fig.tight_layout()
    path = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, List[str], List[str], str]:
    feature_columns = [
        "author_followers",
        "text_length",
        "token_count",
        "readability_score",
        "num_urls",
        "num_mentions",
        "num_hashtags",
        "sentiment_score",
        "toxicity_score",
        "detected_synthetic_score",
        "embedding_sim_to_facts",
        "external_factchecks_count",
        "source_domain_reliability",
        "engagement",
    ]
    categorical_columns = [
        "platform",
        "author_verified",
        "factcheck_verdict",
        "model_signature",
        "country",
    ]
    X = df[feature_columns + categorical_columns + [TEXT_COLUMN]]
    y = df["is_misinformation"]
    return X, y, feature_columns, categorical_columns, TEXT_COLUMN


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    text_col: str,
    *,
    for_tree: bool = False,
) -> ColumnTransformer:
    if for_tree:
        numeric_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        text_transformer = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=TFIDF_MAX_FEATURES,
                        ngram_range=(1, 2),
                        stop_words=None,
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=min(TEXT_SVD_COMPONENTS, TFIDF_MAX_FEATURES - 1),
                        random_state=42,
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
                ("text", text_transformer, text_col),
            ],
            sparse_threshold=0.0,
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("scaler", MaxAbsScaler()),
            ]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        text_transformer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words=None,
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
                ("text", text_transformer, text_col),
            ],
            sparse_threshold=0.3,
        )
    return preprocessor


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train_models(df: pd.DataFrame) -> ModelArtifacts:
    X, y, numeric_cols, categorical_cols, text_col = prepare_features(df)
    preprocessor_sparse = build_preprocessor(
        numeric_cols,
        categorical_cols,
        text_col,
        for_tree=False,
    )
    preprocessor_tree = build_preprocessor(
        numeric_cols,
        categorical_cols,
        text_col,
        for_tree=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor_sparse),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="saga",
                    penalty="l2",
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    y_pred_log = logistic.predict(X_test)
    y_proba_log = logistic.predict_proba(X_test)[:, 1]
    log_metrics = compute_classification_metrics(y_test, y_pred_log, y_proba_log)

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor_tree),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    y_proba_rf = random_forest.predict_proba(X_test)[:, 1]
    rf_metrics = compute_classification_metrics(y_test, y_pred_rf, y_proba_rf)

    figure_paths: List[str] = []

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_rf,
        display_labels=["Legitimate", "Misinformation"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Random Forest confusion matrix")
    fig.tight_layout()
    cm_path = FIGURES_DIR / "random_forest_confusion_matrix.png"
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(cm_path))

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", color="grey", label="Chance")
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(
        y_test,
        y_proba_rf,
        ax=ax,
        name="Random Forest",
    )
    ax.set_title("ROC curve – Random Forest")
    fig.tight_layout()
    roc_path = FIGURES_DIR / "random_forest_roc_curve.png"
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(roc_path))

    # Feature importances
    rf_clf: RandomForestClassifier = random_forest.named_steps["classifier"]
    preprocessor_tree_fitted = random_forest.named_steps["preprocessor"]
    encoder: OneHotEncoder = preprocessor_tree_fitted.named_transformers_["cat"]
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    text_transformer = preprocessor_tree_fitted.named_transformers_["text"]
    svd: TruncatedSVD = text_transformer.named_steps["svd"]
    text_feature_names = [f"text_topic_{i}" for i in range(svd.components_.shape[0])]
    all_feature_names = numeric_cols + list(encoded_feature_names) + text_feature_names
    importances = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": rf_clf.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    top_importances = importances.head(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("viridis", len(top_importances))
    sns.barplot(
        data=top_importances,
        x="importance",
        y="feature",
        hue="feature",
        palette=palette,
        dodge=False,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Random Forest feature importance (top 15)")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fi_path = FIGURES_DIR / "random_forest_feature_importance.png"
    fig.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(str(fi_path))

    metrics = {
        "logistic_regression": log_metrics,
        "random_forest": rf_metrics,
    }

    return ModelArtifacts(
        pipeline=random_forest,
        metrics=metrics,
        figure_paths=figure_paths,
        feature_importances=importances,
    )


def risk_prioritisation_analysis(
    df: pd.DataFrame,
    pipeline: Pipeline,
    review_steps: List[int] | None = None,
) -> Dict[str, object]:
    """Compute risk-based selection curve and related diagnostics."""
    X, y, _, _, _ = prepare_features(df)
    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]
    df_risk = df.copy()
    df_risk["predicted_misinfo_probability"] = probabilities
    df_risk["expected_harm"] = df_risk["predicted_misinfo_probability"] * df_risk["engagement"]
    df_risk = df_risk.sort_values(by="expected_harm", ascending=False).reset_index(drop=True)

    cumulative_misinfo = df_risk["is_misinformation"].cumsum()
    total_misinfo = df_risk["is_misinformation"].sum()
    review_counts = np.arange(1, len(df_risk) + 1)
    coverage = cumulative_misinfo / total_misinfo
    review_share = review_counts / len(df_risk)

    coverage_df = pd.DataFrame(
        {
            "posts_reviewed": review_counts,
            "share_of_posts": review_share,
            "share_of_misinformation_captured": coverage,
        }
    )
    coverage_df.to_csv(RISK_COVERAGE_CSV, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        data=coverage_df,
        x="share_of_posts",
        y="share_of_misinformation_captured",
        ax=ax,
        label="Risk-based prioritisation",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random review (baseline)")
    ax.set_xlabel("Fraction of posts reviewed")
    ax.set_ylabel("Fraction of misinformation captured")
    ax.set_title("Coverage curve for risk-prioritised review")
    ax.legend()
    fig.tight_layout()
    coverage_path = FIGURES_DIR / "risk_coverage_curve.png"
    fig.savefig(coverage_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if review_steps is None:
        review_steps = [10, 20, 50, 100, 150]

    summary_rows = []
    for k in review_steps:
        top_k = df_risk.head(k)
        captured = top_k["is_misinformation"].sum()
        share_misinfo_captured = captured / total_misinfo
        total_expected_harm = top_k["expected_harm"].sum()
        summary_rows.append(
            {
                "review_capacity": k,
                "share_of_posts": k / len(df_risk),
                "misinformation_captured": int(captured),
                "share_of_misinformation_captured": float(share_misinfo_captured),
                "expected_harm_addressed": float(total_expected_harm),
            }
        )

    return {
        "coverage_curve_path": str(coverage_path),
        "coverage_summary": summary_rows,
    }


def compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Compile descriptive statistics needed for reporting."""
    class_counts = df["is_misinformation"].value_counts().sort_index()
    class_props = (class_counts / len(df)).round(4)

    platform_counts = df["platform"].value_counts().to_dict()
    platform_rates = (
        df.groupby("platform", observed=True)["is_misinformation"].mean().round(4).to_dict()
    )

    month_rates = (
        df.groupby("month", observed=True)["is_misinformation"].mean().dropna().round(4).to_dict()
    )
    weekday_rates = (
        df.groupby("weekday", observed=True)["is_misinformation"].mean().dropna().round(4).to_dict()
    )

    numeric_features = [
        "author_followers",
        "text_length",
        "token_count",
        "sentiment_score",
        "toxicity_score",
        "detected_synthetic_score",
        "embedding_sim_to_facts",
        "source_domain_reliability",
        "engagement",
    ]

    means_by_label = {}
    for feature in numeric_features:
        grouped = df.groupby("is_misinformation")[feature].mean()
        means_by_label[feature] = {
            "legitimate_mean": float(grouped.get(0, np.nan)),
            "misinfo_mean": float(grouped.get(1, np.nan)),
        }

    return {
        "shape": {"n_rows": int(df.shape[0]), "n_columns": int(df.shape[1])},
        "class_counts": class_counts.to_dict(),
        "class_proportions": class_props.to_dict(),
        "platform_counts": platform_counts,
        "platform_misinfo_rate": platform_rates,
        "month_misinfo_rate": month_rates,
        "weekday_misinfo_rate": weekday_rates,
        "feature_means_by_label": means_by_label,
    }


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_output_dirs()
    dataset = load_dataset(DATA_PATH)

    summary = compute_descriptive_stats(dataset)

    figure_paths: List[str] = []

    base_figures = [
        plot_class_distribution(dataset),
        plot_platform_post_counts(dataset),
        plot_platform_misinfo_rate(dataset),
        plot_monthly_misinfo_rate(dataset),
        plot_weekday_misinfo_rate(dataset),
        plot_hourly_misinfo_rate(dataset),
        plot_country_misinfo_rate(dataset),
        plot_factcheck_verdict_counts(dataset),
        plot_factcheck_verdict_misinfo_rate(dataset),
        plot_author_verified_misinfo_rate(dataset),
        plot_model_signature_distribution(dataset),
        plot_model_signature_misinfo_rate(dataset),
        plot_platform_country_heatmap(dataset),
        plot_platform_month_heatmap(dataset),
        plot_sentiment_toxicity_scatter(dataset),
        plot_followers_engagement_scatter(dataset),
    ]
    for path in base_figures:
        if path:
            figure_paths.append(path)

    label_boxplots = [
        plot_feature_boxplot(
            dataset,
            feature="sentiment_score",
            ylabel="Sentiment score",
            title="Sentiment distribution by label",
            filename="sentiment_by_label.png",
        ),
        plot_feature_boxplot(
            dataset,
            feature="toxicity_score",
            ylabel="Toxicity score",
            title="Toxicity distribution by label",
            filename="toxicity_by_label.png",
        ),
        plot_feature_boxplot(
            dataset,
            feature="text_length",
            ylabel="Text length (characters)",
            title="Text length distribution by label",
            filename="text_length_by_label.png",
        ),
        plot_feature_boxplot(
            dataset,
            feature="source_domain_reliability",
            ylabel="Domain reliability score",
            title="Source reliability by label",
            filename="source_reliability_by_label.png",
        ),
        plot_feature_boxplot(
            dataset,
            feature="engagement",
            ylabel="Engagement",
            title="Engagement distribution by label",
            filename="engagement_by_label.png",
        ),
    ]
    for path in label_boxplots:
        if path:
            figure_paths.append(path)

    for feature, pretty_name, bins in NUMERIC_DISTRIBUTION_SPECS:
        hist_path = plot_numeric_histogram(
            dataset,
            feature=feature,
            pretty_name=pretty_name,
            filename=f"{feature}_hist.png",
            bins=bins,
        )
        figure_paths.append(hist_path)
        hist_label_path = plot_numeric_histogram_by_label(
            dataset,
            feature=feature,
            pretty_name=pretty_name,
            filename=f"{feature}_hist_by_label.png",
            bins=bins,
        )
        figure_paths.append(hist_label_path)

    for spec in BOXPLOT_CATEGORY_SPECS:
        path = plot_numeric_box_by_category(
            dataset,
            feature=spec["feature"],
            category=spec["category"],
            pretty_feature=spec["pretty_feature"],
            pretty_category=spec["pretty_category"],
            filename=spec["filename"],
            order=spec["order"],
        )
        figure_paths.append(path)

    numeric_for_corr = [
        "author_followers",
        "text_length",
        "token_count",
        "readability_score",
        "num_urls",
        "num_mentions",
        "num_hashtags",
        "sentiment_score",
        "toxicity_score",
        "detected_synthetic_score",
        "embedding_sim_to_facts",
        "external_factchecks_count",
        "source_domain_reliability",
        "engagement",
    ]
    figure_paths.append(plot_correlation_heatmap(dataset, numeric_for_corr))

    model_artifacts = train_models(dataset)
    figure_paths.extend(model_artifacts.figure_paths)

    risk_outputs = risk_prioritisation_analysis(dataset, model_artifacts.pipeline)
    figure_paths.append(risk_outputs["coverage_curve_path"])

    all_results = {
        "descriptive_stats": summary,
        "model_performance": model_artifacts.metrics,
        "feature_importances_top15": model_artifacts.feature_importances.head(15).to_dict(
            orient="records"
        ),
        "risk_prioritisation": {
            "coverage_summary": risk_outputs["coverage_summary"],
        },
        "figure_paths": figure_paths,
    }

    with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
