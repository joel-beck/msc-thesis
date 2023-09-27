"""
Step 3: Final Evaluation

Numerical and graphical analysis of the evaluation results.
"""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from shared import average_by_group, load_evaluation_frame

sns.set_theme(style="whitegrid")


def compare_feature_weights(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["feature_weights"])
        .select(["feature_weights", "mean_avg_precision_c_to_l_cand"])
        .sort(by="mean_avg_precision_c_to_l_cand", descending=True)
    )


def construct_feature_weights_labels(feature_weights_string: str) -> str:
    return "[" + ", ".join(feature_weights_string.split(", ")) + "]"


def add_bar_labels(ax: plt.Axes) -> None:
    # Displaying exact values on the bars
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            x=width - 0.03,  # move text inside the bars
            y=p.get_y() + p.get_height() / 1.7,  # center text in middle of the bars
            s=f"{width:1.3f}",  # round to 3 decimals
            ha="center",
            va="center",
        )


def plot_feature_weights(evaluation_frame: pl.DataFrame, save_path: str) -> None:
    plot_df = (
        compare_feature_weights(evaluation_frame)
        .sort(by="mean_avg_precision_c_to_l_cand", descending=True)
        .with_columns(
            feature_weight_labels=pl.col("feature_weights").apply(
                construct_feature_weights_labels
            )
        )
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="mean_avg_precision_c_to_l_cand",
        y="feature_weight_labels",
        data=plot_df,
        ax=ax,
        palette="coolwarm",
    )
    add_bar_labels(ax)

    ax.set_xlabel("Mean Average Precision")
    ax.set_ylabel("")
    ax.set_title("MAP of Citation Recommender by Feature Weights", pad=20, fontsize=20)

    fig.savefig(save_path)
    plt.show()


def main() -> None:
    evaluation_frame = load_evaluation_frame()

    compare_feature_weights(evaluation_frame)
    plot_feature_weights(evaluation_frame, save_path="../../plots/feature_weights.png")


if __name__ == "__main__":
    main()
