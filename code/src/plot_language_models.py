from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from shared import (
    average,
    average_by_group,
    construct_feature_weights_labels,
    load_evaluation_frame,
)

sns.set_theme(style="whitegrid")

# LaTeX rendering in plots
plt.rcParams["text.usetex"] = True


def compare_language_models(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["language_model"])
        .select(["language_model", "mean_avg_precision_l_to_c_cand"])
        .sort(by="mean_avg_precision_l_to_c_cand", descending=True)
    )


def plot_language_models(evaluation_frame: pl.DataFrame, save_path: str) -> None:
    plot_df = (
        compare_language_models(evaluation_frame)
        .sort(by="mean_avg_precision_l_to_c_cand", descending=True)
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="mean_avg_precision_l_to_c_cand", y="language_model", data=plot_df, ax=ax
    )
    ax.set(
        xlabel="Mean Average Precision",
        ylabel="Language Model",
        title="MAP of Language Recommender by Language Model",
    )

    fig.savefig(save_path)
    plt.show()


def compare_language_models_feature_weights(
    evaluation_frame: pl.DataFrame,
) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["language_model", "feature_weights"])
        .select(
            [
                "language_model",
                "feature_weights",
                "mean_avg_precision_c_to_l",
                "mean_avg_precision_l_to_c",
            ]
        )
        .sort("mean_avg_precision_c_to_l", descending=True)
    )


def plot_language_models_feature_weights_barplot(
    evaluation_frame: pl.DataFrame, save_path: str
) -> None:
    plot_df = (
        compare_language_models_feature_weights(evaluation_frame)
        .sort(by="mean_avg_precision_c_to_l", descending=True)
        .with_columns(
            feature_weight_labels=pl.col("feature_weights").apply(
                construct_feature_weights_labels
            )
        )
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="mean_avg_precision_c_to_l",
        y="language_model",
        col="feature_weight_labels",
        col_wrap=4,
        height=16,
        aspect=1,
        sharex=False,
        sharey=True,
    )
    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=30)
    g.tick_params(labelsize=30)
    # sns.set(font_scale=3)

    plt.suptitle(
        "Mean Average Precision of Feature Weights by Language Model", y=1.02, size=40
    )

    g.savefig(save_path)
    plt.show()


def plot_language_models_feature_weights_heatmap(
    evaluation_frame: pl.DataFrame,
    save_path: str,
    base_recommender: Literal["citation", "language"],
) -> None:
    column_order = [
        "None",
        "TFIDF",
        "BM25",
        "WORD2VEC",
        "GLOVE",
        "FASTTEXT",
        "BERT",
        "SCIBERT",
        "LONGFORMER",
    ]

    index_order = [
        "[1,0,0,0,0]",
        "[0,1,0,0,0]",
        "[0,0,1,0,0]",
        "[0,0,0,1,0]",
        "[0,0,0,0,1]",
        "[1,1,1,1,1]",
        "[0,3,17,96,34]",
        "[2,7,12,15,72]",
        "[2,13,18,72,66]",
        "[9,9,14,9,87]",
        "[9,12,13,68,95]",
        "[9,19,20,67,1]",
        "[10,13,19,65,14]",
        "[16,15,5,84,10]",
        "[17,2,18,54,4]",
        "[18,10,6,83,63]",
    ]

    feature_weight_candidates = (
        average_by_group(evaluation_frame, grouping_columns=["feature_weights"])
        .select(["feature_weights", "mean_avg_precision_c_to_l_cand"])
        .rename({"mean_avg_precision_c_to_l_cand": "None"})
    )

    labels_frame = compare_language_models_feature_weights(evaluation_frame)

    merged_c_to_l = (
        (
            labels_frame.pivot(
                values="mean_avg_precision_c_to_l",
                columns="language_model",
                index="feature_weights",
            )
            .join(feature_weight_candidates, on="feature_weights")
            .with_columns(
                feature_weights_labels=pl.col("feature_weights").apply(
                    construct_feature_weights_labels
                )
            )
        )
        .to_pandas()
        .set_index("feature_weights_labels")
        .reindex(index_order)
        .loc[:, column_order]
    )

    language_model_candidates = (
        average_by_group(evaluation_frame, grouping_columns=["language_model"])
        .select(["language_model", "mean_avg_precision_l_to_c_cand"])
        .rename({"mean_avg_precision_l_to_c_cand": "None"})
    )

    merged_l_to_c = (
        (
            labels_frame.with_columns(
                feature_weights_labels=pl.col("feature_weights").apply(
                    construct_feature_weights_labels
                )
            )
            .pivot(
                values="mean_avg_precision_l_to_c",
                index="language_model",
                columns="feature_weights_labels",
            )
            .join(language_model_candidates, on="language_model")
        )
        .to_pandas()
        .set_index("language_model")
        .reindex([col for col in column_order if col != "None"])
        .loc[:, ["None", *index_order]]
    ).T

    plot_df = merged_c_to_l if base_recommender == "citation" else merged_l_to_c
    title = (
        r"MAP (Citation $\rightarrow$ Language) of Feature Weights by Language Model"
        if base_recommender == "citation"
        else r"MAP (Language $\rightarrow$ Citation) of Feature Weights by Language Model"
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(plot_df, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, ax=ax)

    ax.set_title(title, size=20, pad=20)
    ax.set(xlabel="", ylabel="")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    fig.savefig(save_path)
    plt.show()


def main() -> None:
    evaluation_frame = load_evaluation_frame()

    # Null model with randomly sampled recommendations with same proportions of
    # relevant documents as in the test set and a recommendation list of length 20 has a
    # MAP of 0.384

    # present as table rather than plot
    average_by_group(evaluation_frame, ["language_model"]).select(
        ["language_model", "mean_avg_precision_l_to_c_cand"]
    ).to_pandas().to_clipboard()

    plot_language_models(evaluation_frame, save_path="../../plots/language_models.png")

    plot_language_models_feature_weights_heatmap(
        evaluation_frame,
        save_path="../../plots/language_models_feature_weights_heatmap_c_to_l.png",
        base_recommender="citation",
    )

    plot_language_models_feature_weights_heatmap(
        evaluation_frame,
        save_path="../../plots/language_models_feature_weights_heatmap_l_to_c.png",
        base_recommender="language",
    )

    average_by_group(evaluation_frame, ["language_model", "feature_weights"]).filter(
        pl.col("feature_weights") == "0,0,0,0,1"
    ).select(
        [
            "language_model",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_l_to_c_cand",
            "mean_avg_precision_l_to_c",
        ]
    )

    average_by_group(evaluation_frame, ["language_model", "feature_weights"]).filter(
        pl.col("language_model") == "TFIDF"
    ).select(
        [
            "feature_weights",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_l_to_c_cand",
            "mean_avg_precision_l_to_c",
        ]
    )

    average_by_group(evaluation_frame, ["feature_weights"]).select(
        [
            "feature_weights",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_l_to_c_cand",
            "mean_avg_precision_l_to_c",
        ]
    )

    average(evaluation_frame).select(
        [
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_l_to_c_cand",
            "mean_avg_precision_l_to_c",
        ]
    )


if __name__ == "__main__":
    main()
