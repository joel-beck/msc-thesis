import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from shared import average, average_by_group, load_evaluation_frame

sns.set_theme(style="whitegrid")


def compare_diversity(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Compare number of unique labels for both hybrid recommender orders. Since the number
    keeps the same when moving from the candidate list to the final recommendations, it
    is sufficient to consider the candidate lists.
    """
    return average(evaluation_frame).select(
        ["mean_num_unique_labels_c_to_l_cand", "mean_num_unique_labels_l_to_c_cand"]
    )


def plot_diversities_barplot(evaluation_frame: pl.DataFrame) -> None:
    label_mapping = {
        "mean_num_unique_labels_c_to_l_cand": "Citation to Language",
        "mean_num_unique_labels_l_to_c_cand": "Language to Citation",
    }

    plot_df = (
        compare_diversity(evaluation_frame)
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="value", y="variable", data=plot_df, ax=ax)
    ax.set(
        xlabel="Number of Unique Labels",
        ylabel="Hybridization Strategy",
        title="Mean Number of Unique Labels of Hybridization Strategies",
    )
    plt.show()


def plot_diversities_boxplot(evaluation_frame: pl.DataFrame, save_path: str) -> None:
    label_mapping = {
        "num_unique_labels_c_to_l_cand": "Citation to Language",
        "num_unique_labels_l_to_c_cand": "Language to Citation",
    }

    plot_df = (
        evaluation_frame.select(
            ["num_unique_labels_c_to_l_cand", "num_unique_labels_l_to_c_cand"]
        )
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="value", y="variable", data=plot_df, ax=ax)
    ax.set(
        xlabel="Number of Unique Labels",
        ylabel="",
        title="Distribution of Unique Labels by Hybridization Strategy",
    )

    fig.savefig(save_path)
    plt.show()


def compare_diversity_by_language_model(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return average_by_group(evaluation_frame, ["language_model"]).select(
        [
            "language_model",
            "mean_num_unique_labels_c_to_l_cand",
            "mean_num_unique_labels_l_to_c_cand",
        ]
    )


def plot_diversity_by_language_model(
    evaluation_frame: pl.DataFrame, save_path: str
) -> None:
    label_mapping = {
        "mean_num_unique_labels_c_to_l_cand": "Citation to Language",
        "mean_num_unique_labels_l_to_c_cand": "Language to Citation",
    }

    plot_df = (
        compare_diversity_by_language_model(evaluation_frame)
        .melt(id_vars=["language_model"])
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="value",
        y="variable",
        col="language_model",
        col_wrap=3,
        height=12,
        aspect=1,
        sharex=True,
        sharey=True,
    )

    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=30)
    g.tick_params(labelsize=30)

    plt.suptitle(
        "Mean Number of Unique Labels of Hybridization Strategies by Language Model",
        y=1.02,
        size=40,
    )
    plt.subplots_adjust(top=0.95)
    g.tight_layout()

    g.savefig(save_path)
    plt.show()


def main() -> None:
    evaluation_frame = load_evaluation_frame()

    # display as table rather than as plot
    evaluation_frame.select(
        ["num_unique_labels_c_to_l_cand", "num_unique_labels_l_to_c_cand"]
    ).describe().rename({"describe": "statistic"}).filter(
        ~pl.col("statistic").is_in(["count", "null_count"])
    ).rename(
        {
            "num_unique_labels_c_to_l_cand": "Citation to Language",
            "num_unique_labels_l_to_c_cand": "Language to Citation",
        }
    ).select(
        pl.col("statistic").str.to_uppercase(),
        pl.col("Citation to Language").round(1),
        pl.col("Language to Citation").round(1),
    ).to_pandas().to_clipboard()

    plot_diversities_boxplot(evaluation_frame, "../../plots/diversities_boxplot.png")


if __name__ == "__main__":
    main()
