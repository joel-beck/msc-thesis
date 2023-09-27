import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
from shared import (
    average,
    average_by_group,
    construct_feature_weights_labels,
    load_evaluation_frame,
)

sns.set_theme(style="whitegrid")

plt.rcParams["text.usetex"] = True


def compare_hybridization_strategies(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return average(evaluation_frame).select(
        [
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_l_to_c_cand",
            "mean_avg_precision_l_to_c",
        ]
    )


def plot_hybridization_strategies_barplot(evaluation_frame: pl.DataFrame) -> None:
    label_mapping = {
        "mean_avg_precision_c_to_l_cand": "Citation Candidates",
        "mean_avg_precision_c_to_l": r"Citation $\rightarrow$ Language",
        "mean_avg_precision_l_to_c_cand": "Language Candidates",
        "mean_avg_precision_l_to_c": r"Language $\rightarrow$ Citation",
    }

    plot_df = (
        compare_hybridization_strategies(evaluation_frame)
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="value", y="variable", data=plot_df, ax=ax)
    ax.set(
        xlabel="Mean Average Precision",
        ylabel="Hybridization Strategy",
        title="Mean Average Precision of Hybridization Strategies",
    )
    plt.show()


def plot_hybridization_strategies_boxplot(
    evaluation_frame: pl.DataFrame, save_path: str
) -> None:
    cols = [
        "avg_precision_c_to_l_cand",
        "avg_precision_c_to_l",
        "avg_precision_l_to_c_cand",
        "avg_precision_l_to_c",
    ]
    strategy_names = {
        "avg_precision_c_to_l_cand": "Citation Candidates",
        "avg_precision_c_to_l": r"Citation $\rightarrow$ Language",
        "avg_precision_l_to_c_cand": "Language Candidates",
        "avg_precision_l_to_c": r"Language $\rightarrow$ Citation",
    }

    boxplot_data = (
        evaluation_frame.select(cols)
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(strategy_names))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(y="variable", x="value", data=boxplot_data, palette="pastel", ax=ax)
    ax.set_title(
        "Average Precision Distribution by Hybridization Strategy", size=25, pad=20
    )
    ax.set_xlabel("Average Precision", size=20)
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), size=15)
    ax.set_yticklabels(ax.get_yticklabels(), size=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()


def compare_hybridization_strategies_by_language_model(
    evaluation_frame: pl.DataFrame,
) -> pl.DataFrame:
    return average_by_group(evaluation_frame, ["language_model"]).select(
        [
            "language_model",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_l_to_c_cand",
            "mean_avg_precision_l_to_c",
        ]
    )


def plot_hybridization_strategies_by_language_model_barplot(
    evaluation_frame: pl.DataFrame,
) -> None:
    label_mapping = {
        "mean_avg_precision_c_to_l_cand": "Citation Candidates",
        "mean_avg_precision_c_to_l": r"Citation $\rightarrow$ Language",
        "mean_avg_precision_l_to_c_cand": "Language Candidates",
        "mean_avg_precision_l_to_c": r"Language $\rightarrow$ Citation",
    }

    plot_df = (
        compare_hybridization_strategies_by_language_model(evaluation_frame)
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
    g.set_titles("{col_name}", size=40)
    g.tick_params(labelsize=30)
    g.set_xticklabels(size=30)
    g.set_yticklabels(size=30)

    plt.suptitle("MAP of Hybridization Strategies by Language Model", y=1.02, size=50)
    plt.subplots_adjust(top=0.95)
    plt.tight_layout()

    plt.show()


def plot_hybridization_strategies_by_language_model_stripplot(
    evaluation_frame: pl.DataFrame, save_path: str
) -> None:
    label_mapping = {
        "mean_avg_precision_c_to_l_cand": "Citation Candidates",
        "mean_avg_precision_c_to_l": r"Citation $\rightarrow$ Language",
        "mean_avg_precision_l_to_c_cand": "Language Candidates",
        "mean_avg_precision_l_to_c": r"Language $\rightarrow$ Citation",
    }

    plot_df = (
        compare_hybridization_strategies_by_language_model(evaluation_frame)
        .melt(id_vars=["language_model"])
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="strip",
        x="value",
        y="variable",
        hue="variable",
        col="language_model",
        col_wrap=3,
        size=35,
        height=12,
        aspect=1,
        sharex=True,
        sharey=True,
        legend=False,
    )

    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=45)

    xticks = [0.45, 0.5, 0.55, 0.6, 0.65]
    g.set(xticks=xticks)
    xticklabels = [f"{x:.2f}" for x in xticks]
    g.set_xticklabels(xticklabels, size=35)

    yticklabels = g.axes.flat[0].get_yticklabels()
    g.set_yticklabels(yticklabels, size=40)

    plt.suptitle("MAP of Hybridization Strategies by Language Model", y=1.02, size=70)
    plt.subplots_adjust(top=0.96, hspace=0.1, wspace=0.1)

    g.savefig(save_path, dpi=300)
    plt.show()


def plot_tfidf_change_in_map(evaluation_frame: pl.DataFrame, save_path: str) -> None:
    plot_df = (
        average_by_group(
            evaluation_frame.filter(pl.col("language_model") == "TFIDF"),
            ["feature_weights"],
        )
        .select(
            [
                pl.col("feature_weights").apply(construct_feature_weights_labels),
                (
                    pl.col("mean_avg_precision_c_to_l")
                    - pl.col("mean_avg_precision_c_to_l_cand")
                ).alias(r"Citation $\rightarrow$ Language"),
                (
                    pl.col("mean_avg_precision_l_to_c")
                    - pl.col("mean_avg_precision_l_to_c_cand")
                ).alias(r"Language $\rightarrow$ Citation"),
            ]
        )
        .melt(id_vars=["feature_weights"])
    )

    g = sns.FacetGrid(
        plot_df,
        col="variable",
        col_order=[
            r"Citation $\rightarrow$ Language",
            r"Language $\rightarrow$ Citation",
        ],
        sharey=True,
        height=5,
        aspect=1.2,
    )

    g.set_titles("{col_name}", size=15)
    g.set_axis_labels(x_var="Change in MAP", y_var="")

    data_groups_mapping = dict(plot_df.groupby("variable"))
    data_citation_to_language = data_groups_mapping[r"Citation $\rightarrow$ Language"]
    data_language_to_citation = data_groups_mapping[r"Language $\rightarrow$ Citation"]

    for ax, data in zip(
        g.axes.flat, [data_citation_to_language, data_language_to_citation]
    ):
        for idx, val in enumerate(reversed(data["value"])):
            color = "green" if val >= 0 else "red"
            ax.hlines(y=idx, xmin=0, xmax=val, color=color, alpha=0.6)
            ax.plot(val, idx, "o", color=color)

        yticks = list(range(len(data)))[::-1]
        ax.set_yticks(yticks)

        yticklabels = data_citation_to_language["feature_weights"].to_list()
        ax.set_yticklabels(yticklabels)

    g.tight_layout()
    g.fig.suptitle("TF-IDF Performance in the Hybrid Recommender", size=18)
    g.fig.subplots_adjust(top=0.85)

    g.savefig(save_path, dpi=300)
    plt.show()


def main() -> None:
    evaluation_frame = load_evaluation_frame()

    plot_hybridization_strategies_boxplot(
        evaluation_frame,
        save_path="../../plots/hybridization_strategies_boxplot.png",
    )

    row_order_mapping = {
        val: idx
        for idx, val in enumerate(
            [
                "mean",
                "std",
                "25%",
                "50%",
                "75%",
            ]
        )
    }

    base_df = (
        evaluation_frame.select(cs.starts_with("avg_precision"))
        .describe()
        .rename({"describe": "value"})
        .filter(~pl.col("value").is_in(["count", "null_count", "min", "max"]))
        .sort(pl.col("value").map_dict(row_order_mapping))
    )

    # get inverse quantiles for each column
    def inverse_quantile(series: pl.Series, value: float = 0.384) -> float:
        return (series < value).mean()

    # 35.86% of the average precisions are below 0.384 and thus worse than the baseline
    inverse_quantile(evaluation_frame["avg_precision_c_to_l"])
    # 38.38% of the average precisions are below 0.384
    inverse_quantile(evaluation_frame["avg_precision_c_to_l_cand"])
    # 24.54% of the average precisions are below 0.384
    inverse_quantile(evaluation_frame["avg_precision_l_to_c"])
    # 21.99% of the average precisions are below 0.384
    inverse_quantile(evaluation_frame["avg_precision_l_to_c_cand"])

    proportions_worse_than_baseline = {
        "value": "proportion < baseline",
        "avg_precision_c_to_l_cand": 0.3838,
        "avg_precision_c_to_l": 0.3586,
        "avg_precision_l_to_c_cand": 0.2199,
        "avg_precision_l_to_c": 0.2454,
    }

    new_df = pl.DataFrame(proportions_worse_than_baseline)
    pl.concat([base_df, new_df])

    plot_hybridization_strategies_by_language_model_stripplot(
        evaluation_frame,
        save_path="../../plots/hybridization_strategies_by_language_model_stripplot.png",
    )

    plot_tfidf_change_in_map(
        evaluation_frame, save_path="../../plots/tfidf_change_in_map.png"
    )


if __name__ == "__main__":
    main()
