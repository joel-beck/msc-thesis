import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def load_evaluation_frame() -> pl.DataFrame:
    load_dotenv()
    results_dirpath = os.getenv("RESULTS_DIRPATH")

    if results_dirpath is None:
        raise ValueError("RESULTS_DIRPATH environment variable not set")

    evaluation_frame_path = Path(results_dirpath) / "evaluation_frame.parquet"

    return pl.read_parquet(evaluation_frame_path).select(
        [
            "semanticscholar_id",
            "language_model",
            "feature_weights",
            "avg_precision_c_to_l_cand",
            "avg_precision_c_to_l",
            "avg_precision_l_to_c_cand",
            "avg_precision_l_to_c",
            "num_unique_labels_c_to_l_cand",
            "num_unique_labels_c_to_l",
            "num_unique_labels_l_to_c_cand",
            "num_unique_labels_l_to_c",
        ]
    )


def average(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        mean_avg_precision_c_to_l_cand=pl.col("avg_precision_c_to_l_cand").mean(),
        mean_avg_precision_c_to_l=pl.col("avg_precision_c_to_l").mean(),
        mean_avg_precision_l_to_c_cand=pl.col("avg_precision_l_to_c_cand").mean(),
        mean_avg_precision_l_to_c=pl.col("avg_precision_l_to_c").mean(),
        mean_num_unique_labels_c_to_l_cand=pl.col(
            "num_unique_labels_c_to_l_cand"
        ).mean(),
        mean_num_unique_labels_c_to_l=pl.col("num_unique_labels_c_to_l").mean(),
        mean_num_unique_labels_l_to_c_cand=pl.col(
            "num_unique_labels_l_to_c_cand"
        ).mean(),
        mean_num_unique_labels_l_to_c=pl.col("num_unique_labels_l_to_c").mean(),
    )


def average_by_group(
    evaluation_frame: pl.DataFrame, grouping_columns: Sequence[str]
) -> pl.DataFrame:
    return evaluation_frame.groupby(grouping_columns, maintain_order=True).agg(
        mean_avg_precision_c_to_l_cand=pl.col("avg_precision_c_to_l_cand").mean(),
        mean_avg_precision_c_to_l=pl.col("avg_precision_c_to_l").mean(),
        mean_avg_precision_l_to_c_cand=pl.col("avg_precision_l_to_c_cand").mean(),
        mean_avg_precision_l_to_c=pl.col("avg_precision_l_to_c").mean(),
        mean_num_unique_labels_c_to_l_cand=pl.col(
            "num_unique_labels_c_to_l_cand"
        ).mean(),
        mean_num_unique_labels_c_to_l=pl.col("num_unique_labels_c_to_l").mean(),
        mean_num_unique_labels_l_to_c_cand=pl.col(
            "num_unique_labels_l_to_c_cand"
        ).mean(),
        mean_num_unique_labels_l_to_c=pl.col("num_unique_labels_l_to_c").mean(),
    )


def construct_feature_weights_labels(feature_weights_string: str) -> str:
    return "[" + ", ".join(feature_weights_string.split(", ")) + "]"
