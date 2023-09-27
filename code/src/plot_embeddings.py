import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    categories = {
        "Recommender Systems": [
            "collaborative",
            "content-based",
            "hybrid",
            "cold-start",
            "user-based",
            "item-based",
        ],
        "Citation Analysis": [
            "co-citation analysis",
            "bibliographic coupling",
            "citation count",
            "CPA",
            "h-index",
            "reference",
        ],
        "NLP Basics": [
            "embedding",
            "tokenization",
            "stemming",
            "lemmatization",
            "stopwords",
            "cosine similarity",
        ],
        "Language Models": [
            "TF-IDF",
            "Word2Vec",
            "FastText",
            "SciBERT",
            "BERT",
            "Longformer",
        ],
        "Metrics": ["Precision", "Recall", "F1-Score", "MAP", "NDCG", "MRR"],
    }

    cluster_centers = {
        "Recommender Systems": (-4, -4),
        "Citation Analysis": (4, -4),
        "NLP Basics": (3, 4),
        "Language Models": (5, 4),
        "Metrics": (-4, 4),
    }

    np.random.seed(108)

    data = []
    for category, words in categories.items():
        center_x, center_y = cluster_centers[category]
        for word in words:
            x = center_x + np.random.normal(0, 1.3)
            y = center_y + np.random.normal(0, 1.3)
            data.append([word, x, y, category])

    df = pd.DataFrame(data, columns=["word", "x", "y", "category"])

    palette = sns.color_palette("husl", 5)
    swapped_palette = [palette[i] for i in [2, 1, 0, 3, 4]]

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.scatterplot(
        x="x",
        y="y",
        hue="category",
        data=df,
        palette=swapped_palette,
        s=250,
        edgecolor="w",
        linewidth=0.5,
        legend=None,
        style="category",
        markers=["o"] * 5,
        ax=ax,
    )

    for index, row in df.iterrows():
        offset = 0.1
        ax.text(
            row["x"] + offset,
            row["y"] + offset,
            row["word"],
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=12,
            color="black",
        )

    ax.set_title("Word Embeddings projected to 2 Dimensions", fontsize=16)
    ax.set_xlabel("Embedding Dimension 1", fontsize=14)
    ax.set_ylabel("Embedding Dimension 2", fontsize=14)
    fig.tight_layout()

    fig.savefig("../../plots/word_embeddings.png")


if __name__ == "__main__":
    main()
