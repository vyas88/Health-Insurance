import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from data import build_data


ROOT = Path(__file__).resolve().parents[1]
PALETTE = {"Low": "#0072B2", "Medium": "#E69F00", "High": "#009E73"}


def save(fig, name):
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ROOT / "outputs" / "figures" / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pca_scree(result):
    proportion = np.asarray(result["pca"]["proportion_variance"])
    cumulative = np.asarray(result["pca"]["cumulative_variance"])
    labels = [f"PC{i}" for i in range(1, len(proportion) + 1)]
    needed = np.searchsorted(cumulative, 0.90) + 1
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(labels, proportion, color="#0072B2", width=0.7)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Proportion of variance explained")
    ax.set_title("PCA scree plot")
    ax2 = ax.twinx()
    ax2.plot(labels, cumulative, color="#D55E00", marker="o", linewidth=1.5)
    ax2.axhline(0.90, color="0.35", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_ylim(0, 1)
    fig.text(
        0.5,
        0.01,
        f"PC1 explains {proportion[0]:.0%}; {needed} components are needed for 90%, consistent with near-orthogonal predictors.",
        ha="center",
        fontsize=9,
    )
    save(fig, "pca_scree.png")


def pca_biplot(result, tier):
    pca = result["pca"]
    scores = np.asarray(pca["scores"])
    loadings = np.asarray(pca["loadings"])[:, :2]
    explained = np.asarray(pca["proportion_variance"])
    scale = 0.63 * min(np.ptp(scores[:, 0]), np.ptp(scores[:, 1]))
    short = []
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, color in PALETTE.items():
        mask = tier == label
        ax.scatter(scores[mask, 0], scores[mask, 1], s=14, alpha=0.5, color=color, label=label)
    for name, vector in zip(pca["features"], loadings):
        x, y = vector * scale
        ax.arrow(0, 0, x, y, color="0.2", width=0.006, head_width=0.11, length_includes_head=True)
        if np.linalg.norm(vector) >= 0.2:
            ax.annotate(name, (x, y), xytext=(0, 6 if y >= 0 else -10), textcoords="offset points", fontsize=8, ha="center")
        else:
            short.append(name)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title("PCA biplot by risk tier")
    ax.legend(title="Tier", frameon=False)
    fig.text(0.5, 0.01, f"PC1 and PC2 chiefly reflect regional contrasts; short loadings ({', '.join(short)}) are unlabelled.", ha="center", fontsize=9)
    save(fig, "pca_biplot.png")


def feature_correlation(F):
    corr = F.corr().to_numpy()
    labels = list(F.columns)
    fig, ax = plt.subplots(figsize=(7.0, 6.1))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(corr):
        for j, value in enumerate(row):
            color = "white" if abs(value) > 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title("Feature correlation matrix")
    fig.text(0.5, 0.01, "Near-zero off-diagonal correlations explain why PCA offers little compression.", ha="center", fontsize=9)
    save(fig, "feature_correlation.png")


def qda_regions(result, tier):
    scores = np.asarray(result["pca"]["scores"])
    labels = list(tier.cat.categories)
    codes = tier.cat.codes.to_numpy()
    model = QuadraticDiscriminantAnalysis(reg_param=0.001).fit(scores, codes)
    x_pad, y_pad = 0.5, 0.5
    x_min, x_max = scores[:, 0].min() - x_pad, scores[:, 0].max() + x_pad
    y_min, y_max = scores[:, 1].min() - y_pad, scores[:, 1].max() + y_pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    regions = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap = ListedColormap([PALETTE[label] for label in labels])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.contourf(xx, yy, regions, levels=np.arange(len(labels) + 1) - 0.5, cmap=cmap, alpha=0.18)
    for code, label in enumerate(labels):
        mask = codes == code
        ax.scatter(scores[mask, 0], scores[mask, 1], s=14, alpha=0.55, color=PALETTE[label], label=label)
    explained = np.asarray(result["pca"]["proportion_variance"])
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title("Illustrative QDA regions in two PCs")
    ax.legend(title="Tier", frameon=False)
    fig.text(0.5, 0.01, "Illustrative only: QDA is trained on two PCs here; the reported model uses all eight features.", ha="center", fontsize=9)
    save(fig, "qda_regions.png")


def mahalanobis_hist(result):
    mahalanobis = result["mahalanobis"]
    scores = np.asarray(mahalanobis["scores"])
    threshold = mahalanobis["threshold"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(scores, bins=28, color="#0072B2", edgecolor="white")
    ax.axvline(threshold, color="#D55E00", linestyle="--", linewidth=1.5, label=r"$\chi^2_{0.975}$ (df=3) cutoff")
    ax.text(0.97, 0.92, f"Outliers: {mahalanobis['count']}", transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("Squared Mahalanobis distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Within-tier Mahalanobis distances")
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.01, "Distances are measured from each record to its own tier centroid.", ha="center", fontsize=9)
    save(fig, "mahalanobis_hist.png")


def main():
    with (ROOT / "outputs" / "results.json").open() as file:
        result = json.load(file)
    _, F, tier, _, _, _, _ = build_data()
    (ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    pca_scree(result)
    pca_biplot(result, tier)
    feature_correlation(F)
    qda_regions(result, tier)
    mahalanobis_hist(result)


if __name__ == "__main__":
    main()
