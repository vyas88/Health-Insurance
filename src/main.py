import json
from pathlib import Path

import joblib
import numpy as np

from assumptions import box_m, mardia_test
from classify import fit_classifier
from data import build_data
from inference import hotelling_t2, mahalanobis, manova_wilks
from reduction import pca


ROOT = Path(__file__).resolve().parents[1]


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main():
    C, F, tier, _, _, scaler, edges = build_data()
    mardia = mardia_test(C)
    box = box_m(C, tier)
    hotelling = hotelling_t2(C[tier == "High"], C[tier == "Low"])
    pca_result = pca(F)
    model, classification = fit_classifier(F, tier, use_qda=box["p_value"] < 0.05)
    distances = mahalanobis(C, tier)
    manova = manova_wilks(C, tier)

    results = {
        "data": {
            "n": int(C.attrs["n_rows"]),
            "imputed_cells": int(C.attrs["imputed_cells"]),
            "tercile_edges": edges,
        },
        "mardia": mardia,
        "box_m": box,
        "hotelling_t2": hotelling,
        "pca": pca_result,
        "classification": classification,
        "mahalanobis": distances,
        "manova": manova,
    }

    output_dir = ROOT / "outputs"
    model_dir = ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(json_ready(results), indent=2) + "\n")
    centroids = {
        label: {"mean": C[tier == label].mean().to_numpy().tolist(), "cov": C[tier == label].cov().to_numpy().tolist()}
        for label in tier.cat.categories
    }
    joblib.dump({"classifier": model, "scaler": scaler, "tier_edges": edges, "features": list(F.columns), "centroids": centroids}, model_dir / "model.joblib")

    print(f"n: {results['data']['n']}")
    print(f"Mardia skewness p: {mardia['skewness_p_value']:.4g}")
    print(f"Mardia kurtosis p: {mardia['kurtosis_p_value']:.4g}")
    print(f"Box's M p: {box['p_value']:.4g}")
    print(f"Hotelling T2 p: {hotelling['p_value']:.4g}")
    print(f"PC1 variance: {pca_result['proportion_variance'][0]:.4f}")
    print(f"{classification['method']} CV accuracy: {classification['cv_accuracy_mean']:.4f} ({classification['cv_accuracy_sd']:.4f})")
    print(f"Mahalanobis outliers: {distances['count']}")
    print(f"MANOVA Wilks' lambda: {manova['wilks_lambda']:.4f}, p: {manova['p_value']:.4g}")


if __name__ == "__main__":
    main()
