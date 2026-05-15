import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from fit_laplace import LastLayerLaplace, LinearModel

SUBSET_SIZES = [30, 100, 10000]


def compute_auroc(id_scores, ood_scores):
    labels = [1] * len(id_scores) + [0] * len(ood_scores)
    scores = list(id_scores) + list(ood_scores)
    return roc_auc_score(labels, scores)


def laplace_msp(la, X):
    probs = la.predict_proba(X, n_samples=100)
    return probs.max(axis=1)


def map_msp(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1)
    return probs.numpy().max(axis=1)


def main():
    os.makedirs('results', exist_ok=True)

    data = np.load('data/features.npz')
    X_test, X_ood = data['X_test'], data['X_ood']

    results = {}

    for n in SUBSET_SIZES:
        print(f"Evaluating Laplace n={n}...")
        la = torch.load(f'models/laplace_n{n}.pt', weights_only=False)
        auroc = compute_auroc(laplace_msp(la, X_test), laplace_msp(la, X_ood))
        results[f'laplace_n{n}'] = round(float(auroc), 4)
        print(f"  AUROC = {auroc:.4f}")

    print("Evaluating MAP baseline...")
    checkpoint = torch.load('models/head_map.pt', weights_only=False)
    model = LinearModel()
    model.load_state_dict(checkpoint['state_dict'])
    auroc = compute_auroc(map_msp(model, X_test), map_msp(model, X_ood))
    results['map'] = round(float(auroc), 4)
    print(f"  AUROC = {auroc:.4f}")

    with open('results/auroc.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("Saved results/auroc.json")


if __name__ == '__main__':
    main()
