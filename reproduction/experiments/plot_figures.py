import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# Match paper: three shades of blue for ID classes
CLASS_COLORS = ['#aec6e8', '#4a90d9', '#1a3f7a']


class LinearModel(nn.Module):
    def __init__(self, in_features=512, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def get_logits(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model(X_t).numpy()


def plot_panel(ax, logits_id, logits_ood, y_test, title):
    pca = PCA(n_components=2)
    pca.fit(logits_id)
    id_2d  = pca.transform(logits_id)
    ood_2d = pca.transform(logits_ood)

    margin = 1.0
    all_2d = np.concatenate([id_2d, ood_2d])
    x_min, x_max = all_2d[:, 0].min() - margin, all_2d[:, 0].max() + margin
    y_min, y_max = all_2d[:, 1].min() - margin, all_2d[:, 1].max() + margin

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_logits = pca.inverse_transform(np.column_stack([xx.ravel(), yy.ravel()]))
    grid_exp    = np.exp(grid_logits - grid_logits.max(axis=1, keepdims=True))
    grid_probs  = grid_exp / grid_exp.sum(axis=1, keepdims=True)
    msp_grid    = grid_probs.max(axis=1).reshape(xx.shape)

    cf = ax.contourf(xx, yy, msp_grid, levels=50, cmap='Blues', vmin=0.4, vmax=1.0)

    for c in range(3):
        mask = y_test == c
        ax.scatter(id_2d[mask, 0], id_2d[mask, 1],
                   c=CLASS_COLORS[c], s=12, alpha=0.85,
                   linewidths=0, zorder=3)

    ax.scatter(ood_2d[:, 0], ood_2d[:, 1],
               c='red', s=8, alpha=0.35, marker='x',
               linewidths=0.6, zorder=2)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=8)
    return cf


def main():
    os.makedirs('results', exist_ok=True)

    data = np.load('data/features.npz')
    X_test, y_test = data['X_test'], data['y_test']
    X_ood = data['X_ood']

    with open('results/auroc.json') as f:
        auroc = json.load(f)

    panels = [
        ('models/head_n30.pt',    f"Train Points: 30\nLaplace AUROC {auroc['laplace_n30']:.3f}"),
        ('models/head_n10000.pt', f"Train Points: 10000\nLaplace AUROC {auroc['laplace_n10000']:.3f}"),
        ('models/head_map.pt',    f"MAP AUROC {auroc['map']:.3f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    cf = None

    for ax, (model_path, title) in zip(axes, panels):
        checkpoint = torch.load(model_path, weights_only=False)
        model = LinearModel()
        model.load_state_dict(checkpoint['state_dict'])
        cf = plot_panel(ax, get_logits(model, X_test), get_logits(model, X_ood), y_test, title)

    # Shared colorbar on the right
    cbar = fig.colorbar(cf, ax=axes.tolist(), fraction=0.018, pad=0.02)
    cbar.set_label('MSP', fontsize=9)
    cbar.set_ticks([0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=8)

    # Legend at the bottom matching the paper
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[0], markersize=7, label='Airplane (ID)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[1], markersize=7, label='Dog (ID)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS[2], markersize=7, label='Truck (ID)'),
        Line2D([0], [0], marker='x', color='red', markersize=7, linestyle='None', label='CIFAR100 (OOD)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.46, -0.04), fontsize=9, frameon=False)

    plt.savefig('results/figure6.png', dpi=150, bbox_inches='tight')
    print("Saved results/figure6.png")


if __name__ == '__main__':
    main()
