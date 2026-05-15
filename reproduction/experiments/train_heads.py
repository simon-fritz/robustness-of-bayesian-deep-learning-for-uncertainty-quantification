import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SUBSET_SIZES = [30, 100, 10000]
EPOCHS = 200
LR = 1e-3


class LinearModel(nn.Module):
    def __init__(self, in_features=512, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def stratified_sample(X, y, n):
    classes = np.unique(y)
    per_class = n // len(classes)
    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        chosen = np.random.choice(idx, size=per_class, replace=False)
        indices.extend(chosen.tolist())
    return X[indices], y[indices]


def train_head(X, y, epochs=EPOCHS, lr=LR):
    model = LinearModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    return model


def main():
    os.makedirs('models', exist_ok=True)
    np.random.seed(42)

    data = np.load('data/features.npz')
    X_train, y_train = data['X_train'], data['y_train']

    for n in SUBSET_SIZES:
        print(f"Training head on {n} samples...")
        X_sub, y_sub = stratified_sample(X_train, y_train, n)
        model = train_head(X_sub, y_sub)
        torch.save({'state_dict': model.state_dict(), 'X_sub': X_sub, 'y_sub': y_sub},
                   f'models/head_n{n}.pt')
        print(f"  Saved models/head_n{n}.pt")

    # MAP baseline = the largest trained head (no Laplace on top)
    import shutil
    shutil.copy(f'models/head_n{max(SUBSET_SIZES)}.pt', 'models/head_map.pt')
    print("Saved models/head_map.pt (MAP baseline)")


if __name__ == '__main__':
    main()
