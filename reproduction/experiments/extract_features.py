import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

CIFAR10_CLASSES = [0, 5, 9]  # airplane, dog, truck
LABEL_MAP = {0: 0, 5: 1, 9: 2}


def build_backbone():
    backbone = models.resnet18(weights='IMAGENET1K_V1')
    backbone.fc = nn.Identity()
    backbone.eval()
    return backbone


def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def filter_indices(dataset, classes):
    return [i for i, (_, label) in enumerate(dataset) if label in classes]


def extract(backbone, dataset, indices=None, label_map=None):
    if indices is not None:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            features.append(backbone(x).cpu().numpy())
            if label_map is not None:
                y = torch.tensor([label_map[int(yi)] for yi in y])
            labels.append(y.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)


def main():
    os.makedirs('data', exist_ok=True)
    transform = get_transform()
    backbone = build_backbone()

    cifar10_train = datasets.CIFAR10(root='data/raw', train=True,  download=True, transform=transform)
    cifar10_test  = datasets.CIFAR10(root='data/raw', train=False, download=True, transform=transform)
    cifar100_test = datasets.CIFAR100(root='data/raw', train=False, download=True, transform=transform)

    train_idx = filter_indices(cifar10_train, CIFAR10_CLASSES)
    test_idx  = filter_indices(cifar10_test,  CIFAR10_CLASSES)

    print("Extracting CIFAR-10 train features...")
    X_train, y_train = extract(backbone, cifar10_train, train_idx, LABEL_MAP)

    print("Extracting CIFAR-10 test features...")
    X_test, y_test = extract(backbone, cifar10_test, test_idx, LABEL_MAP)

    print("Extracting CIFAR-100 OOD features...")
    X_ood, _ = extract(backbone, cifar100_test)

    np.savez('data/features.npz',
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test,
             X_ood=X_ood)

    print(f"Done. X_train={X_train.shape}, X_test={X_test.shape}, X_ood={X_ood.shape}")


if __name__ == '__main__':
    main()
