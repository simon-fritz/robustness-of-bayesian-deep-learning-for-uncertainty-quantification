import os
import numpy as np
import torch
import torch.nn as nn

SUBSET_SIZES = [30, 100, 10000]


class LinearModel(nn.Module):
    def __init__(self, in_features=512, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class LastLayerLaplace:
    """
    Manual last-layer Laplace approximation for a linear classifier.
    Uses the closed-form GGN Hessian (no laplace-torch required).
    """

    def __init__(self, model):
        self.model = model
        self.prior_precision = 1.0
        self.posterior_cov = None
        self.theta_map = None
        self.K = None
        self.D = None

    def fit(self, X, y):
        W = self.model.fc.weight.detach()  # (K, D)
        b = self.model.fc.bias.detach()    # (K,)
        self.K, self.D = W.shape
        K, D = self.K, self.D

        X_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.softmax(X_t @ W.T + b, dim=1)  # (N, K)

        n_params = K * D + K
        H = torch.zeros(n_params, n_params)

        for k in range(K):
            for l in range(K):
                lam = probs[:, k] * (float(k == l) - probs[:, l])  # (N,)
                H[k*D:(k+1)*D, l*D:(l+1)*D] = (X_t * lam.unsqueeze(1)).T @ X_t
                H[k*D:(k+1)*D, K*D + l]     = (X_t * lam.unsqueeze(1)).sum(0)
                H[K*D + k,     K*D + l]      = lam.sum()

        H[K*D:, :K*D] = H[:K*D, K*D:].T

        self.H_ggn = H
        self.theta_map = torch.cat([W.flatten(), b])
        self.n_params = n_params

    def optimize_prior_precision(self, method='marglik'):
        best_score = float('inf')
        best_log_prior = 0.0

        for log_prior in np.linspace(-4, 4, 40):
            prior = float(np.exp(log_prior))
            H = self.H_ggn + prior * torch.eye(self.n_params)
            sign, log_det = torch.linalg.slogdet(H)
            if sign <= 0:
                continue
            score = 0.5 * log_det.item() - 0.5 * self.n_params * log_prior
            if score < best_score:
                best_score = score
                best_log_prior = log_prior

        self.prior_precision = float(np.exp(best_log_prior))
        H_post = self.H_ggn + self.prior_precision * torch.eye(self.n_params)
        self.posterior_cov = torch.linalg.inv(H_post)
        print(f"  prior_precision = {self.prior_precision:.4f}")

    def predict_proba(self, X, n_samples=100):
        X_t = torch.tensor(X, dtype=torch.float32)
        dist = torch.distributions.MultivariateNormal(
            self.theta_map, covariance_matrix=self.posterior_cov
        )
        probs_list = []
        for _ in range(n_samples):
            theta = dist.sample()
            W = theta[:self.K * self.D].reshape(self.K, self.D)
            b = theta[self.K * self.D:]
            with torch.no_grad():
                probs_list.append(torch.softmax(X_t @ W.T + b, dim=1))
        return torch.stack(probs_list).mean(0).numpy()


def main():
    os.makedirs('models', exist_ok=True)

    for n in SUBSET_SIZES:
        print(f"Fitting Laplace for n={n}...")
        checkpoint = torch.load(f'models/head_n{n}.pt', weights_only=False)

        model = LinearModel()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        la = LastLayerLaplace(model)
        la.fit(checkpoint['X_sub'], checkpoint['y_sub'])
        la.optimize_prior_precision(method='marglik')

        torch.save(la, f'models/laplace_n{n}.pt')
        print(f"  Saved models/laplace_n{n}.pt")


if __name__ == '__main__':
    main()
