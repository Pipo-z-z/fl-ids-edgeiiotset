"""
Perceptrón multicapa (MLP) implementado desde cero en NumPy, con
optimizador Adam y término proximal opcional (FedProx).

Se implementa a mano (sin PyTorch/TensorFlow) para que cada operación del
entrenamiento local y de la agregación federada sea explícita y
verificable, y para que el experimento corra en cualquier laptop con CPU.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32


class MLP:
    """MLP n_in -> hidden[0] -> hidden[1] -> ... -> n_out con ReLU y softmax."""

    def __init__(self, n_in: int, n_out: int, hidden=(64, 32), seed: int = 0):
        rng = np.random.RandomState(seed)
        sizes = [n_in, *hidden, n_out]
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = (rng.randn(a, b) * np.sqrt(2.0 / a)).astype(DTYPE)  # inicialización He
            self.params += [W, np.zeros(b, dtype=DTYPE)]
        self.n_out = n_out

    # ----- parámetros (lo que viaja entre cliente y servidor) -----
    def get_params(self):
        return [p.copy() for p in self.params]

    def set_params(self, params):
        self.params = [p.astype(DTYPE, copy=True) for p in params]

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    # ----- inferencia -----
    def _forward(self, X):
        acts = [X]
        h = X
        L = len(self.params) // 2
        for i in range(L):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            z = h @ W + b
            h = np.maximum(z, 0) if i < L - 1 else z
            acts.append(h)
        return acts

    def predict_proba(self, X, batch: int = 65536):
        out = []
        for i in range(0, len(X), batch):
            z = self._forward(X[i:i + batch])[-1]
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            out.append(e / e.sum(axis=1, keepdims=True))
        return np.vstack(out) if out else np.zeros((0, self.n_out), dtype=DTYPE)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # ----- entrenamiento -----
    def _grads(self, X, y):
        acts = self._forward(X)
        z = acts[-1]
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        n = len(X)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        d = p
        d[np.arange(n), y] -= 1.0
        d /= n
        grads = [None] * len(self.params)
        L = len(self.params) // 2
        for i in range(L - 1, -1, -1):
            h_prev = acts[i]
            grads[2 * i] = h_prev.T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.params[2 * i].T) * (acts[i] > 0)
        return loss, grads

    def fit(self, X, y, epochs: int = 1, lr: float = 1e-3, batch: int = 256, seed: int = 0,
            mu: float = 0.0, global_params=None):
        """Entrena `epochs` épocas con Adam (estado nuevo en cada llamada).

        Si mu > 0 se añade el término proximal de FedProx (Li et al., 2020):
            loss_local + (mu/2) * ||w - w_global||^2
        cuyo gradiente es mu * (w - w_global).
        Devuelve la pérdida media de la última época.
        """
        rng = np.random.RandomState(seed)
        m = [np.zeros_like(p) for p in self.params]
        v = [np.zeros_like(p) for p in self.params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        n = len(X)
        last = np.nan
        for _ in range(epochs):
            idx = rng.permutation(n)
            losses = []
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                loss, grads = self._grads(X[bi], y[bi])
                if mu > 0 and global_params is not None:
                    for j in range(len(grads)):
                        grads[j] = grads[j] + mu * (self.params[j] - global_params[j])
                t += 1
                for j, g in enumerate(grads):
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    mhat = m[j] / (1 - b1 ** t)
                    vhat = v[j] / (1 - b2 ** t)
                    self.params[j] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(DTYPE)
                losses.append(loss)
            last = float(np.mean(losses)) if losses else np.nan
        return last
