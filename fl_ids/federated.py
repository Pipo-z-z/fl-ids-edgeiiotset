"""
Entrenamiento centralizado, local (aislado) y federado (FedAvg / FedProx)
del IDS, más las métricas del instrumento de observación.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import f1_score

from .flows import FEATURES
from .model import DTYPE, MLP

BYTES_PER_PARAM = 4  # float32


# ---------------------------------------------------------------------------
# Preprocesamiento: log con signo + estandarización
# ---------------------------------------------------------------------------
def to_matrix(df):
    X = df[FEATURES].to_numpy(dtype=np.float64)
    return (np.sign(X) * np.log1p(np.abs(X))).astype(DTYPE)


def local_stats(X):
    """Lo único que el cliente comparte para la estandarización:
    n, suma y suma de cuadrados por característica (no los datos)."""
    X = X.astype(np.float64)
    return len(X), X.sum(axis=0), (X * X).sum(axis=0)


def merge_stats(stats):
    n = sum(s[0] for s in stats)
    s1 = sum(s[1] for s in stats)
    s2 = sum(s[2] for s in stats)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(DTYPE), std.astype(DTYPE)


def standardize(X, scaler):
    mean, std = scaler
    return ((X - mean) / std).astype(DTYPE)


# ---------------------------------------------------------------------------
# Métricas del instrumento
# ---------------------------------------------------------------------------
def client_metrics(y_true, y_pred, unseen_ids, normal_id: int = 0) -> dict:
    present = np.unique(y_true)
    f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0) * 100
    ben = y_true == normal_id
    fpr = float((y_pred[ben] != normal_id).mean() * 100) if ben.any() else np.nan
    uns = [u for u in unseen_ids if u in set(present)]
    f1_nv = (f1_score(y_true, y_pred, labels=uns, average="macro", zero_division=0) * 100
             if uns else np.nan)
    return {"F1": f1, "FPR": fpr, "F1nv": f1_nv}


def evaluate(model, scaler, clients_data):
    rows = []
    for c in clients_data:
        yp = model.predict(standardize(c["Xte"], scaler))
        rows.append({"cliente": c["name"], **client_metrics(c["yte"], yp, c["unseen_ids"])})
    return rows


# ---------------------------------------------------------------------------
# Esquemas de entrenamiento
# ---------------------------------------------------------------------------
def run_centralized(clients_data, n_classes, cfg, seed):
    """Cota superior: todos los datos de entrenamiento en un servidor."""
    X = np.vstack([c["Xtr"] for c in clients_data])
    y = np.concatenate([c["ytr"] for c in clients_data])
    scaler = merge_stats([local_stats(X)])
    model = MLP(X.shape[1], n_classes, cfg["hidden"], seed=seed)
    epochs = cfg["rounds"] * cfg["local_epochs"]
    t0 = time.perf_counter()
    model.fit(standardize(X, scaler), y, epochs=epochs, lr=cfg["lr"], batch=cfg["batch"], seed=seed)
    elapsed = time.perf_counter() - t0
    # Volumen si cada cliente enviara sus registros de flujo al servidor
    flow_records = X.size * BYTES_PER_PARAM
    raw_bytes = float(sum(c["raw_bytes"] for c in clients_data))
    return model, scaler, {"comm_MB": flow_records / 1e6, "comm_raw_MB": raw_bytes / 1e6,
                           "t_ronda_s": elapsed / cfg["rounds"]}


def run_local(clients_data, n_classes, cfg, seed):
    """Línea base (pre-test): cada cliente entrena solo con sus datos."""
    models, times = [], []
    epochs = cfg["rounds"] * cfg["local_epochs"]
    for k, c in enumerate(clients_data):
        scaler = merge_stats([local_stats(c["Xtr"])])
        m = MLP(c["Xtr"].shape[1], n_classes, cfg["hidden"], seed=seed)
        t0 = time.perf_counter()
        m.fit(standardize(c["Xtr"], scaler), c["ytr"], epochs=epochs, lr=cfg["lr"],
              batch=cfg["batch"], seed=seed * 1000 + k)
        times.append((time.perf_counter() - t0) / cfg["rounds"])
        models.append((m, scaler))
    return models, {"comm_MB": 0.0, "comm_raw_MB": 0.0, "t_ronda_s": float(np.mean(times))}


def run_federated(clients_data, n_classes, cfg, seed, mu: float = 0.0, track: bool = True):
    """FedAvg (mu = 0) o FedProx (mu > 0), McMahan et al. (2017); Li et al. (2020).

    Por ronda: el servidor difunde w_global -> cada cliente entrena E épocas
    locales -> el servidor promedia los pesos ponderando por n_k.
    """
    K = len(clients_data)
    scaler = merge_stats([local_stats(c["Xtr"]) for c in clients_data])  # estandarización federada
    Xs = [standardize(c["Xtr"], scaler) for c in clients_data]
    n_k = np.array([len(c["ytr"]) for c in clients_data], dtype=float)
    w_k = n_k / n_k.sum()
    glob = MLP(Xs[0].shape[1], n_classes, cfg["hidden"], seed=seed)
    local = MLP(Xs[0].shape[1], n_classes, cfg["hidden"], seed=seed)
    history, times = [], []
    for r in range(1, cfg["rounds"] + 1):
        gparams = glob.get_params()
        new = [np.zeros_like(p) for p in gparams]
        for k, c in enumerate(clients_data):
            local.set_params(gparams)
            t0 = time.perf_counter()
            local.fit(Xs[k], c["ytr"], epochs=cfg["local_epochs"], lr=cfg["lr"], batch=cfg["batch"],
                      seed=seed * 100000 + r * 100 + k, mu=mu, global_params=gparams)
            times.append(time.perf_counter() - t0)
            for j, p in enumerate(local.params):
                new[j] += w_k[k] * p
        glob.set_params(new)
        if track:
            ev = evaluate(glob, scaler, clients_data)
            history.append({"ronda": r, "F1": np.mean([e["F1"] for e in ev]),
                            "F1nv": np.nanmean([e["F1nv"] for e in ev]),
                            "FPR": np.mean([e["FPR"] for e in ev])})
    n_feat = Xs[0].shape[1]
    comm = 2 * K * cfg["rounds"] * glob.n_params * BYTES_PER_PARAM  # bajada + subida
    comm += K * (2 * n_feat + 1) * 8                                 # estadísticas del escalador
    return glob, scaler, history, {"comm_MB": comm / 1e6, "comm_raw_MB": comm / 1e6,
                                   "t_ronda_s": float(np.mean(times))}
