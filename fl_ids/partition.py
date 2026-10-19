"""
División entrenamiento/prueba y reparto de datos entre clientes
(dispositivos IoT) en escenarios IID y no-IID.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split


def split_train_test(data: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """División estratificada por clase y dispositivo."""
    key = data["label"].astype(str) + "|" + data["device"].fillna("-").astype(str)
    tr, te = train_test_split(data.index, test_size=test_size, random_state=seed, stratify=key)
    return data.loc[tr].reset_index(drop=True), data.loc[te].reset_index(drop=True)


def assign_attacks(attacks, clients, seed=None) -> dict:
    """Reparte los tipos de ataque entre clientes de forma DISJUNTA
    (round-robin sobre la lista ordenada; con `seed` se baraja antes).
    Cada ataque lo observa un solo cliente -> sesgo de etiquetas no-IID."""
    attacks = sorted(attacks)
    if seed is not None:
        attacks = list(np.random.RandomState(seed).permutation(attacks))
    out = {c: [] for c in clients}
    for i, a in enumerate(attacks):
        out[clients[i % len(clients)]].append(a)
    return out


def make_clients(train: pd.DataFrame, test: pd.DataFrame, clients, assignment: dict,
                 mode: str = "noniid", seed: int = 0) -> list[dict]:
    """Construye los datos de cada cliente.

    no-IID: el cliente k entrena con el tráfico normal de SU sensor y solo
            con los ataques que le fueron asignados.
    IID   : el mismo conjunto total de entrenamiento se baraja y se reparte
            en K partes iguales (cada cliente ve todas las clases).
    Prueba (igual en ambos modos): tráfico normal de su sensor + TODOS los
    ataques (un ataque puede golpear a cualquier dispositivo). Así se mide
    la detección de ataques que el cliente nunca vio localmente.
    """
    all_attacks = sorted({a for v in assignment.values() for a in v})
    out = []
    parts = {}
    for c in clients:
        m = ((train["label"] == "Normal") & (train["device"] == c)) | train["label"].isin(assignment[c])
        parts[c] = train[m]
    if mode == "iid":
        pool = pd.concat(parts.values()).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        chunks = np.array_split(np.arange(len(pool)), len(clients))
        parts = {c: pool.iloc[ix] for c, ix in zip(clients, chunks)}
    elif mode != "noniid":
        raise ValueError(mode)
    test_att = test[test["label"].isin(all_attacks)]
    for c in clients:
        te = pd.concat([test[(test["label"] == "Normal") & (test["device"] == c)], test_att])
        out.append({
            "name": c,
            "train": parts[c].reset_index(drop=True),
            "test": te.reset_index(drop=True),
            "seen_attacks": list(assignment[c]),
            "unseen_attacks": [a for a in all_attacks if a not in assignment[c]],
        })
    return out


def label_distribution(clients: list[dict], classes: list[str]) -> pd.DataFrame:
    rows = {c["name"]: c["train"]["label"].value_counts().reindex(classes, fill_value=0) for c in clients}
    return pd.DataFrame(rows).T


def js_divergence(dist: pd.DataFrame) -> pd.Series:
    """Divergencia de Jensen-Shannon (base 2, rango [0, 1]) entre la
    distribución de clases de cada cliente y la distribución global."""
    glob = dist.sum(axis=0).to_numpy(dtype=float)
    glob /= glob.sum()
    vals = {}
    for c, row in dist.iterrows():
        p = row.to_numpy(dtype=float)
        p /= p.sum()
        vals[c] = float(jensenshannon(p, glob, base=2) ** 2)
    return pd.Series(vals, name="JS")
