"""
Orquestación del experimento: para cada semilla se re-divide el dataset,
se construyen los clientes y se entrenan/evalúan todos los esquemas.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .federated import (evaluate, run_centralized, run_federated, run_local,
                        standardize, to_matrix)
from .flows import FEATURES
from .model import MLP
from .partition import assign_attacks, make_clients, split_train_test

DEFAULT_CONFIG = {
    "clients": ["Temperature_and_Humidity", "Distance", "Water_Level", "phValue", "Soil_Moisture"],
    "label_level": "attack",      # "attack" (tipo de ataque) o "category" (5 categorías)
    "test_size": 0.2,
    "seeds": [0, 1, 2, 3, 4],
    "rounds": 30,                 # R
    "local_epochs": 2,            # E
    "lr": 1e-3,
    "batch": 256,
    "hidden": (64, 32),
    "mus": [0.01, 0.1],           # coeficientes proximales de FedProx
    "attack_assignment": None,    # None = round-robin automático
}


def method_names(cfg):
    names = ["Centralizado", "Local", "FedAvg (IID)", "FedAvg (no-IID)"]
    names += [f"FedProx μ={mu} (no-IID)" for mu in cfg["mus"]]
    return names


def _prepare(clients, classes, label_col, cat_of=None):
    cid = {c: i for i, c in enumerate(classes)}
    out = []
    for c in clients:
        tr, te = c["train"], c["test"]
        unseen_labels = c["unseen_attacks"]
        if label_col == "category":
            seen_cats = {cat_of[a] for a in c["seen_attacks"]}
            unseen_labels = sorted({cat_of[a] for a in c["unseen_attacks"]} - seen_cats)
        out.append({
            "name": c["name"],
            "Xtr": to_matrix(tr), "ytr": tr[label_col].map(cid).to_numpy(),
            "Xte": to_matrix(te), "yte": te[label_col].map(cid).to_numpy(),
            "unseen_ids": [cid[u] for u in unseen_labels if u in cid],
            "raw_bytes": float((tr["fwd_bytes"] + tr["bwd_bytes"]).sum()),
            "n_train": len(tr),
        })
    return out


def run_experiment(data: pd.DataFrame, cfg: dict | None = None, verbose: bool = True):
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    label_col = "label" if cfg["label_level"] == "attack" else "category"
    attacks = sorted(data.loc[data["label"] != "Normal", "label"].unique())
    assignment = cfg["attack_assignment"] or assign_attacks(attacks, cfg["clients"])
    classes = ["Normal"] + sorted(data.loc[data[label_col] != "Normal", label_col].unique())
    data = data[(data["label"] != "Normal") | data["device"].isin(cfg["clients"])]
    cat_of = dict(zip(data["label"], data["category"]))

    per_client, history, method_info = [], [], []
    conf = None
    for seed in cfg["seeds"]:
        t_seed = time.time()
        train, test = split_train_test(data, cfg["test_size"], seed)
        noniid = _prepare(make_clients(train, test, cfg["clients"], assignment, "noniid", seed), classes, label_col, cat_of)
        iid = _prepare(make_clients(train, test, cfg["clients"], assignment, "iid", seed), classes, label_col, cat_of)
        C = len(classes)

        def log(method, rows, info):
            for r in rows:
                per_client.append({"semilla": seed, "metodo": method, **r})
            method_info.append({"semilla": seed, "metodo": method, **info})

        m, sc, info = run_centralized(noniid, C, cfg, seed)
        log("Centralizado", evaluate(m, sc, noniid), info)

        models, info = run_local(noniid, C, cfg, seed)
        rows = []
        for (lm, lsc), c in zip(models, noniid):
            rows += evaluate(lm, lsc, [c])
        log("Local", rows, info)

        for name, clients, mu in [("FedAvg (IID)", iid, 0.0), ("FedAvg (no-IID)", noniid, 0.0)] + \
                [(f"FedProx μ={mu} (no-IID)", noniid, mu) for mu in cfg["mus"]]:
            g, gsc, hist, info = run_federated(clients, C, cfg, seed, mu=mu)
            # se evalúa siempre sobre los conjuntos de prueba no-IID (los mismos para todos)
            log(name, evaluate(g, gsc, noniid), info)
            for h in hist:
                history.append({"semilla": seed, "metodo": name, **h})
            if name == "FedAvg (no-IID)" and seed == cfg["seeds"][0]:
                Xall = np.vstack([standardize(c["Xte"], gsc) for c in noniid])
                yall = np.concatenate([c["yte"] for c in noniid])
                conf = confusion_matrix(yall, g.predict(Xall), labels=range(C))
        if verbose:
            done = pd.DataFrame([r for r in per_client if r["semilla"] == seed])
            res = done.groupby("metodo", sort=False)[["F1", "F1nv", "FPR"]].mean().round(2)
            print(f"\n=== Semilla {seed} ({time.time() - t_seed:.0f} s) ===")
            print(res.to_string())

    res = pd.DataFrame(per_client)
    central = res[res["metodo"] == "Centralizado"].set_index(["semilla", "cliente"])["F1"]
    res["dF1"] = central.reindex(pd.MultiIndex.from_frame(res[["semilla", "cliente"]])).to_numpy() - res["F1"]
    return {
        "config": cfg, "classes": classes, "assignment": assignment,
        "per_client": res, "history": pd.DataFrame(history),
        "method_info": pd.DataFrame(method_info), "confusion": conf,
        "n_params": MLP(len(FEATURES), len(classes), cfg["hidden"]).n_params,
    }


def run_sensitivity(data: pd.DataFrame, cfg: dict | None = None, local_epochs=(1, 2, 5), mus=(0.0, 0.01),
                    max_rounds: int = 100, seeds=(0, 1, 2), verbose: bool = True) -> pd.DataFrame:
    """Sensibilidad a la configuración del entrenamiento federado (dimensión 3 de la VI):
    rondas de comunicación R (1..max_rounds) y épocas locales E, en el escenario no-IID.
    El historial por ronda de una ejecución con max_rounds equivale a haber fijado R = r."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    label_col = "label" if cfg["label_level"] == "attack" else "category"
    attacks = sorted(data.loc[data["label"] != "Normal", "label"].unique())
    assignment = cfg["attack_assignment"] or assign_attacks(attacks, cfg["clients"])
    classes = ["Normal"] + sorted(data.loc[data[label_col] != "Normal", label_col].unique())
    data = data[(data["label"] != "Normal") | data["device"].isin(cfg["clients"])]
    cat_of = dict(zip(data["label"], data["category"]))
    rows = []
    for seed in seeds:
        train, test = split_train_test(data, cfg["test_size"], seed)
        noniid = _prepare(make_clients(train, test, cfg["clients"], assignment, "noniid", seed), classes, label_col, cat_of)
        for E in local_epochs:
            for mu in mus:
                t0 = time.time()
                run_cfg = {**cfg, "rounds": max_rounds, "local_epochs": E}
                _, _, hist, _ = run_federated(noniid, len(classes), run_cfg, seed, mu=mu)
                name = "FedAvg" if mu == 0 else f"FedProx μ={mu}"
                for h in hist:
                    rows.append({"semilla": seed, "metodo": name, "E": E, **h})
                if verbose:
                    print(f"semilla {seed} | {name} | E={E}: F1 final {hist[-1]['F1']:.1f} % ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows)
