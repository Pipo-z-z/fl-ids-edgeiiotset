"""
Resúmenes, pruebas estadísticas, registro de observación (pre/post) y figuras.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import wilcoxon

# Colores fijos por esquema (el color sigue al método, nunca a su posición)
METHOD_COLORS = {
    "Centralizado": "#52514e",
    "Local": "#eb6834",
    "FedAvg (IID)": "#1baf7a",
    "FedAvg (no-IID)": "#2a78d6",
}
_EXTRA = ["#6250d6", "#e87ba4", "#eda100", "#008300"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def color_of(method: str, methods: list[str]) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    extra = [m for m in methods if m not in METHOD_COLORS]
    return _EXTRA[extra.index(method) % len(_EXTRA)]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
def summary(res: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Media ± desviación estándar entre semillas del promedio por cliente."""
    order = list(dict.fromkeys(res["metodo"]))
    per_seed = res.groupby(["metodo", "semilla"])[["F1", "FPR", "dF1", "F1nv"]].mean()
    per_seed = per_seed.join(info.set_index(["metodo", "semilla"])[["comm_MB", "t_ronda_s"]])
    agg = per_seed.groupby("metodo").agg(["mean", "std"]).reindex(order)
    out = pd.DataFrame(index=agg.index)
    names = {"F1": "F1 macro (%)", "FPR": "FPR (%)", "dF1": "Brecha vs. central (pp)",
             "F1nv": "F1 ataques no vistos (%)", "comm_MB": "Comunicación (MB)",
             "t_ronda_s": "Tiempo local/ronda (s)"}
    for k, v in names.items():
        out[v] = [f"{m:.2f} ± {s:.2f}" if not np.isnan(s) else f"{m:.2f}"
                  for m, s in zip(agg[(k, "mean")], agg[(k, "std")])]
    out.index.name = "Esquema"
    return out


def wilcoxon_tests(res: pd.DataFrame, comparisons=None, alpha: float = 0.05) -> pd.DataFrame:
    """Prueba de rangos con signo de Wilcoxon, pareada por (semilla, cliente)."""
    methods = list(dict.fromkeys(res["metodo"]))
    prox = [m for m in methods if m.startswith("FedProx")]
    if comparisons is None:
        comparisons = [("Local", "FedAvg (no-IID)", "F1"), ("Local", "FedAvg (no-IID)", "F1nv"),
                       ("Local", "FedAvg (no-IID)", "FPR"), ("Local", "FedAvg (no-IID)", "dF1"),
                       ("FedAvg (IID)", "FedAvg (no-IID)", "F1")]
        comparisons += [("FedAvg (no-IID)", p, "F1") for p in prox]
        comparisons += [("FedAvg (no-IID)", p, "F1nv") for p in prox]
    piv = {m: res[res["metodo"] == m].set_index(["semilla", "cliente"]) for m in methods}
    rows = []
    for a, b, metric in comparisons:
        x = piv[a][metric]
        y = piv[b][metric].reindex(x.index)
        ok = ~(x.isna() | y.isna())
        d = (y - x)[ok]
        if len(d) < 2 or np.allclose(d, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x[ok], y[ok])
        rows.append({"comparación": f"{a} → {b}", "métrica": metric, "n pares": int(ok.sum()),
                     "media A": round(x[ok].mean(), 2), "media B": round(y[ok].mean(), 2),
                     "mediana Δ (B−A)": round(float(np.median(d)), 2) if len(d) else np.nan,
                     "W": stat, "p-valor": p, f"significativo (α={alpha})": "Sí" if p < alpha else "No"})
    return pd.DataFrame(rows)


LEVELS = {  # índices de la matriz de operacionalización
    "F1": [(95, "Alto (Óptimo)"), (85, "Medio (Aceptable)"), (-1, "Bajo (Insuficiente)")],
    "FPR": [(5, "Deficiente"), (1, "Aceptable"), (-1, "Óptimo")],
    "dF1": [(10, "Deficiente"), (3, "Aceptable"), (-1e9, "Óptimo")],
    "F1nv": [(80, "Alto (Óptimo)"), (50, "Medio"), (-1, "Bajo")],
    "comm_MB": [(100, "Deficiente"), (10, "Aceptable"), (-1, "Óptimo")],
    "t_ronda_s": [(30, "Deficiente"), (5, "Aceptable"), (-1, "Óptimo")],
}


def level(metric: str, value: float) -> str:
    if value is None or np.isnan(value):
        return "-"
    for thr, name in LEVELS[metric]:
        if value > thr:
            return name
    return LEVELS[metric][-1][1]


def observation_register(res, info, pre="Local", post="FedAvg (no-IID)", pre_comm="Centralizado"):
    """Llena la 'Matriz del registro de observación estructurada (pre vs. post)'
    con los promedios entre semillas."""
    labels = {"F1": "Desempeño: F1-score macro (%)", "FPR": "Desempeño: Tasa de falsos positivos (%)",
              "dF1": "Generalización: Brecha frente al modelo centralizado (pp)",
              "F1nv": "Generalización: F1 en ataques no vistos localmente (%)"}
    m = res.groupby(["metodo", "cliente"], sort=False)[list(labels)].mean()
    rows = []
    clients = list(dict.fromkeys(res["cliente"]))

    def add(group, name, a, b, metric):
        d = b - a
        # con una línea base casi nula (< 1 unidad) la variación relativa no tiene sentido
        rel = d / abs(a) * 100 if (not np.isnan(a) and abs(a) >= 1) else np.nan
        rows.append({"Bloque": group, "Indicador": name, "Pre-test": round(a, 2), "Post-test": round(b, 2),
                     "Diferencia (Δ)": round(d, 2), "Variación relativa (%)": round(rel, 1) if not np.isnan(rel) else "N/A",
                     "Nivel post-test": level(metric, b)})

    for i, c in enumerate(clients, 1):
        for k, lab in labels.items():
            add(f"CLIENTE {i}: {c}", lab, m.loc[(pre, c), k], m.loc[(post, c), k], k)
    tot = res.groupby("metodo")[list(labels)].mean()
    for j, (k, lab) in enumerate(labels.items(), 1):
        add("TOTALES", f"{j}. {lab} (promedio)", tot.loc[pre, k], tot.loc[post, k], k)
    inf = info.groupby("metodo")[["comm_MB", "t_ronda_s"]].mean()
    add("TOTALES", "5. Volumen de comunicación acumulado (MB)*", inf.loc[pre_comm, "comm_MB"],
        inf.loc[post, "comm_MB"], "comm_MB")
    add("TOTALES", "6. Tiempo de entrenamiento local promedio por ronda (s)", inf.loc[pre, "t_ronda_s"],
        inf.loc[post, "t_ronda_s"], "t_ronda_s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_convergence(history, res, metric="F1", path=None):
    methods = list(dict.fromkeys(res["metodo"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for meth in [m for m in methods if m in set(history["metodo"])]:
        h = history[history["metodo"] == meth].groupby("ronda")[metric].agg(["mean", "std"])
        col = color_of(meth, methods)
        ax.plot(h.index, h["mean"], color=col, linewidth=2, label=meth)
        ax.fill_between(h.index, h["mean"] - h["std"], h["mean"] + h["std"], color=col, alpha=0.12, linewidth=0)
    for ref, ls in [("Centralizado", "--"), ("Local", ":")]:
        if ref in methods:
            v = res[res["metodo"] == ref][metric].mean()
            ax.axhline(v, color=color_of(ref, methods), linestyle=ls, linewidth=1.5, label=f"{ref} (final)")
    ax.set_xlabel("Ronda de comunicación", color="#52514e")
    ax.set_ylabel({"F1": "F1 macro promedio (%)", "F1nv": "F1 en ataques no vistos (%)",
                   "FPR": "FPR promedio (%)"}[metric], color="#52514e")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_client_bars(res, metric="F1", methods=None, path=None):
    allm = list(dict.fromkeys(res["metodo"]))
    methods = methods or allm
    clients = list(dict.fromkeys(res["cliente"]))
    m = res.groupby(["cliente", "metodo"])[metric].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    w = 0.8 / len(methods)
    x = np.arange(len(clients))
    for i, meth in enumerate(methods):
        vals = [m.loc[(c, meth), "mean"] for c in clients]
        err = [m.loc[(c, meth), "std"] for c in clients]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w * 0.92, yerr=err, color=color_of(meth, allm),
               label=meth, error_kw={"elinewidth": 0.8, "ecolor": "#52514e", "capsize": 2})
    ax.set_xticks(x, [c.replace("_", " ") for c in clients], fontsize=8)
    ax.set_ylabel({"F1": "F1 macro (%)", "F1nv": "F1 en ataques no vistos (%)", "FPR": "FPR (%)"}[metric],
                  color="#52514e")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=min(3, len(methods)), loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_partition(dist: pd.DataFrame, title: str, path=None):
    share = dist.div(dist.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.55 * dist.shape[1], 11), 0.6 + 0.45 * dist.shape[0]), dpi=150)
    im = ax.imshow(share.to_numpy(), cmap=SEQ_BLUE, aspect="auto", vmin=0, vmax=max(1, share.to_numpy().max()))
    ax.set_xticks(range(dist.shape[1]), [c.replace("_", " ") for c in dist.columns], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(dist.shape[0]), [c.replace("_", " ") for c in dist.index], fontsize=8)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            v = share.iat[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 * share.to_numpy().max() else "#0b0b0b")
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% de los flujos del cliente", fontsize=8, color="#52514e")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion(conf, classes, path=None):
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1) * 100
    n = len(classes)
    fig, ax = plt.subplots(figsize=(0.9 + 0.48 * n, 0.6 + 0.45 * n), dpi=150)
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=100)
    ax.set_xticks(range(n), [c.replace("_", " ") for c in classes], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n), [c.replace("_", " ") for c in classes], fontsize=7)
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 1:
                ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 55 else "#0b0b0b")
    ax.set_xlabel("Clase predicha", color="#52514e")
    ax.set_ylabel("Clase real", color="#52514e")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("% de la clase real", fontsize=8, color="#52514e")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def sensitivity_table(sens: pd.DataFrame, rounds=(10, 30, 60, 100), metric="F1") -> pd.DataFrame:
    s = sens[sens["ronda"].isin(rounds)]
    g = s.groupby(["metodo", "E", "ronda"])[metric].agg(["mean", "std"]).reset_index()
    g["valor"] = [f"{m:.2f} ± {d:.2f}" for m, d in zip(g["mean"], g["std"])]
    out = g.pivot_table(index=["metodo", "E"], columns="ronda", values="valor", aggfunc="first")
    out.columns = [f"R = {c}" for c in out.columns]
    return out


def plot_sensitivity(sens: pd.DataFrame, metric="F1", path=None):
    methods = list(dict.fromkeys(sens["metodo"]))
    Es = sorted(sens["E"].unique())
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 3.9), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, meth in zip(axes, methods):
        for i, E in enumerate(Es):
            h = sens[(sens["metodo"] == meth) & (sens["E"] == E)].groupby("ronda")[metric].agg(["mean", "std"])
            ax.plot(h.index, h["mean"], color=cols[i % len(cols)], linewidth=2, label=f"E = {E}")
            ax.fill_between(h.index, h["mean"] - h["std"], h["mean"] + h["std"], color=cols[i % len(cols)],
                            alpha=0.12, linewidth=0)
        ax.set_title(f"{meth} (no-IID)", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("Ronda de comunicación (R)", color="#52514e")
        _style(ax)
    axes[0].set_ylabel({"F1": "F1 macro promedio (%)", "F1nv": "F1 en ataques no vistos (%)"}[metric], color="#52514e")
    axes[-1].legend(frameon=False, fontsize=8, title="Épocas locales", title_fontsize=8, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
