"""
Inventario de los PCAP de Edge-IIoTset, extracción en lote con caché,
etiquetado de flujos y construcción del dataset final.

Estructura esperada (la del zip oficial de IEEE DataPort / Kaggle; los
nombres exactos de las carpetas superiores pueden variar, por eso se
busca de forma recursiva):

    <raíz>/.../Normal traffic/<Sensor>/<Sensor>.pcap
    <raíz>/.../Attack traffic/.../<Ataque>_attack.pcap
"""
from __future__ import annotations

import ipaddress
from pathlib import Path

import numpy as np
import pandas as pd

from .flows import FEATURES, extract_flows

# Categorías de ataque según Ferrag et al. (2022), Edge-IIoTset
ATTACK_CATEGORY = {
    "DDoS_HTTP_Flood": "DDoS",
    "DDoS_ICMP_Flood": "DDoS",
    "DDoS_TCP_SYN_Flood": "DDoS",
    "DDoS_UDP_Flood": "DDoS",
    "Port_Scanning": "Reconocimiento",
    "OS_Fingerprinting": "Reconocimiento",
    "Vulnerability_scanner": "Reconocimiento",
    "MITM": "MITM",
    "SQL_injection": "Inyeccion",
    "XSS": "Inyeccion",
    "Uploading": "Inyeccion",
    "Backdoor": "Malware",
    "Password": "Malware",
    "Ransomware": "Malware",
}


def _attack_name(stem: str) -> str:
    """Normaliza el nombre del archivo de ataque. Acepta las variantes del
    zip oficial, p. ej. 'DDoS HTTP Flood Attacks', 'Backdoor_attack',
    'MITM (ARP spoofing + DNS) Attack' -> 'DDoS_HTTP_Flood', 'Backdoor', 'MITM'."""
    import re
    s = re.sub(r"\(.*?\)", " ", stem.lower())
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in ("attack", "attacks")]
    norm = "_".join(toks)
    for k in ATTACK_CATEGORY:
        if k.lower() == norm:
            return k
    return norm or stem


def discover_pcaps(root) -> pd.DataFrame:
    """Lista todos los .pcap/.pcapng bajo `root` y los clasifica en normal/ataque."""
    root = Path(root)
    rows = []
    for p in sorted(list(root.rglob("*.pcap")) + list(root.rglob("*.pcapng"))):
        parts = [x.lower() for x in p.parts]
        if any("normal" in x for x in parts):
            kind, name = "normal", p.parent.name
        elif any("attack" in x for x in parts) or "attack" in p.stem.lower():
            kind, name = "attack", _attack_name(p.stem)
        else:
            kind, name = "desconocido", p.stem
        rows.append({"kind": kind, "name": name, "size_MB": round(p.stat().st_size / 1e6, 1),
                     "path": str(p)})
    return pd.DataFrame(rows, columns=["kind", "name", "size_MB", "path"])


def extract_all(inventory: pd.DataFrame, cache_dir, max_packets=None, max_flows=None,
                idle_timeout=60.0, active_timeout=120.0, overwrite=False) -> dict:
    """Extrae flujos de cada PCAP y los guarda en caché (parquet).

    Si se interrumpe, al volver a ejecutar continúa donde se quedó.
    Devuelve {(kind, name): DataFrame}.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, r in enumerate(inventory.itertuples(), 1):
        if r.kind == "desconocido":
            continue
        f = cache_dir / f"{r.kind}__{r.name}.parquet"
        if f.exists() and not overwrite:
            df = pd.read_parquet(f)
            print(f"[{i}/{len(inventory)}] {r.name}: {len(df):,} flujos (desde caché)")
        else:
            print(f"[{i}/{len(inventory)}] Extrayendo {r.name} ({r.size_MB} MB)...")
            df = extract_flows(r.path, idle_timeout=idle_timeout, active_timeout=active_timeout,
                               max_packets=max_packets, max_flows=max_flows)
            df.to_parquet(f, index=False)
        out[(r.kind, r.name)] = df
    return out


def _is_special(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return a.is_multicast or a.is_unspecified or a.is_loopback or a.is_link_local or \
        ip == "255.255.255.255" or ip.endswith(".255")


def label_attack_flows(flows: dict, mode: str = "ip_no_vista", attacker_ips=None):
    """Etiqueta los flujos de las capturas de ataque.

    Las capturas de ataque también contienen tráfico de fondo legítimo
    (sensores publicando al broker, DNS, etc.). Etiquetar todo el archivo
    como ataque introduciría ruido en las etiquetas. Modos:

    * "ip_no_vista" (por defecto): un flujo es de ataque si alguno de sus
      extremos es una IP que NUNCA aparece en las capturas de tráfico
      normal (el atacante y, en DDoS, las IP falsificadas). Los flujos
      entre IP conocidas se descartan como tráfico de fondo.
    * "manual": `attacker_ips` es un set de IP o un dict {ataque: set};
      un flujo es de ataque si involucra alguna de esas IP.
    * "archivo": todos los flujos del archivo son del ataque.

    Devuelve (dict {ataque: DataFrame de flujos de ataque}, tabla diagnóstico).
    """
    benign_ips = set()
    for (kind, _), df in flows.items():
        if kind == "normal":
            benign_ips.update(df["src_ip"].unique())
            benign_ips.update(df["dst_ip"].unique())
    benign_ips = {ip for ip in benign_ips}

    labeled, diag = {}, []
    for (kind, name), df in flows.items():
        if kind != "attack":
            continue
        if mode == "archivo":
            mask = np.ones(len(df), dtype=bool)
        elif mode == "manual":
            ips = attacker_ips.get(name, set()) if isinstance(attacker_ips, dict) else set(attacker_ips)
            mask = (df["src_ip"].isin(ips) | df["dst_ip"].isin(ips)).to_numpy()
        elif mode == "ip_no_vista":
            src_new = ~df["src_ip"].isin(benign_ips) & ~df["src_ip"].map(_is_special)
            dst_new = ~df["dst_ip"].isin(benign_ips) & ~df["dst_ip"].map(_is_special)
            mask = (src_new | dst_new).to_numpy()
        else:
            raise ValueError(f"modo desconocido: {mode}")
        att = df[mask]
        top = att["src_ip"].value_counts().head(3)
        diag.append({
            "ataque": name, "flujos_archivo": len(df), "flujos_ataque": int(mask.sum()),
            "flujos_descartados": int((~mask).sum()),
            "pct_ataque": round(100 * mask.mean(), 1) if len(df) else 0.0,
            "ips_iniciadoras_top": ", ".join(f"{ip} ({c})" for ip, c in top.items()),
        })
        labeled[name] = att
    return labeled, pd.DataFrame(diag)


def build_dataset(flows: dict, attack_flows: dict, devices=None, max_per_class: int = 15000,
                  drop_duplicates: bool = True, min_per_class: int = 50, min_flows: int = 100,
                  seed: int = 42):
    """Une flujos normales (por dispositivo) y de ataque en un solo DataFrame.

    * Normal: etiqueta "Normal", columna `device` = sensor de origen.
    * Ataque: etiqueta = nombre del ataque, `category` = categoría.
    * Elimina vectores de características duplicados dentro de cada clase
      (evita que filas idénticas caigan a la vez en entrenamiento y prueba).
      Excepción: si una clase queda con menos de `min_per_class` vectores
      únicos (p. ej. inundaciones de paquetes idénticos), se conservan
      flujos originales hasta `min_per_class` para poder estratificar; se
      marca en el reporte como "baja variabilidad".
    * Excluye las clases con menos de `min_flows` flujos etiquetados (p. ej.
      MITM por ARP spoofing, que ocurre en capa 2 y casi no deja flujos IP).
    * Submuestrea cada clase/dispositivo a `max_per_class` flujos.
    """
    rng = np.random.RandomState(seed)
    parts, report = [], []

    def _take(df, label, category, device):
        n0 = len(df)
        if n0 < min_flows:
            report.append({"clase": label, "dispositivo": device or "-", "flujos": n0, "sin_duplicados": 0,
                           "usados": 0, "nota": f"excluida: menos de {min_flows} flujos"})
            return
        d = df.drop_duplicates(subset=FEATURES) if drop_duplicates else df
        n1 = len(d)
        low_var = n1 < min_per_class <= n0
        if low_var:
            extra = df.drop(index=d.index)
            k = min(min_per_class - n1, len(extra))
            d = pd.concat([d, extra.iloc[rng.choice(len(extra), k, replace=False)]])
        if len(d) > max_per_class:
            d = d.iloc[rng.choice(len(d), max_per_class, replace=False)]
        d = d.copy()
        d["label"], d["category"], d["device"] = label, category, device
        report.append({"clase": label, "dispositivo": device or "-", "flujos": n0,
                       "sin_duplicados": n1, "usados": len(d),
                       "nota": "baja variabilidad: se conservan duplicados" if low_var else ""})
        parts.append(d)

    for (kind, name), df in flows.items():
        if kind == "normal" and (devices is None or name in devices):
            _take(df, "Normal", "Normal", name)
    for name, df in attack_flows.items():
        if len(df):
            _take(df, name, ATTACK_CATEGORY.get(name, "Otro"), None)

    data = pd.concat(parts, ignore_index=True)
    return data, pd.DataFrame(report)
