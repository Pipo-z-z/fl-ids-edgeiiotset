"""
Extracción de flujos bidireccionales desde capturas PCAP crudas.

Implementación propia sobre dpkt (Python puro, funciona igual en Windows,
Linux y macOS, sin Npcap ni dependencias en C). Cada flujo se identifica
por la 5-tupla (protocolo, IP/puerto de ambos extremos) sin importar la
dirección. El primer paquete define la dirección "forward" (iniciador).

Un flujo termina por:
  * inactividad  (idle_timeout, por defecto 60 s)
  * duración máx (active_timeout, por defecto 120 s)
  * fin de la captura
El límite de duración segmenta las conexiones persistentes típicas de IoT
(MQTT, Modbus/TCP mantienen la sesión abierta durante horas) en ventanas
de 120 s; sin él, cada sensor produciría muy pocos flujos.

Las estadísticas se acumulan de forma incremental (sumas, sumas de
cuadrados, mínimos y máximos), por lo que la memoria no depende del
número de paquetes de cada flujo.

IMPORTANTE (evitar fuga de información): las columnas de identidad
(IP, puertos, timestamps) se conservan solo como METADATOS para el
etiquetado y el diagnóstico; NO forman parte de FEATURES. Así el modelo
aprende el comportamiento del tráfico y no "quién" lo envía.
"""
from __future__ import annotations

import math
import socket
import time
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

META = ["src_ip", "dst_ip", "src_port", "dst_port", "proto", "first_ts"]

FEATURES = [
    "proto_tcp", "proto_udp", "proto_icmp",
    "duration",
    "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "payload_bytes", "payload_mean",
    "iat_mean", "iat_std", "iat_min", "iat_max",
    "fwd_iat_mean", "bwd_iat_mean",
    "syn_cnt", "ack_cnt", "fin_cnt", "rst_cnt", "psh_cnt", "urg_cnt",
    "bytes_per_s", "pkts_per_s", "down_up_ratio",
]

# Índices del vector de estado de cada flujo (lista mutable por eficiencia)
(_FIRST, _LAST, _NF, _NB, _BF, _BB, _SMIN, _SMAX, _SSUM, _SSQ, _PAY,
 _ISUM, _ISQ, _IMIN, _IMAX, _ICNT, _LASTF, _IFSUM, _IFCNT, _LASTB, _IBSUM,
 _IBCNT, _SYN, _ACK, _FIN, _RST, _PSH, _URG) = range(28)
_STATE_LEN = 28

_TH_FIN, _TH_SYN, _TH_RST, _TH_PUSH, _TH_ACK, _TH_URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


def _ip_str(raw: bytes) -> str:
    if len(raw) == 4:
        return socket.inet_ntoa(raw)
    return socket.inet_ntop(socket.AF_INET6, raw)


def _open_reader(fh):
    magic = fh.read(4)
    fh.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(fh)
    return dpkt.pcap.Reader(fh)


def _l3_from_frame(buf: bytes, datalink: int):
    """Devuelve el objeto IP/IP6 de dpkt o None si el paquete no es IP."""
    try:
        if datalink == dpkt.pcap.DLT_EN10MB:
            l3 = dpkt.ethernet.Ethernet(buf).data
        elif datalink == dpkt.pcap.DLT_LINUX_SLL:
            l3 = dpkt.sll.SLL(buf).data
        elif datalink in (dpkt.pcap.DLT_RAW, 12, 14):
            l3 = dpkt.ip.IP(buf) if (buf[0] >> 4) == 4 else dpkt.ip6.IP6(buf)
        elif datalink == dpkt.pcap.DLT_NULL:
            l3 = dpkt.loopback.Loopback(buf).data
        else:
            return None
    except (dpkt.UnpackError, IndexError, ValueError):
        return None
    if isinstance(l3, (dpkt.ip.IP, dpkt.ip6.IP6)):
        return l3
    return None


def _finalize(key, meta, st) -> tuple:
    proto = meta[4]
    n = st[_NF] + st[_NB]
    dur = st[_LAST] - st[_FIRST]
    size_mean = st[_SSUM] / n
    size_var = max(st[_SSQ] / n - size_mean * size_mean, 0.0)
    if st[_ICNT] > 0:
        iat_mean = st[_ISUM] / st[_ICNT]
        iat_std = math.sqrt(max(st[_ISQ] / st[_ICNT] - iat_mean * iat_mean, 0.0))
        iat_min, iat_max = st[_IMIN], st[_IMAX]
    else:
        iat_mean = iat_std = iat_min = iat_max = 0.0
    tot_bytes = st[_BF] + st[_BB]
    feats = (
        1.0 if proto == 6 else 0.0,
        1.0 if proto == 17 else 0.0,
        1.0 if proto in (1, 58) else 0.0,
        dur,
        st[_NF], st[_NB], st[_BF], st[_BB],
        st[_SMIN], st[_SMAX], size_mean, math.sqrt(size_var),
        st[_BF] / st[_NF] if st[_NF] else 0.0,
        st[_BB] / st[_NB] if st[_NB] else 0.0,
        st[_PAY], st[_PAY] / n,
        iat_mean, iat_std, iat_min, iat_max,
        st[_IFSUM] / st[_IFCNT] if st[_IFCNT] else 0.0,
        st[_IBSUM] / st[_IBCNT] if st[_IBCNT] else 0.0,
        st[_SYN], st[_ACK], st[_FIN], st[_RST], st[_PSH], st[_URG],
        tot_bytes / dur if dur > 0 else 0.0,
        n / dur if dur > 0 else 0.0,
        st[_NB] / st[_NF] if st[_NF] else 0.0,
    )
    return tuple(meta) + feats


def extract_flows(pcap_path, idle_timeout: float = 60.0, active_timeout: float = 120.0,
                  max_packets: int | None = None, max_flows: int | None = None,
                  sweep_every: int = 200_000, verbose: bool = True) -> pd.DataFrame:
    """Convierte un archivo .pcap/.pcapng en un DataFrame de flujos.

    max_packets / max_flows: límites opcionales para capturas muy grandes
    (p. ej. inundaciones DDoS). Se documentan como submuestreo en la
    metodología.
    """
    pcap_path = Path(pcap_path)
    active: dict = {}
    out: list = []
    n_pkts = n_ip = 0
    t0 = time.time()
    last_ts = 0.0

    def flush_expired(now):
        expired = [k for k, (m, s) in active.items() if now - s[_LAST] > idle_timeout]
        for k in expired:
            m, s = active.pop(k)
            out.append(_finalize(k, m, s))

    with open(pcap_path, "rb") as fh:
        reader = _open_reader(fh)
        datalink = reader.datalink()
        it = iter(reader)
        while True:
            try:
                ts, buf = next(it)
            except StopIteration:
                break
            except (dpkt.NeedData, dpkt.UnpackError, ValueError):
                break  # captura truncada al final: se detiene sin error
            n_pkts += 1
            if max_packets and n_pkts > max_packets:
                break
            if max_flows and len(out) >= max_flows:
                break
            ip = _l3_from_frame(buf, datalink)
            if ip is None:
                continue
            n_ip += 1
            last_ts = ts
            if isinstance(ip, dpkt.ip.IP):
                proto, size = ip.p, ip.len
            else:
                proto, size = ip.nxt, ip.plen + 40
            l4 = ip.data
            sport = dport = 0
            flags = 0
            payload = 0
            if proto == 6 and isinstance(l4, dpkt.tcp.TCP):
                sport, dport, flags = l4.sport, l4.dport, l4.flags
                payload = len(l4.data)
            elif proto == 17 and isinstance(l4, dpkt.udp.UDP):
                sport, dport = l4.sport, l4.dport
                payload = len(l4.data)
            elif isinstance(l4, (bytes, bytearray)):
                payload = len(l4)
            else:
                try:
                    payload = len(l4.data)
                except AttributeError:
                    payload = 0
            a = (ip.src, sport)
            b = (ip.dst, dport)
            key = (proto, a, b) if a <= b else (proto, b, a)

            entry = active.get(key)
            if entry is not None:
                meta, st = entry
                if ts - st[_LAST] > idle_timeout or ts - st[_FIRST] > active_timeout:
                    out.append(_finalize(key, meta, st))
                    entry = None
            if entry is None:
                meta = [_ip_str(ip.src), _ip_str(ip.dst), sport, dport, proto, ts, ip.src, sport]
                st = [0.0] * _STATE_LEN
                st[_FIRST] = st[_LAST] = ts
                st[_SMIN] = float("inf")
                st[_IMIN] = float("inf")
                st[_LASTF] = st[_LASTB] = -1.0
                active[key] = (meta, st)
                first = True
            else:
                meta, st = entry
                first = False

            fwd = ip.src == meta[6] and sport == meta[7]
            if not first:
                iat = ts - st[_LAST]
                st[_ISUM] += iat
                st[_ISQ] += iat * iat
                st[_ICNT] += 1
                if iat < st[_IMIN]:
                    st[_IMIN] = iat
                if iat > st[_IMAX]:
                    st[_IMAX] = iat
            st[_LAST] = ts
            if fwd:
                st[_NF] += 1
                st[_BF] += size
                if st[_LASTF] >= 0:
                    st[_IFSUM] += ts - st[_LASTF]
                    st[_IFCNT] += 1
                st[_LASTF] = ts
            else:
                st[_NB] += 1
                st[_BB] += size
                if st[_LASTB] >= 0:
                    st[_IBSUM] += ts - st[_LASTB]
                    st[_IBCNT] += 1
                st[_LASTB] = ts
            if size < st[_SMIN]:
                st[_SMIN] = size
            if size > st[_SMAX]:
                st[_SMAX] = size
            st[_SSUM] += size
            st[_SSQ] += size * size
            st[_PAY] += payload
            if flags:
                if flags & _TH_SYN: st[_SYN] += 1
                if flags & _TH_ACK: st[_ACK] += 1
                if flags & _TH_FIN: st[_FIN] += 1
                if flags & _TH_RST: st[_RST] += 1
                if flags & _TH_PUSH: st[_PSH] += 1
                if flags & _TH_URG: st[_URG] += 1

            if n_pkts % sweep_every == 0:
                flush_expired(ts)
                if verbose:
                    print(f"    {pcap_path.name}: {n_pkts:,} paquetes, "
                          f"{len(out) + len(active):,} flujos ({time.time() - t0:.0f} s)")

    for key, (meta, st) in active.items():
        out.append(_finalize(key, meta, st))
    if max_flows:
        out = out[:max_flows]

    cols = META + ["_raw_src", "_raw_sport"] + FEATURES
    df = pd.DataFrame(out, columns=cols).drop(columns=["_raw_src", "_raw_sport"])
    df[FEATURES] = df[FEATURES].astype(np.float32)
    if verbose:
        print(f"  {pcap_path.name}: {n_pkts:,} paquetes leídos ({n_ip:,} IP) -> "
              f"{len(df):,} flujos en {time.time() - t0:.1f} s")
    return df
