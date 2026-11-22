"""Build compact, leakage-free, de-duplicated flow-feature samples for the CIC datasets.

Output: data_cache/<name>.parquet with the 76 CICFlowMeter flow-statistic features that are
shared by CIC-IDS2017 (MachineLearningCSV) and CIC-DDoS2019, plus `label` (original string)
and `y` (0 = BENIGN, 1 = attack).

Design decisions:
  * Identifier / socket columns (Flow ID, IPs, ports, Timestamp, row index) are dropped: they
    leak the attacker identity and do not transfer across networks.
  * `Protocol` and `Inbound` exist only in CIC-DDoS2019 and are dropped so that both datasets
    share an identical input space (required for cross-dataset ticket transfer, RQ2).
  * `Fwd Header Length.1` is an exact duplicate column in CICFlowMeter output and is dropped.
  * Sampling (seed 0, nominal target `--cap` = 30,000, Bernoulli thinning, so counts are approximate):
      CIC-DDoS2019: all BENIGN rows are kept; attack rows are thinned with one probability per file.
      CIC-IDS2017:  BENIGN and attack rows are thinned separately within each file.
  * +-inf -> NaN -> row dropped; rows duplicated in the 76 features AND the original label are dropped
    after concatenation; feature vectors that occur with different labels are kept (counted in the meta file).
  * The CIC-DDoS2019 input is the Parquet conversion made by src/convert_ddos2019_to_parquet.py.

Usage:
  python src/prepare_data.py ddos2019 --day 1 --ddos_root <dir with train_parquet/ and test_parquet/>   # 01-12
  python src/prepare_data.py ddos2019 --day 2 --ddos_root <same dir>                                     # 03-11
  python src/prepare_data.py ids2017 --root <dir with the 8 MachineLearningCVE CSV files>
The CIC-DDoS2019 root can also be given with the environment variable CIC_DDOS2019_PARQUET.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data_cache")
DDOS_PARQUET = os.environ.get("CIC_DDOS2019_PARQUET", os.path.join(ROOT, "raw_data", "CIC-DDoS2019"))

# 76 flow-statistic features common to CIC-IDS2017 MachineLearningCSV and CIC-DDoS2019
# (canonical names = original names with surrounding whitespace stripped).
FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
assert len(FEATURES) == 76


def _finalize(frames: list[pd.DataFrame], name: str, meta: dict) -> None:
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df[FEATURES] = df[FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    n_nan = n0 - len(df)
    df[FEATURES] = df[FEATURES].astype(np.float32)
    n1 = len(df)
    df = df.drop_duplicates(subset=FEATURES + ["label"]).reset_index(drop=True)
    n_dup = n1 - len(df)
    # Feature vectors that occur with BOTH labels are irreducibly ambiguous; keep them (reported).
    amb = df.duplicated(subset=FEATURES, keep=False).sum()
    df["y"] = (df["label"] != "BENIGN").astype(np.int8)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    meta.update(rows_after_sampling=n0, dropped_nan_inf=n_nan, dropped_duplicates=n_dup,
                label_ambiguous_rows=int(amb), final_rows=len(df),
                class_counts=df["label"].value_counts().to_dict(),
                binary_counts=df["y"].value_counts().to_dict())
    with open(os.path.join(OUT_DIR, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps(meta, indent=1, default=str))


def _canon_label(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(s.str.upper() != "BENIGN", "BENIGN")


def prepare_ddos2019(day: int, attack_cap: int, seed: int) -> None:
    sub = "train_parquet" if day == 1 else "test_parquet"
    files = sorted(glob.glob(os.path.join(DDOS_PARQUET, sub, "*.parquet")))
    rng = np.random.default_rng(seed)
    frames, per_file = [], {}
    t0 = time.time()
    for f in files:
        pf = pq.ParquetFile(f)
        rename = {c: c.strip() for c in pf.schema_arrow.names}
        cols = [c for c in pf.schema_arrow.names if rename[c] in FEATURES + ["Label"]]
        n_rows = pf.metadata.num_rows
        p_keep = min(1.0, attack_cap / max(1, n_rows))
        kept_b = kept_a = 0
        for batch in pf.iter_batches(batch_size=500_000, columns=cols):
            df = batch.to_pandas().rename(columns=rename)
            lab = _canon_label(df["Label"])
            ben = (lab == "BENIGN").to_numpy()
            keep = ben | (rng.random(len(df)) < p_keep)
            part = df.loc[keep, FEATURES].copy()
            part["label"] = lab[keep].to_numpy()
            frames.append(part)
            kept_b += int(ben.sum())
            kept_a += int((keep & ~ben).sum())
        per_file[os.path.basename(f)] = dict(rows=n_rows, benign_kept=kept_b, attack_kept=kept_a)
        print(f"{os.path.basename(f)}: rows={n_rows} benign={kept_b} attack_kept={kept_a} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    meta = dict(dataset="CIC-DDoS2019", day=("01-12" if day == 1 else "03-11"), source=sub,
                attack_cap_per_file=attack_cap, seed=seed, per_file=per_file)
    _finalize(frames, f"ddos2019_day{day}", meta)


def prepare_ids2017(root: str, attack_cap: int, seed: int) -> None:
    """CIC-IDS2017 MachineLearningCVE: 8 CSVs (one per capture session)."""
    files = sorted(glob.glob(os.path.join(root, "*.csv")))
    assert files, f"no CSV files in {root}"
    rng = np.random.default_rng(seed)
    frames, per_file = [], {}
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        missing = [c for c in FEATURES if c not in df.columns]
        assert not missing, f"{f}: missing {missing}"
        lab = _canon_label(df["Label"])
        ben = (lab == "BENIGN").to_numpy()
        n_att = int((~ben).sum())
        # IDS2017 benign traffic is abundant (~2.3 M rows) -> cap benign too, per file.
        p_att = min(1.0, attack_cap / max(1, n_att))
        p_ben = min(1.0, attack_cap / max(1, int(ben.sum())))
        u = rng.random(len(df))
        keep = np.where(ben, u < p_ben, u < p_att)
        part = df.loc[keep, FEATURES].copy()
        part["label"] = lab[keep].to_numpy()
        frames.append(part)
        per_file[os.path.basename(f)] = dict(rows=len(df), benign=int(ben.sum()), attack=n_att,
                                             kept=int(keep.sum()))
        print(os.path.basename(f), per_file[os.path.basename(f)], flush=True)
    meta = dict(dataset="CIC-IDS2017 MachineLearningCVE", root=root, cap_per_file_per_class=attack_cap,
                seed=seed, per_file=per_file)
    _finalize(frames, "ids2017", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ddos2019", "ids2017"])
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--root", default="")
    ap.add_argument("--cap", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ddos_root", default="", help="folder with train_parquet/ and test_parquet/ (CIC-DDoS2019)")
    a = ap.parse_args()
    if a.ddos_root:
        DDOS_PARQUET = a.ddos_root
    if a.dataset == "ddos2019":
        prepare_ddos2019(a.day, a.cap, a.seed)
    else:
        prepare_ids2017(a.root, a.cap, a.seed)
