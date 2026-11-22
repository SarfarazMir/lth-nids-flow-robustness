"""Per-epoch learning curves and confusion matrices for the paper (no new experimental conditions).

Re-trains, with the exact exp_rq1.py / exp_robust.py recipe, networks whose masks and rewind weights
were saved by those scripts, and records after every epoch: mean training loss, validation loss and
validation balanced accuracy.  The final balanced accuracy is compared with the logged value of the
original run (GPU kernels are not bit-deterministic, so small differences are expected).

  standard  : dense + IMP tickets (rounds 4/7/10/13/16/20), ERK at rounds 13/16; both datasets, 5 seeds
  adversarial: dense_at + A-IMP (rounds 7/13/16), CIC-IDS2017, 5 seeds
Also: confusion matrices (clean test set; and the 3,000 attack flows under the flow attack at rho = 1)
for the dense network and the 97.2 % ticket (CIC-IDS2017, seed 0, saved weights).
Outputs results/raw/curves/*.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os

import torch
import torch.nn.functional as F

from lthlib import prune as P
from lthlib.common import DEVICE, RESULTS, Bundle, Domain, attack_subset, make_pre
from lthlib.flowattack import craft
from lthlib.models import build_model
from lthlib.train import evaluate, set_seed

OUT = os.path.join(RESULTS, "raw", "curves")
os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def val_loss(model, X, y, cw, bs=8192):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), bs):
        tot += F.cross_entropy(model(X[i:i + bs]), y[i:i + bs], weight=cw, reduction="sum").item()
    return tot / len(X)


def train_logged(model, masks, B, *, epochs=10, lr=1e-3, wd=1e-4, bs=1024, seed=0, adv=None):
    """Same loop as lthlib.train.train, plus per-epoch logging."""
    X, y, cw = B.Xtr, B.ytr, B.cw
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(X)
    spe = math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.1)
    P.apply_masks(model, masks)
    hist = []
    it = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(X.device)
        model.train()
        run, cnt = 0.0, 0
        for b in range(spe):
            idx = perm[b * bs:(b + 1) * bs]
            xb, yb = X[idx], y[idx]
            if adv is not None:
                att = (yb == 1).nonzero().flatten()
                if len(att):
                    xa = craft(model, adv["fa"], adv["X_raw"][idx][att], adv["rho"],
                               n_random=adv["n_random"], steps=adv["steps"], seed=it)
                    xb = xb.clone()
                    xb[att] = xa
                model.train()
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            P.apply_masks(model, masks)
            run += loss.item() * len(yb)
            cnt += len(yb)
            it += 1
        m = evaluate(model, B.Xva, B.yva)
        hist.append(dict(epoch=ep + 1, train_loss=run / cnt, val_loss=val_loss(model, B.Xva, B.yva, cw),
                         val_bacc=m["bacc"]))
    return model, hist


def logged_bacc(path, method, rnd):
    d = json.load(open(path))
    for r in d["records"]:
        if r["method"] == method and r["round"] == rnd:
            return r["src_test/bacc"]
    return None


def standard(src, seed):
    dom = Domain(src)
    B = Bundle(dom, make_pre(dom))
    st = torch.load(os.path.join(RESULTS, "raw", "rq1", f"{src}_mlp_s{seed}_state.pt"), map_location=DEVICE)
    log = os.path.join(RESULTS, "raw", "rq1", f"{src}_mlp_s{seed}.json")
    d_in = B.Xtr.shape[1]
    out = []

    def go(name, rnd, net, mk):
        net, h = train_logged(net, mk, B, seed=seed)
        test = evaluate(net, B.Xte, B.yte)["bacc"]
        out.append(dict(method=name, round=rnd, sparsity=P.sparsity(mk), history=h, test_bacc=test,
                        logged_test_bacc=logged_bacc(log, name, rnd)))
        print(f"{src} s{seed} {name:6s} r={rnd:2d} test={test:.4f} logged={out[-1]['logged_test_bacc']}", flush=True)

    set_seed(seed)
    net = build_model("mlp", d_in).to(DEVICE)
    go("dense", 0, net, P.full_masks(net))
    tk = {k: v.to(DEVICE) for k, v in st["theta_k"].items()}
    for rnd in (4, 7, 10, 13, 16, 20):
        mk = {k: v.to(DEVICE) for k, v in st["imp_masks"][rnd].items()}
        set_seed(seed)
        net = build_model("mlp", d_in).to(DEVICE)
        net.load_state_dict(tk)
        go("imp", rnd, net, mk)
    for rnd in (13, 16):
        s = P.sparsity({k: v for k, v in st["imp_masks"][rnd].items()})
        set_seed(seed)
        net = build_model("mlp", d_in).to(DEVICE)
        net.load_state_dict({k: v.to(DEVICE) for k, v in st["theta0"].items()})
        go("erk", rnd, net, P.erk_random_masks(net, s, seed=seed + rnd))
    json.dump(out, open(os.path.join(OUT, f"std_{src}_s{seed}.json"), "w"))


def adversarial(seed):
    src = "ids17"
    dom = Domain(src)
    B = Bundle(dom, make_pre(dom))
    st = torch.load(os.path.join(RESULTS, "raw", "robust", f"{src}_mlp_s{seed}_flow1.0_state.pt"), map_location=DEVICE)
    log = os.path.join(RESULTS, "raw", "robust", f"{src}_mlp_s{seed}_flow1.0.json")
    adv = dict(fa=B.flow, X_raw=B.Xtr_raw, rho=1.0, steps=5, n_random=32)
    d_in = B.Xtr.shape[1]
    out = []

    def go(name, rnd, net, mk):
        net, h = train_logged(net, mk, B, seed=seed, lr=2e-3, bs=4096, adv=adv)
        test = evaluate(net, B.Xte, B.yte)["bacc"]
        out.append(dict(method=name, round=rnd, sparsity=P.sparsity(mk), history=h, test_bacc=test,
                        logged_test_bacc=logged_bacc(log, name, rnd)))
        print(f"AT s{seed} {name:8s} r={rnd:2d} test={test:.4f} logged={out[-1]['logged_test_bacc']}", flush=True)

    set_seed(seed)
    net = build_model("mlp", d_in).to(DEVICE)
    go("dense_at", 0, net, P.full_masks(net))
    tk = {k: v.to(DEVICE) for k, v in st["theta_k"].items()}
    for rnd in (7, 13, 16):
        mk = {k: v.to(DEVICE) for k, v in st["aimp_masks"][rnd].items()}
        set_seed(seed)
        net = build_model("mlp", d_in).to(DEVICE)
        net.load_state_dict(tk)
        go("aimp", rnd, net, mk)
    json.dump(out, open(os.path.join(OUT, f"adv_{src}_s{seed}.json"), "w"))


@torch.no_grad()
def confusion(pred, y):
    return [[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)]


def confusions():
    src, seed = "ids17", 0
    dom = Domain(src)
    B = Bundle(dom, make_pre(dom))
    st = torch.load(os.path.join(RESULTS, "raw", "rq1", f"{src}_mlp_s{seed}_state.pt"), map_location=DEVICE)
    X_att, X_raw = attack_subset(B.Xte, B.yte, 3000, seed=0, X_raw=B.Xte_raw)
    d_in = B.Xtr.shape[1]
    res = {}
    for name, sd in (("dense", st["dense"]), ("imp_97.2", st["imp_weights"][16])):
        net = build_model("mlp", d_in).to(DEVICE)
        net.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()})
        net.eval()
        pred = torch.cat([net(B.Xte[i:i + 8192]).argmax(1) for i in range(0, len(B.Xte), 8192)])
        clean = confusion(pred, B.yte)
        det = net(X_att).argmax(1) == 1
        hit = B.flow.attack(net, X_raw[det], 1.0)
        res[name] = dict(clean_test=clean, attack_flows=len(X_att), detected_before=int(det.sum()),
                         evaded_rho1=int(hit.sum()))
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(OUT, "confusion_ids17_s0.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["standard", "adversarial", "confusion"], required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    for s in [int(x) for x in a.seeds.split(",")]:
        if a.part == "standard":
            for src in ("ids17", "ddos19_d1"):
                standard(src, s)
        elif a.part == "adversarial":
            adversarial(s)
    if a.part == "confusion":
        confusions()
