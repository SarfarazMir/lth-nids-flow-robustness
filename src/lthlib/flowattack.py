"""Flow-consistent injection attack (traffic-space evasion for CICFlowMeter features).

Attacker model (sender side of a malicious flow): the attacker may
  * inject k >= 0 extra FORWARD packets, each carrying s bytes of payload (0 <= s <= 1460, MSS),
  * stretch the flow by an extra delay dd >= 0 (microseconds),
within a relative budget rho:  k <= max(1, ceil(rho * nf)),  dd <= rho * (d + 1e5).
It cannot remove or shrink packets (the attack must still work) and cannot change backward
(victim/response) traffic, TCP window fields or flag semantics.

Every feature that is a deterministic function of (nf, lf, d, packet sizes) is RECOMPUTED with
the CICFlowMeter relations verified on our caches (literature/05_protocol_notes.md, Sect. 2.6,
relations E1-E16): totals, means, Avg Segment Size, Subflow counts/bytes, Max/Min packet length,
pooled std/variance, Average Packet Size (E11 quirk), per-second rates, Down/Up ratio, header
length, act_data_pkt_fwd, Flow/Fwd IAT mean and totals. Features whose new value depends on the
unknown packet timing order (IAT std/max/min, Active/Idle stats) and flag counts are held at their
original values; this is a stated modelling assumption, not a guarantee of problem-space
realizability.

Search: (a) random search over (k, s, dd) and (b) PGD on a sigmoid relaxation of (k, s, dd)
with a final integer rounding; the attack succeeds if ANY candidate is classified BENIGN.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MSS = 1460.0


class FlowAttack:
    def __init__(self, features, pre, device):
        self.f = {n: i for i, n in enumerate(features)}
        self.mu = torch.tensor(pre.mu, dtype=torch.float32, device=device)
        self.sd = torch.tensor(pre.sd, dtype=torch.float32, device=device)
        self.device = device

    # ------------------------------------------------------------------ feature recomputation
    def to_input(self, X):
        L = torch.sign(X) * torch.log1p(X.abs())
        return ((L - self.mu) / self.sd).float()

    def apply(self, X, k, s, dd):
        """X: raw features (N,76) float64; k, s, dd: (N,) float64 (k may be fractional in PGD)."""
        f = self.f
        g = lambda n: X[:, f[n]]
        nf, nb = g("Total Fwd Packets"), g("Total Backward Packets")
        lf, lb = g("Total Length of Fwd Packets"), g("Total Length of Bwd Packets")
        d = g("Flow Duration")
        n = nf + nb
        nf2 = nf + k
        lf2 = lf + k * s
        d2 = d + dd
        n2 = nf2 + nb
        Y = X.clone()

        def setf(name, val):
            Y[:, f[name]] = val

        fmean = g("Fwd Packet Length Mean")
        fstd = g("Fwd Packet Length Std")
        fmean2 = lf2 / nf2.clamp_min(1)
        # pooled sample variance of {old fwd packets} U {k packets of size s}
        var_f = ((nf - 1).clamp_min(0) * fstd ** 2 + nf * (fmean - fmean2) ** 2 + k * (s - fmean2) ** 2) \
            / (nf2 - 1).clamp_min(1)
        setf("Total Fwd Packets", nf2)
        setf("Total Length of Fwd Packets", lf2)
        setf("Fwd Packet Length Mean", fmean2)
        setf("Avg Fwd Segment Size", fmean2)
        setf("Fwd Packet Length Std", var_f.clamp_min(0).sqrt())
        has = (k > 0).float()
        setf("Fwd Packet Length Max", torch.maximum(g("Fwd Packet Length Max"), s * has))
        setf("Fwd Packet Length Min", torch.where(k > 0, torch.minimum(g("Fwd Packet Length Min"), s),
                                                  g("Fwd Packet Length Min")))
        setf("Max Packet Length", torch.maximum(g("Max Packet Length"), s * has))
        setf("Min Packet Length", torch.where(k > 0, torch.minimum(g("Min Packet Length"), s),
                                              g("Min Packet Length")))
        setf("Subflow Fwd Packets", nf2)
        setf("Subflow Fwd Bytes", lf2)
        # overall packet-length stats use the (n+1)-sample convention observed in the data (E11)
        pm = g("Packet Length Mean")
        pstd = g("Packet Length Std")
        m1 = n + 1
        pm2 = (pm * m1 + k * s) / (m1 + k)
        var_p = ((m1 - 1) * pstd ** 2 + m1 * (pm - pm2) ** 2 + k * (s - pm2) ** 2) / (m1 + k - 1).clamp_min(1)
        setf("Packet Length Mean", pm2)
        setf("Packet Length Std", var_p.clamp_min(0).sqrt())
        setf("Packet Length Variance", var_p.clamp_min(0))
        setf("Average Packet Size", pm2 * (n2 + 1) / n2.clamp_min(1))
        # rates (d in microseconds); flows with d == 0 keep their original rate unless delayed
        sec = d2 / 1e6
        pos = d2 > 0
        setf("Flow Bytes/s", torch.where(pos, (lf2 + lb) / sec.clamp_min(1e-12), g("Flow Bytes/s")))
        setf("Flow Packets/s", torch.where(pos, n2 / sec.clamp_min(1e-12), g("Flow Packets/s")))
        setf("Fwd Packets/s", torch.where(pos, nf2 / sec.clamp_min(1e-12), g("Fwd Packets/s")))
        setf("Bwd Packets/s", torch.where(pos, nb / sec.clamp_min(1e-12), g("Bwd Packets/s")))
        setf("Flow Duration", d2)
        setf("Down/Up Ratio", torch.floor(nb / nf2.clamp_min(1)))
        setf("Flow IAT Mean", torch.where(n2 > 1, d2 / (n2 - 1).clamp_min(1), g("Flow IAT Mean")))
        fiat_tot = g("Fwd IAT Total") + dd
        setf("Fwd IAT Total", fiat_tot)
        setf("Fwd IAT Mean", torch.where(nf2 > 1, fiat_tot / (nf2 - 1).clamp_min(1), g("Fwd IAT Mean")))
        hdr = g("min_seg_size_forward").clamp_min(0)
        setf("Fwd Header Length", g("Fwd Header Length") + k * hdr)
        setf("act_data_pkt_fwd", g("act_data_pkt_fwd") + k * (s >= 1).float())
        # unperturbed flows pass through unchanged (avoids re-deriving legacy anomalies)
        changed = ((k > 0) | (dd > 0))[:, None]
        return torch.where(changed, Y, X)

    # ------------------------------------------------------------------ budget
    def limits(self, X, rho):
        nf = X[:, self.f["Total Fwd Packets"]]
        d = X[:, self.f["Flow Duration"]].clamp_min(0)
        kmax = torch.clamp(torch.ceil(rho * nf), min=1.0)
        ddmax = rho * (d + 1e5)
        return kmax, ddmax

    # ------------------------------------------------------------------ search
    @torch.no_grad()
    def _benign(self, model, X, k, s, dd):
        return model(self.to_input(self.apply(X, k, s, dd))).argmax(1) == 0

    def attack(self, model, X, rho, n_random=128, pgd_steps=30, seed=0):
        """Returns a bool tensor: evasion found for each flow in X (raw, float64)."""
        model.eval()
        N = len(X)
        kmax, ddmax = self.limits(X, rho)
        found = torch.zeros(N, dtype=torch.bool, device=X.device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        # (a) corner + random search, vectorized over candidates (chunks of `chunk` candidates)
        cands = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 1.0),
                 (1.0, 0.5, 1.0), (0.5, 0.5, 0.5)]
        U = torch.cat([torch.tensor(cands, dtype=torch.float32), torch.rand(n_random, 3, generator=g)]).to(X.device)
        chunk = max(1, 65536 // max(1, N))
        for i in range(0, len(U), chunk):
            u = U[i:i + chunk]                      # (c, 3)
            c = len(u)
            Xr = X.repeat(c, 1)                     # (c*N, 76), candidate-major
            km, dm = kmax.repeat(c), ddmax.repeat(c)
            uu = u.repeat_interleave(N, 0)
            m = self._benign(model, Xr, torch.round(uu[:, 0] * km), uu[:, 1] * MSS, uu[:, 2] * dm)
            found |= m.view(c, N).any(0)
        # (b) PGD on a sigmoid relaxation for the rest
        rest = (~found).nonzero().flatten()
        if len(rest) and pgd_steps > 0:
            Xr, km, dm = X[rest], kmax[rest], ddmax[rest]
            for init in (0.0, 1.5):  # two initial states
                theta = torch.full((len(rest), 3), init, dtype=torch.float32, device=X.device, requires_grad=True)
                opt = torch.optim.Adam([theta], lr=0.2)
                for _ in range(pgd_steps):
                    u = torch.sigmoid(theta)
                    x = self.to_input(self.apply(Xr, u[:, 0] * km, u[:, 1] * MSS, u[:, 2] * dm))
                    logits = model(x)
                    loss = (logits[:, 0] - logits[:, 1]).sum()  # push towards BENIGN
                    opt.zero_grad()
                    (-loss).backward()
                    opt.step()
                with torch.no_grad():
                    u = torch.sigmoid(theta)
                    for kk in (torch.floor(u[:, 0] * km), torch.ceil(u[:, 0] * km)):
                        found[rest] |= self._benign(model, Xr, kk, u[:, 1] * MSS, u[:, 2] * dm)
        return found


def craft(model, fa: FlowAttack, X, rho, n_random=16, steps=10, seed=0):
    """Adversarial inputs (model-input space) for adversarial training: best-margin candidate among
    random (k, s, dd) draws and a short sigmoid-relaxed PGD; k is rounded to an integer."""
    model.eval()
    kmax, ddmax = fa.limits(X, rho)
    N = len(X)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def margin(u, kk):
        lo = model(fa.to_input(fa.apply(X, kk, u[:, 1] * MSS, u[:, 2] * ddmax)))
        return lo[:, 0] - lo[:, 1]

    with torch.no_grad():
        U = torch.rand(n_random, N, 3, generator=g).to(X.device)
        lo = model(fa.to_input(fa.apply(X.repeat(n_random, 1), torch.round(U[..., 0].reshape(-1) * kmax.repeat(n_random)),
                                         U[..., 1].reshape(-1) * MSS, U[..., 2].reshape(-1) * ddmax.repeat(n_random))))
        M = (lo[:, 0] - lo[:, 1]).view(n_random, N)
        j = M.argmax(0)
        best_m = M.max(0).values
        best_u = U[j, torch.arange(N, device=X.device)]
    theta = torch.logit(best_u.clamp(1e-3, 1 - 1e-3)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.2)
    for _ in range(steps):
        u = torch.sigmoid(theta)
        loss = -margin(u, u[:, 0] * kmax).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        u = torch.sigmoid(theta)
        m = margin(u, torch.round(u[:, 0] * kmax))
        upd = m > best_m
        best_u[upd] = u[upd]
        u = best_u
        return fa.to_input(fa.apply(X, torch.round(u[:, 0] * kmax), u[:, 1] * MSS, u[:, 2] * ddmax))


def flow_eval(model, fa: FlowAttack, X_raw, rhos=(0.1, 0.25, 0.5, 1.0, 2.0), bs=4096, **kw) -> dict:
    """ESR of the flow-consistent attack on malicious flows (raw float64 tensor) per budget rho."""
    model.eval()
    with torch.no_grad():
        det = torch.cat([model(fa.to_input(X_raw[i:i + bs])).argmax(1) == 1 for i in range(0, len(X_raw), bs)])
    out = {"clean_DR": float(det.float().mean())}
    Xd = X_raw[det]
    for rho in rhos:
        hit = torch.cat([fa.attack(model, Xd[i:i + bs], rho, **kw) for i in range(0, len(Xd), bs)]) \
            if len(Xd) else torch.zeros(0, dtype=torch.bool)
        out[f"ESR@{rho}"] = float(hit.float().mean()) if len(Xd) else float("nan")
    e = [0.0] + list(rhos)
    s = [0.0] + [out[f"ESR@{r}"] for r in rhos]
    out["ESR_AUC"] = sum((e[i + 1] - e[i]) * (s[i + 1] + s[i]) / 2 for i in range(len(rhos))) / rhos[-1]
    return out
