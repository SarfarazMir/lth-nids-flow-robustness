"""Validation of lthlib/flowattack.py: (1) identity check, (2) ESR vs budget on a dense MLP."""
import json, os, torch, numpy as np
from lthlib import prune as P
from lthlib.common import DEVICE, RESULTS, Bundle, Domain, make_pre
from lthlib.flowattack import FlowAttack, flow_eval
from lthlib.models import build_model
from lthlib.train import set_seed, train
out = {}
for key in ("ids17", "ddos19_d1"):
    dom = Domain(key); pre = make_pre(dom); B = Bundle(dom, pre)
    fa = FlowAttack(dom.features, pre, DEVICE)
    te = dom.splits["test"]
    Xr = torch.tensor(te.X, dtype=torch.float64, device=DEVICE)
    z = torch.zeros(len(Xr), dtype=torch.float64, device=DEVICE)
    Y = fa.apply(Xr, z, z, z)
    rel = ((Y - Xr).abs() / (Xr.abs() + 1e-6))
    bad = (rel > 1e-3)
    per_feat = {dom.features[j]: float(bad[:, j].float().mean()) for j in range(76) if bad[:, j].any()}
    Zin = fa.to_input(Y); Zref = torch.from_numpy(pre.transform(te.X)).to(DEVICE)
    out[key] = {"identity_rows_exact": float((~bad.any(1)).float().mean()),
                "identity_mismatch_by_feature": per_feat,
                "max_input_space_dev": float((Zin - Zref).abs().max())}
    set_seed(0); net = build_model("mlp", B.Xtr.shape[1]).to(DEVICE)
    net, _ = train(net, P.full_masks(net), B.Xtr, B.ytr, epochs=10, cw=B.cw, seed=0)
    att = torch.tensor(te.X[te.y == 1][:3000], dtype=torch.float64, device=DEVICE)
    out[key]["flow_attack_dense"] = flow_eval(net, fa, att)
    print(key, json.dumps(out[key], indent=1))
json.dump(out, open(os.path.join(RESULTS, "raw", "flowattack_validation.json"), "w"), indent=1)
