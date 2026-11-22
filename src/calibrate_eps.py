"""Calibrate the evasion budget: ESR of dense MLPs vs eps, both threat models, 2 source domains."""
import json, os, torch
from lthlib import prune as P
from lthlib.common import DEVICE, RESULTS, Bundle, Domain, attack_subset, make_pre, threat_models
from lthlib.attacks import evasion_eval
from lthlib.models import build_model
from lthlib.train import set_seed, train, evaluate
EPS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
out = {}
for key in ("ids17", "ddos19_d1"):
    dom = Domain(key); pre = make_pre(dom); B = Bundle(dom, pre); tms = threat_models(dom.features, pre)
    set_seed(0); net = build_model("mlp", B.Xtr.shape[1]).to(DEVICE)
    net, _ = train(net, P.full_masks(net), B.Xtr, B.ytr, epochs=10, cw=B.cw, seed=0)
    Xa = attack_subset(B.Xte, B.yte, 3000)
    r = {"clean": evaluate(net, B.Xte, B.yte)}
    for k, tm in tms.items():
        r[k] = evasion_eval(net, Xa, tm, EPS, steps=50, restarts=2)
        r[k + "_fgsm"] = evasion_eval(net, Xa, tm, EPS, attack="fgsm")
    out[key] = r
    print(key, json.dumps(r, indent=0))
json.dump(out, open(os.path.join(RESULTS, "raw", "calibrate_eps.json"), "w"), indent=1)
