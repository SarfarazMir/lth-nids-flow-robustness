import time, torch
from lthlib import prune as P
from lthlib.common import DEVICE, Bundle, Domain, make_pre, attack_subset, FLOW_RHOS
from lthlib.flowattack import flow_eval
from lthlib.models import build_model
from lthlib.train import set_seed, train, evaluate
dom = Domain("ids17"); pre = make_pre(dom); B = Bundle(dom, pre)
_, Xr = attack_subset(B.Xte, B.yte, 3000, X_raw=B.Xte_raw)
adv = dict(kind="flow", fa=B.flow, X_raw=B.Xtr_raw, rho=1.0, steps=5, n_random=32)
set_seed(0); net = build_model("mlp", 76).to(DEVICE)
t=time.time(); net,_ = train(net, P.full_masks(net), B.Xtr, B.ytr, epochs=10, cw=B.cw, adv=adv, bs=4096, lr=2e-3); print("flowAT 10ep bs4096", time.time()-t)
print(evaluate(net, B.Xte, B.yte)); print(flow_eval(net, B.flow, Xr, FLOW_RHOS))
