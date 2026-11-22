import time, torch
from lthlib import prune as P
from lthlib.common import DEVICE, Bundle, Domain, make_pre, attack_subset, FLOW_RHOS
from lthlib.flowattack import flow_eval, craft
from lthlib.models import build_model
from lthlib.train import set_seed, train
dom = Domain("ids17"); pre = make_pre(dom); B = Bundle(dom, pre)
set_seed(0); net = build_model("mlp", 76).to(DEVICE)
t=time.time(); net,_ = train(net, P.full_masks(net), B.Xtr, B.ytr, epochs=2, cw=B.cw); print("train 2ep", time.time()-t)
_, Xr = attack_subset(B.Xte, B.yte, 3000, X_raw=B.Xte_raw)
for nr, st in [(128, 50), (64, 30)]:
    t=time.time(); r = flow_eval(net, B.flow, Xr, FLOW_RHOS, n_random=nr, pgd_steps=st); torch.cuda.synchronize(); print(nr, st, "eval", round(time.time()-t,1), {k: round(v,3) for k,v in r.items()})
adv = dict(kind="flow", fa=B.flow, X_raw=B.Xtr_raw, rho=1.0, steps=10)
t=time.time(); train(net, P.full_masks(net), B.Xtr, B.ytr, epochs=1, cw=B.cw, adv=adv); print("flowAT 1ep", time.time()-t)
