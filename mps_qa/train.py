"""Tahap 2: latih satu model per fold (5-fold, split per pasien). Pakai: python train.py  (semua fold) atau python train.py 2"""
import sys, time
import numpy as np, pandas as pd, torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
import common as M
import config as C

def run_fold(df, k):
    torch.manual_seed(C.SEED + k); np.random.seed(C.SEED + k)
    tv = df[df.split == "trainval"]
    tr = [{"image": r.image, "label": r.label} for _, r in tv[tv.fold != k].iterrows()]
    va = [{"image": r.image, "label": r.label} for _, r in tv[tv.fold == k].iterrows()]
    print(f"[fold {k}] train={len(tr)} scan, val={len(va)} scan")
    tr_dl = DataLoader(CacheDataset(tr, M.train_transforms(), cache_rate=1.0, num_workers=0), batch_size=C.BATCH, shuffle=True)
    va_ds = CacheDataset(va, M.eval_transforms(True), cache_rate=1.0, num_workers=0)

    model = M.build_model()
    loss_fn = DiceCELoss(sigmoid=True)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best, bad, t0 = -1.0, 0, time.time()
    for ep in range(1, C.EPOCHS + 1):
        model.train(); tot = 0
        for b in tr_dl:
            x, y = b["image"].to(M.DEVICE), b["label"].to(M.DEVICE)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if ep % C.VAL_EVERY == 0:
            model.eval(); ds = []
            with torch.no_grad():
                for s in va_ds:
                    p = M.member_probs([model], s["image"][None])[0] > 0.5
                    ds.append(M.dice(p, s["label"][0].numpy() > 0))
            d = float(np.mean(ds))
            print(f"[fold {k}] ep {ep:3d} loss {tot/len(tr_dl):.4f} val_dice {d:.4f} ({time.time()-t0:.0f}s)")
            if d > best:
                best, bad = d, 0; torch.save(model.state_dict(), C.MODEL_DIR / f"fold{k}_best.pth")
            else:
                bad += 1
                if bad >= C.PATIENCE: print("early stop"); break
    print(f"[fold {k}] selesai, val dice terbaik {best:.4f}")
    return best

if __name__ == "__main__":
    df = pd.read_csv(C.WORK / "manifest.csv")
    folds = [int(sys.argv[1])] if len(sys.argv) > 1 else range(C.N_FOLDS)
    res = {k: run_fold(df, k) for k in folds}
    pd.Series(res).to_csv(C.WORK / "cv_val_dice.csv", header=["val_dice"])
    print("CV val dice per fold:", res)
