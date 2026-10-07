"""Tahap 3: evaluasi di pasien test (terkunci). Menghasilkan tiga hal yang dipakai di proposal:
 1) akurasi segmentasi: model tunggal vs ensemble (Dice, HD95)
 2) akurasi volume dinding (mL) vs referensi
 3) apakah flag QA / kesepakatan model benar-benar menandai segmentasi yang jelek"""
import json
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import common as M
import config as C

def main():
    df = pd.read_csv(C.WORK / "manifest.csv")
    te = df[df.split == "test"]
    models, stats = M.load_ensemble(), M.load_stats()
    ds = __import__("monai").data.Dataset([{"image": r.image, "label": r.label} for _, r in te.iterrows()], M.eval_transforms(True))
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    vox_ml = float(np.prod(C.SPACING)) / 1000.0
    rows = []
    for (_, r), s in zip(te.iterrows(), ds):
        gt = s["label"][0].numpy() > 0
        final, p, ent, f, probs = M.analyse_case(models, s, stats)
        single = [M.dice(pm > 0.5, gt) for pm in probs]
        row = {"case": r.case, "patient": r.patient, "dice_ensemble": M.dice(final, gt),
               "dice_single_mean": float(np.mean(single)), "dice_single_min": float(np.min(single)),
               "hd95_mm": M.hd95(final, gt, C.SPACING),
               "vol_gt_ml": float(gt.sum() * vox_ml), **f}
        row["vol_err_pct"] = 100 * (f["wall_volume_ml"] - row["vol_gt_ml"]) / max(row["vol_gt_ml"], 1e-6)
        rows.append(row)
        aff = s["image"].meta["affine"].numpy() if hasattr(s["image"], "meta") else np.eye(4)
        nib.save(nib.Nifti1Image(final.astype(np.uint8), aff), C.RESULT_DIR / f"{r.case}_pred.nii.gz")
        nib.save(nib.Nifti1Image(ent.astype(np.float32), aff), C.RESULT_DIR / f"{r.case}_uncert.nii.gz")
    res = pd.DataFrame(rows); res.to_csv(C.RESULT_DIR / "test_per_case.csv", index=False)

    poor = (res.dice_ensemble < C.POOR_DICE).astype(int)
    summ = {"n_test_scans": len(res), "n_test_patients": int(res.patient.nunique()),
            "dice_ensemble_mean": res.dice_ensemble.mean(), "dice_ensemble_median": res.dice_ensemble.median(),
            "dice_single_model_mean": res.dice_single_mean.mean(), "hd95_mm_median": res.hd95_mm.median(),
            "vol_abs_err_pct_mean": res.vol_err_pct.abs().mean(),
            "vol_pearson_r": float(np.corrcoef(res.wall_volume_ml, res.vol_gt_ml)[0, 1]) if len(res) > 2 else None,
            "n_poor_segmentations": int(poor.sum()), "n_flagged": int(res.qa_needs_review.sum())}
    if len(res) > 2:
        rho, pv = spearmanr(res.agreement, res.dice_ensemble); summ["spearman_agreement_vs_dice"] = [rho, pv]
    if poor.nunique() == 2:
        summ["auroc_agreement_detects_poor"] = roc_auc_score(poor, -res.agreement)
        summ["flag_sensitivity"] = float(res.qa_needs_review[poor == 1].mean())
        summ["flag_specificity"] = float(1 - res.qa_needs_review[poor == 0].mean())
    else:
        summ["catatan"] = "Tidak ada (atau semua) segmentasi buruk di test; AUROC QA tidak bisa dihitung."
    (C.RESULT_DIR / "test_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(res.agreement, res.dice_ensemble, c=res.qa_needs_review, cmap="coolwarm")
    ax[0].axhline(C.POOR_DICE, ls="--", c="gray"); ax[0].axvline(C.AGREEMENT_MIN, ls="--", c="gray")
    ax[0].set_xlabel("Kesepakatan antar-model"); ax[0].set_ylabel("Dice sebenarnya"); ax[0].set_title("QA: apakah kesepakatan menandai error?")
    ax[1].scatter(res.vol_gt_ml, res.wall_volume_ml); lim = [0, max(res.vol_gt_ml.max(), res.wall_volume_ml.max()) * 1.05]
    ax[1].plot(lim, lim, "k--"); ax[1].set_xlabel("Volume referensi (mL)"); ax[1].set_ylabel("Volume prediksi (mL)"); ax[1].set_title("Volume dinding")
    fig.tight_layout(); fig.savefig(C.RESULT_DIR / "test_plots.png", dpi=150)

if __name__ == "__main__":
    main()
