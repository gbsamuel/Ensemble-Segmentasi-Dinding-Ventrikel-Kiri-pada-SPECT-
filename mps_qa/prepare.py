"""Tahap 1: cek data, bikin manifest, split PER PASIEN (bukan per file), hitung statistik training.
Kenapa per pasien: satu pasien bisa punya >1 scan. Kalau scan-nya terpecah ke train dan val,
Dice jadi terlalu optimis (bocor)."""
import json
import numpy as np, pandas as pd, nibabel as nib
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import config as C

def patient_id_from_dicom(stem):
    f = C.DICOM_DIR / f"{stem}.dcm"
    if not f.exists():
        return None
    try:
        import pydicom
        pid = str(pydicom.dcmread(f, stop_before_pixels=True).get("PatientID", "")).strip()
        return pid or None
    except Exception:
        return None

def main():
    C.WORK.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for m in sorted(C.NIFTI_DIR.glob("*_mask.nii.gz")):
        stem = m.name.replace("_mask.nii.gz", "")
        img = C.NIFTI_DIR / f"{stem}.nii.gz"
        if not img.exists():
            img = C.VOL_DIR / f"{stem}.nii.gz"       # hasil convert_dicom.py
        if not img.exists():
            problems.append(f"{stem}: citra tidak ditemukan"); continue
        ni, nm = nib.load(img), nib.load(m)
        if ni.shape != nm.shape or not np.allclose(ni.affine, nm.affine, atol=1e-3):
            problems.append(f"{stem}: geometri citra != masker (akan di-resample otomatis)")
        pid = patient_id_from_dicom(stem)
        rows.append({"case": stem, "image": str(img), "label": str(m), "patient": pid or f"UNKNOWN_{stem}",
                     "pid_from_dicom": pid is not None,
                     "mask_voxels": int((nm.get_fdata() > 0).sum()), "voxel_mm": tuple(np.round(nm.header.get_zooms()[:3], 2))})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Tidak ada pasangan citra+masker. Cek MPS_DATA dan struktur folder.")
    if not df.pid_from_dicom.all():
        print(f"PERINGATAN: {(~df.pid_from_dicom).sum()} kasus tidak punya PatientID dari DICOM; "
              "tiap file dianggap pasien berbeda (risiko bocor kalau ternyata satu pasien).")
    print(f"{len(df)} scan dari {df.patient.nunique()} pasien unik")
    for p in problems: print(" -", p)

    # test dikunci per pasien
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_FRACTION, random_state=C.SEED)
    tv_idx, te_idx = next(gss.split(df, groups=df.patient))
    df["split"] = "trainval"; df.loc[df.index[te_idx], "split"] = "test"
    df["fold"] = -1
    tv = df[df.split == "trainval"]
    for k, (_, va) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(tv, groups=tv.patient)):
        df.loc[tv.index[va], "fold"] = k
    df.to_csv(C.WORK / "manifest.csv", index=False)

    # statistik dari data training saja (tanpa test) untuk ambang QA
    trn = df[df.split == "trainval"]
    vox_ml = float(np.prod(C.SPACING)) / 1000.0
    vols = []
    for _, r in trn.iterrows():
        n = (nib.load(r.label).get_fdata() > 0)
        zoom = np.prod(nib.load(r.label).header.get_zooms()[:3])
        vols.append(n.sum() * zoom / 1000.0)
    vols = np.array(vols)
    lo, hi = np.percentile(vols, [1, 99]) if len(vols) > 20 else (vols.min() * 0.7, vols.max() * 1.3)
    ratios = []
    for _, r in trn.iterrows():
        x = nib.load(r.image).get_fdata()
        body = x[x > 0.05 * x.max()]
        ratios.append(x.max() / (np.percentile(body, 99) + 1e-6))
    stats = {"wall_volume_ml": [float(lo), float(hi)], "hotspot_ratio_p99": float(np.percentile(ratios, 99)),
             "n_trainval_scans": int(len(trn)), "n_test_scans": int((df.split == "test").sum())}
    (C.WORK / "train_stats.json").write_text(json.dumps(stats, indent=2))
    print("statistik training:", stats)
    print(df.groupby(["split", "fold"]).size())

if __name__ == "__main__":
    main()
