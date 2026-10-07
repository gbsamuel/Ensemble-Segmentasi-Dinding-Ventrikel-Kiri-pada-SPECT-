"""Pakai pipeline ke citra baru (tanpa masker): python infer.py folder_citra_nifti
Keluaran: mask, peta ketidakpastian, dan results/infer_report.csv berisi fitur kuantitatif + flag QA.
Ini alat bantu pendukung keputusan; BUKAN diagnosis."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from monai.data import Dataset
import common as M
import config as C

def main(folder):
    files = sorted(Path(folder).glob("*.nii.gz"))
    models, stats = M.load_ensemble(), M.load_stats()
    ds = Dataset([{"image": str(f)} for f in files], M.eval_transforms(False))
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, s in zip(files, ds):
        final, p, ent, feats, _ = M.analyse_case(models, s, stats)
        aff = s["image"].meta["affine"].numpy()
        name = f.name.replace(".nii.gz", "")
        nib.save(nib.Nifti1Image(final, aff), C.RESULT_DIR / f"{name}_pred.nii.gz")
        nib.save(nib.Nifti1Image(ent.astype(np.float32), aff), C.RESULT_DIR / f"{name}_uncert.nii.gz")
        rows.append({"case": name, **feats})
    pd.DataFrame(rows).to_csv(C.RESULT_DIR / "infer_report.csv", index=False)
    print(pd.DataFrame(rows).to_string())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else C.NIFTI_DIR)
