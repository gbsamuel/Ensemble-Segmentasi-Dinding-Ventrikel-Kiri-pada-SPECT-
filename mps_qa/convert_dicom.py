"""Opsional: konversi DICOM -> NIfTI untuk kasus yang citra NIfTI-nya tidak ada di folder NIfTI/.
Jalankan sebelum prepare.py kalau perlu."""
import SimpleITK as sitk
import config as C

C.VOL_DIR.mkdir(parents=True, exist_ok=True)
ok = 0
for f in sorted(C.DICOM_DIR.glob("*.dcm")):
    try:
        sitk.WriteImage(sitk.ReadImage(str(f)), str(C.VOL_DIR / (f.stem + ".nii.gz"))); ok += 1
    except Exception as e:
        print("GAGAL", f.name, e)
print(f"{ok} file dikonversi ke {C.VOL_DIR}")
