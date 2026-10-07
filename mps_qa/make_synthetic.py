"""Bikin data sintetis mirip struktur dataset asli (hanya untuk uji kode, BUKAN untuk hasil lomba).
Dinding ventrikel kiri = cangkang elipsoid; ada beberapa kasus dengan hot spot dan count rendah."""
import numpy as np, nibabel as nib, pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
import config as C

rng = np.random.default_rng(0)
C.NIFTI_DIR.mkdir(parents=True, exist_ok=True)
C.DICOM_DIR.mkdir(parents=True, exist_ok=True)
N_PATIENT, SHAPE = 24, (64, 64, 56)
aff = np.diag([6.4, 6.4, 6.4, 1.0])

def make(seed):
    r = np.random.default_rng(seed)
    g = np.indices(SHAPE).astype(float)
    c = np.array(SHAPE) / 2 + r.normal(0, 3, 3)
    ax = np.array([14, 11, 17]) * r.uniform(0.85, 1.2)
    d = (((g - c[:, None, None, None]) / ax[:, None, None, None]) ** 2).sum(0)
    wall = (d < 1.0) & (d > 0.45)
    img = np.where(wall, 120 * r.uniform(0.8, 1.2), 5.0) + r.normal(0, 4, SHAPE)
    img[wall & (g[0] > c[0] + 6)] *= 0.55          # defek perfusi buatan
    kind = "normal"
    if seed % 7 == 0:
        img[8:12, 8:12, 8:12] += 900; kind = "hotspot"
    if seed % 11 == 0:
        img *= 0.15; kind = "lowcount"
    return np.clip(img, 0, None).astype(np.float32), wall.astype(np.uint8), kind

for p in range(N_PATIENT):
    for scan in range(2 if p % 3 == 0 else 1):     # sebagian pasien punya >1 scan, utk menguji split per pasien
        uid = f"1.2.840.9999.{p:03d}{scan}"
        img, wall, kind = make(p * 10 + scan)
        nib.save(nib.Nifti1Image(img, aff), C.NIFTI_DIR / f"{uid}.nii.gz")
        nib.save(nib.Nifti1Image(wall, aff), C.NIFTI_DIR / f"{uid}_mask.nii.gz")
        meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
        meta.MediaStorageSOPInstanceUID = uid; meta.TransferSyntaxUID = "1.2.840.10008.1.2"
        ds = FileDataset(str(C.DICOM_DIR / f"{uid}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientID = f"PAT{p:03d}"; ds.SOPInstanceUID = uid
        ds.save_as(str(C.DICOM_DIR / f"{uid}.dcm"), enforce_file_format=True)
print("sintetis selesai di", C.DATA_ROOT)
