"""Konfigurasi tunggal. Ubah di sini saja, jangan di tiap skrip.
Path bisa diganti lewat environment variable supaya jalan di Windows / Colab / Linux."""
import os
from pathlib import Path

# Folder hasil unduhan PhysioNet (yang isinya DICOM/ dan NIfTI/)
DATA_ROOT = Path(os.environ.get("MPS_DATA", "./myocardial-perfusion-scintigraphy-image-database-1.0.0"))
DICOM_DIR = DATA_ROOT / "DICOM"
NIFTI_DIR = DATA_ROOT / "NIfTI"

# Semua keluaran (manifest, model, hasil) masuk ke sini
WORK = Path(os.environ.get("MPS_WORK", "./work"))
VOL_DIR = WORK / "volumes"      # citra hasil konversi DICOM -> NIfTI
MODEL_DIR = WORK / "models"
RESULT_DIR = WORK / "results"

SMOKE = os.environ.get("SMOKE") == "1"   # mode uji cepat di CPU (data sintetis)

SEED = 42
N_FOLDS = 5                 # jumlah model di ensemble
TEST_FRACTION = 0.15        # pasien yang dikunci, tidak disentuh saat training/seleksi model

SPACING = (4.0, 4.0, 4.0)   # mm, semua volume di-resample ke sini supaya volume bisa dihitung dalam mL
ROI = (48, 48, 48) if SMOKE else (96, 96, 64)
EPOCHS = 3 if SMOKE else 120
VAL_EVERY = 1 if SMOKE else 2
PATIENCE = 15               # berhenti kalau Dice validasi tidak naik selama N kali validasi
LR = 2e-4
BATCH = 1
SAMPLES_PER_VOLUME = 2
INIT_FILTERS = 8 if SMOKE else 16

# Ambang QA (semua ditandai "perlu dicek ulang", bukan "pasti salah")
AGREEMENT_MIN = 0.85        # kesepakatan antar-model (Dice antar anggota ensemble) minimal
HOTSPOT_RATIO_MAX = None    # diisi otomatis dari statistik data training (persentil 99)
POOR_DICE = 0.75            # definisi "segmentasi buruk" saat menguji apakah flag QA berguna
HYPO_FRACTION = 0.60        # voxel dinding dgn count < 60% puncak dihitung hipoperfusi relatif
