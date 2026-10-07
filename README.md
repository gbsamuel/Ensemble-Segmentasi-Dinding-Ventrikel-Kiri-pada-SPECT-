# Ensemble Segmentasi Dinding Ventrikel Kiri pada MPS SPECT

Proyek segmentasi dinding ventrikel kiri (Myocardial Perfusion Scintigraphy / MPS SPECT) menggunakan pendekatan Ensemble 5-Fold Deep Learning, kuantifikasi fitur anatomis/fungsional, dan Quality Assurance (QA) otomatis.

---

## 📁 Struktur Direktori

```text
├── mps_qa/                                                   # Kode sumber, pipeline, dan notebook
│   ├── PIM_MPS_Pipeline.ipynb                                # Notebook alur kerja utama end-to-end
│   ├── PIM_MPS_Pipeline_hasil_run_CPU.ipynb                  # Notebook hasil eksekusi mode CPU
│   ├── config.py                                             # Konfigurasi parameter & path
│   ├── common.py                                             # Helper & fungsi utilitas
│   ├── convert_dicom.py                                      # Konversi data DICOM -> NIfTI
│   ├── prepare.py                                            # Verifikasi geometri & patient-wise splitting
│   ├── train.py                                              # Pelatihan ensemble 5-fold model
│   ├── evaluate.py                                           # Evaluasi performa ensemble & metric
│   ├── infer.py                                              # Inferensi citra baru + laporan QA
│   ├── make_synthetic.py                                     # Pembuatan data sintetis untuk pengujian
│   └── myocardial-perfusion-scintigraphy-image-database-1.0.0/ # Subset dataset PhysioNet
├── work_mps/                                                 # Output eksekusi, bobot model, & manifest
│   ├── hasil/                                                # Log metrik training per fold (CSV)
│   ├── models/                                               # Model checkpoints (fold0.pth - fold4.pth)
│   └── manifest.csv                                          # Manifest pembagian data
└── README.md
```

---

## 🚀 Persiapan & Instalasi

Pastikan pustaka Python berikut telah terpasang:

```bash
pip install torch monai nibabel SimpleITK pydicom scipy scikit-learn pandas tqdm matplotlib
```

---

## 🏃 Urutan Eksekusi

1. **Konfigurasi Path Dataset (Opsional)**
   - Windows PowerShell:
     ```powershell
     $env:MPS_DATA=".\mps_qa\myocardial-perfusion-scintigraphy-image-database-1.0.0"
     ```
   - Linux / Colab:
     ```bash
     export MPS_DATA="./mps_qa/myocardial-perfusion-scintigraphy-image-database-1.0.0"
     ```

2. **Jalankan Pipeline**
   ```bash
   cd mps_qa
   python prepare.py           # Split per pasien + audit data
   python train.py             # Pelatihan 5-fold ensemble
   python evaluate.py          # Evaluasi hasil di test set
   python infer.py <path_citra> # Inferensi citra baru & analisis QA
   ```
   Atau buka dan jalankan notebook [PIM_MPS_Pipeline.ipynb](mps_qa/PIM_MPS_Pipeline.ipynb).

---

## 📚 Sitasi Dataset
Dataset yang digunakan merujuk pada:
- Calixto et al. (2025). *Myocardial Perfusion Scintigraphy Image Database* (v1.0.0). PhysioNet. [DOI: 10.13026/ce2z-dw74](https://doi.org/10.13026/ce2z-dw74).
- Nogueira et al. (2025), PLoS ONE.
