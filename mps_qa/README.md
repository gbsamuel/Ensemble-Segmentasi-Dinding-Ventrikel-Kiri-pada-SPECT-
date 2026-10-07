# Segmentasi dinding ventrikel kiri (MPS SPECT) + kuantifikasi + QA

Urutan jalan (butuh GPU; di Colab cukup T4):

    pip install torch monai nibabel SimpleITK pydicom scipy scikit-learn pandas tqdm matplotlib
    set MPS_DATA=D:\path\myocardial-perfusion-scintigraphy-image-database-1.0.0     (Windows)
    export MPS_DATA=/path/...                                                         (Linux/Colab)
    python convert_dicom.py     # opsional, hanya kalau citra NIfTI tidak ada di folder NIfTI/
    python prepare.py           # cek data + split per pasien + statistik training
    python train.py             # 5 model (5-fold)
    python evaluate.py          # hasil di work/results/test_summary.json + test_plots.png
    python infer.py <folder>    # pakai ke citra baru -> laporan fitur + flag QA

Yang perlu dicek setelah prepare.py: baris "scan dari N pasien unik" dan peringatan geometri.
Kalau banyak kasus "UNKNOWN_", PatientID di DICOM kosong; split per pasien jadi tidak bisa dijamin.
Semua parameter ada di config.py. Mode uji cepat: SMOKE=1 (hanya untuk cek kode di data sintetis).
