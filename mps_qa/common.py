"""Fungsi bersama: transform, model, inferensi ensemble, fitur kuantitatif, flag QA."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ResampleToMatchd, Orientationd, Spacingd,
    CopyItemsd, ScaleIntensityRangePercentilesd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandGaussianNoised,
)
import config as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _base(with_label):
    keys = ["image", "label"] if with_label else ["image"]
    t = [LoadImaged(keys=keys, image_only=False), EnsureChannelFirstd(keys=keys)]
    if with_label:
        # paksa masker ke grid citra, jaga-jaga geometri DICOM->NIfTI tidak identik
        t.append(ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest"))
    t += [
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=C.SPACING, mode=("bilinear", "nearest") if with_label else "bilinear"),
        CopyItemsd(keys=["image"], names=["raw"]),   # count mentah, dipakai untuk fitur kuantitatif
        # persentil 99.5 supaya satu hot spot (usus/ekstravasasi) tidak merusak skala seluruh citra
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=keys, spatial_size=C.ROI),
    ]
    return t


def eval_transforms(with_label=True):
    return Compose(_base(with_label))


def train_transforms():
    return Compose(_base(True) + [
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=C.ROI,
                               pos=3, neg=1, num_samples=C.SAMPLES_PER_VOLUME),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.03),
    ])


def build_model():
    return SegResNet(spatial_dims=3, in_channels=1, out_channels=1,
                     init_filters=C.INIT_FILTERS, dropout_prob=0.2).to(DEVICE)


def load_ensemble():
    paths = sorted(C.MODEL_DIR.glob("fold*_best.pth"))
    if not paths:
        raise FileNotFoundError(f"Belum ada model di {C.MODEL_DIR}. Jalankan train.py dulu.")
    models = []
    for p in paths:
        m = build_model()
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def member_probs(models, image):
    """image: tensor [1,1,X,Y,Z]. Return array [K,X,Y,Z] probabilitas tiap anggota ensemble."""
    out = []
    for m in models:
        logit = sliding_window_inference(image.to(DEVICE), C.ROI, 4, m, overlap=0.5)
        out.append(torch.sigmoid(logit)[0, 0].cpu().numpy())
    return np.stack(out)


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(np.uint8), n
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return (lab == (1 + int(np.argmax(sizes)))).astype(np.uint8), n


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def hd95(a, b, spacing):
    a, b = a.astype(bool), b.astype(bool)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    sa = a & ~ndi.binary_erosion(a)
    sb = b & ~ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def ensemble_agreement(binary_members):
    """Rata-rata Dice berpasangan antar anggota ensemble. 1.0 = semua model sepakat."""
    k = len(binary_members)
    if k < 2:
        return float("nan")
    ds = [dice(binary_members[i], binary_members[j]) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(ds))


def analyse_case(models, sample, stats):
    """Satu kasus -> mask akhir, peta ketidakpastian, fitur kuantitatif, flag QA."""
    img = sample["image"][None]
    probs = member_probs(models, img)
    p = probs.mean(0)
    members = probs > 0.5
    final, n_comp = largest_component(p > 0.5)
    eps = 1e-6
    entropy = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))

    raw = sample["raw"][0].numpy()
    vox_ml = float(np.prod(C.SPACING)) / 1000.0
    wall = final.astype(bool)
    feats = {"wall_volume_ml": float(wall.sum() * vox_ml), "n_components_raw": int(n_comp)}
    if wall.sum() > 0:
        vals = raw[wall]
        peak = np.percentile(vals, 95)
        feats.update({
            "wall_mean_counts": float(vals.mean()),
            "wall_cv": float(vals.std() / (vals.mean() + eps)),               # keseragaman: makin kecil makin seragam
            "wall_hypo_fraction": float((vals < C.HYPO_FRACTION * peak).mean()),  # proksi defek relatif, bukan diagnosis
            "mean_entropy_wall": float(entropy[wall].mean()),
        })
        body = raw[raw > 0.05 * raw.max()]
        feats["hotspot_ratio"] = float(raw.max() / (np.percentile(body, 99) + eps))
    feats["agreement"] = ensemble_agreement(members)

    flags = []
    if feats["agreement"] < C.AGREEMENT_MIN:
        flags.append("model_tidak_sepakat")
    if n_comp > 1:
        flags.append("masker_terpecah")
    lo, hi = stats["wall_volume_ml"]
    if not (lo <= feats["wall_volume_ml"] <= hi):
        flags.append("volume_di_luar_rentang_training")
    hmax = C.HOTSPOT_RATIO_MAX or stats.get("hotspot_ratio_p99")
    if hmax and feats.get("hotspot_ratio", 0) > hmax:
        flags.append("hot_spot_ekstrem")
    feats["qa_flags"] = ";".join(flags)
    feats["qa_needs_review"] = int(len(flags) > 0)
    return final, p, entropy, feats, probs


def load_stats():
    f = C.WORK / "train_stats.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"wall_volume_ml": [0, 1e9]}
