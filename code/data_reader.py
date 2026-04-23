import os
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.signal import stft
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# ==============================================================
# KONFIGURASI GLOBAL
# ==============================================================
RANDOM_SEED = 1907
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Parameter STFT (sesuai proposal)
SAMPLE_RATE   = 200    # Hz
WINDOW_SEC    = 5      # detik per segmen
SAMPLE_LEN    = SAMPLE_RATE * WINDOW_SEC  # = 1000 titik

STFT_NPERSEG  = 256    # window length STFT
STFT_NOVERLAP = 128    # 50% overlap → hop = 128
STFT_NFFT     = 256    # ukuran FFT

IMG_SIZE      = 224    # ukuran akhir gambar untuk ResNet-18

# ------------------------------------------------------------------
# Pembagian 62 elektroda ke 3 zona otak (sesuai proposal & SEED)
# Merah  = Frontal
# Hijau  = Central & Temporal
# Biru   = Parietal & Occipital
# ------------------------------------------------------------------
FRONTAL_CH = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'
]

CENTRAL_TEMPORAL_CH = [
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8'
]

PARIETAL_OCCIPITAL_CH = [
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2'
]

# Urutan lengkap 62 kanal SEED (indeks 0–61)
SEED_62CH = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2'
]

def _zone_indices(zone_names):
    return [SEED_62CH.index(ch) for ch in zone_names if ch in SEED_62CH]

IDX_FRONTAL   = _zone_indices(FRONTAL_CH)
IDX_CENTRAL_T = _zone_indices(CENTRAL_TEMPORAL_CH)
IDX_PARIETAL  = _zone_indices(PARIETAL_OCCIPITAL_CH)


# ==============================================================
# TAHAP 2 — FUNGSI UTILITAS & EKSPLORASI DATA
# ==============================================================

def detect_eeg_keys(mat):
    """
    Deteksi nama variabel EEG dalam file .mat secara otomatis.
    Kembalikan list key terurut (eeg1 … eeg15).
    """
    valid_keys = [k for k in mat.keys() if not k.startswith("__")]
    prefix = next(
        (k.split("_eeg1")[0] for k in valid_keys if "_eeg1" in k),
        next((k.split("_eeg")[0] for k in valid_keys if "_eeg" in k), None)
    )
    if prefix is None:
        print(f"[WARN] Prefix EEG tidak ditemukan. Keys: {valid_keys}")
        return []
    eeg_keys = [f"{prefix}_eeg{i}" for i in range(1, 16)
                if f"{prefix}_eeg{i}" in valid_keys]
    eeg_keys.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return eeg_keys


def read_labels(label_path):
    """
    Baca label dari CSV. Label SEED asli: -1 / 0 / 1.
    Di-mapping ke: 0 (Negatif) / 1 (Netral) / 2 (Positif).
    """
    df = pd.read_csv(label_path, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if not {'filmname', 'label'}.issubset(df.columns):
        raise ValueError(f"Kolom CSV tidak sesuai: {list(df.columns)}")

    labels = df['label'].astype(int).tolist()
    unique = sorted(set(labels))
    print(f"[INFO] Label unik dalam CSV: {unique}")

    if unique == [-1, 0, 1]:
        labels = [l + 1 for l in labels]
        print("[INFO] Mapping: -1→0 (Neg), 0→1 (Net), 1→2 (Pos)")
    elif unique == [1, 2, 3]:
        labels = [l - 1 for l in labels]
        print("[INFO] Mapping: 1→0, 2→1, 3→2")
    elif unique == [0, 1, 2]:
        print("[INFO] Label sudah dalam rentang [0,1,2]")
    else:
        raise ValueError(f"Label tidak dikenal: {unique}")

    return labels


def list_mat_files(data_dir):
    """Ambil semua file .mat, urutkan berdasarkan nomor subjek."""
    files = [f for f in os.listdir(data_dir)
             if f.endswith('.mat') and f[0].isdigit()]
    files.sort(key=lambda f: int(f.split('_')[0]))
    return files


def explore_dataset(data_dir, label_path):
    """
    Tahap 2: Eksplorasi dataset.
    Cetak statistik dimensi sinyal, jumlah segmen estimasi,
    dan distribusi kelas per subjek.
    """
    print("\n" + "="*60)
    print("TAHAP 2 — EKSPLORASI DATASET")
    print("="*60)

    labels    = read_labels(label_path)
    mat_files = list_mat_files(data_dir)

    label_names        = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    total_trials       = 0
    total_segments_est = 0

    for mat_file in mat_files:
        path     = os.path.join(data_dir, mat_file)
        subj_id  = mat_file.split('_')[0]
        mat      = loadmat(path)
        eeg_keys = detect_eeg_keys(mat)

        print(f"\n[Subjek {subj_id}] — {mat_file}")
        print(f"  Jumlah trial: {len(eeg_keys)}")

        for i, key in enumerate(eeg_keys):
            eeg     = mat[key]
            n_ch, n_samp = eeg.shape
            dur_sec = n_samp / SAMPLE_RATE
            n_seg   = int(dur_sec // WINDOW_SEC)
            lbl     = labels[i] if i < len(labels) else '?'
            print(f"  Trial {i+1:2d} | shape: {n_ch}×{n_samp} "
                  f"| durasi: {dur_sec:.1f}s "
                  f"| segmen: {n_seg} "
                  f"| label: {label_names.get(lbl, lbl)}")
            total_segments_est += n_seg

        total_trials += len(eeg_keys)

    print(f"\n{'='*60}")
    print(f"Total file .mat   : {len(mat_files)}")
    print(f"Total trial       : {total_trials}")
    print(f"Estimasi segmen   : {total_segments_est}")
    print(f"{'='*60}\n")


# ==============================================================
# TAHAP 3 — PRA-PEMROSESAN & TRANSFORMASI STFT → RGB
# ==============================================================

def segment_eeg(eeg, window_len=SAMPLE_LEN):
    """
    Potong sinyal EEG (62 × N) menjadi segmen-segmen
    berukuran (62 × window_len) tanpa overlap.
    Sisa yang tidak habis dibagi dibuang.
    """
    n_ch, n_samp = eeg.shape
    n_seg        = n_samp // window_len
    segments     = []
    for i in range(n_seg):
        start = i * window_len
        seg   = eeg[:, start:start + window_len]
        segments.append(seg)
    return segments   # list of (62, 1000)


def compute_stft_spectrogram(channel_data):
    """
    Hitung STFT untuk satu kanal sinyal EEG (panjang 1000).
    Magnitude dikonversi ke skala logaritmik (dB) agar
    frekuensi tinggi dan rendah sama-sama terlihat jelas.
    """
    _, _, Zxx = stft(
        channel_data,
        fs=SAMPLE_RATE,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT
    )
    magnitude    = np.abs(Zxx)
    magnitude_db = 10 * np.log10(magnitude + 1e-10)

    return magnitude_db   # shape: (129, time_frames)


def zone_average_spectrogram(segment, zone_indices):
    """
    Rata-ratakan spektrogram STFT dari semua kanal dalam satu zona.
    Output: satu spektrogram 2D representatif zona tersebut.
    """
    specs = []
    for idx in zone_indices:
        spec = compute_stft_spectrogram(segment[idx])
        specs.append(spec)
    return np.mean(specs, axis=0)   # (129, time_frames)


def normalize_spectrogram(spec):
    """
    Normalisasi spektrogram ke rentang [0, 255].
    Menggunakan persentil 2–98 untuk memotong nilai ekstrem.
    """
    p_low  = np.percentile(spec, 2)
    p_high = np.percentile(spec, 98)

    if p_high - p_low < 1e-10:
        return np.zeros_like(spec, dtype=np.uint8)

    spec_clipped = np.clip(spec, p_low, p_high)
    normalized   = (spec_clipped - p_low) / (p_high - p_low) * 255
    return normalized.astype(np.uint8)


def segment_to_rgb_image(segment):
    """
    Ubah satu segmen EEG (62 × 1000) menjadi citra RGB 224×224.
      - Channel R (Merah)  = zona Frontal
      - Channel G (Hijau)  = zona Central & Temporal
      - Channel B (Biru)   = zona Parietal & Occipital

    Output: torch.Tensor (3, 224, 224) — siap masuk ResNet-18.
    """
    spec_r = zone_average_spectrogram(segment, IDX_FRONTAL)
    spec_g = zone_average_spectrogram(segment, IDX_CENTRAL_T)
    spec_b = zone_average_spectrogram(segment, IDX_PARIETAL)

    r = normalize_spectrogram(spec_r)
    g = normalize_spectrogram(spec_g)
    b = normalize_spectrogram(spec_b)

    rgb    = np.stack([r, g, b], axis=-1)
    img    = Image.fromarray(rgb, mode='RGB')
    img    = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = transforms.ToTensor()(img)

    return tensor   # torch.Tensor (3, 224, 224)


# ==============================================================
# TAHAP 4 — PEMBAGIAN DATA (SUBJECT-WISE 5-FOLD)
# ==============================================================

def get_subject_ids(data_dir):
    """
    Ambil daftar ID subjek unik dari nama file .mat.
    Contoh: ['1', '2', ..., '15']
    """
    mat_files   = list_mat_files(data_dir)
    subject_ids = []
    for f in mat_files:
        sid = f.split('_')[0]
        if sid not in subject_ids:
            subject_ids.append(sid)
    return subject_ids


def subject_wise_kfold(subject_ids, n_splits=5, seed=RANDOM_SEED):
    """
    Bagi daftar subjek menjadi 5 fold secara deterministik.
    Dengan 15 subjek → 3 subjek per fold untuk validasi.
    """
    rng      = np.random.default_rng(seed)
    ids      = np.array(subject_ids)
    shuffled = ids[rng.permutation(len(ids))]

    folds  = np.array_split(shuffled, n_splits)
    result = []
    for i in range(n_splits):
        val_subjects   = list(folds[i])
        train_subjects = [s for j, f in enumerate(folds)
                          for s in f if j != i]
        result.append((train_subjects, val_subjects))

    return result


def print_fold_info(folds):
    """Cetak ringkasan pembagian subjek tiap fold."""
    print("\n" + "="*60)
    print("TAHAP 4 — SUBJECT-WISE 5-FOLD CROSS VALIDATION")
    print("="*60)
    for i, (train_subs, val_subs) in enumerate(folds):
        print(f"\nFold {i+1}:")
        print(f"  Train ({len(train_subs)} subjek): {sorted(train_subs, key=int)}")
        print(f"  Val   ({len(val_subs)} subjek) : {sorted(val_subs, key=int)}")
    print()


# ==============================================================
# DATASET CLASS
# Mendukung dua mode:
#   1. Mode Gambar  — baca PNG hasil precompute.py (ringan di CPU)
#   2. Mode Fallback — baca .mat langsung (default lama)
# ==============================================================

class EEGDataset(Dataset):
    """
    Dataset PyTorch untuk sinyal EEG SEED.

    Mode Gambar (img_dir diisi):
    - __getitem__ hanya membaca file PNG yang sudah jadi
    - Tidak ada komputasi STFT → sangat ringan di CPU
    - GPU bisa dimanfaatkan maksimal

    Mode Fallback (img_dir=None):
    - Perilaku sama seperti versi sebelumnya
    - Baca .mat → windowing acak → STFT → tensor
    """

    def __init__(self,
                 data_dir='/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG',
                 label_path='/kaggle/input/datasets/tawakkal19/kode/label.csv',
                 fold=0,
                 split='train',
                 n_splits=5,
                 img_dir=None):

        self.data_dir = data_dir
        self.split    = split
        self.img_dir  = img_dir

        # Baca label
        self.labels = read_labels(label_path)

        # Buat pembagian subject-wise fold
        subject_ids      = get_subject_ids(data_dir)
        folds            = subject_wise_kfold(subject_ids, n_splits=n_splits)
        train_subs, val_subs = folds[fold]
        self.target_subs = train_subs if split == 'train' else val_subs

        if img_dir is not None and os.path.isdir(img_dir):
            # Mode Gambar
            self.mode = 'image'
            self.data = self._collect_image_paths()
            print(f"\n✓ EEGDataset [{split.upper()}] Fold {fold+1} "
                  f"— MODE GAMBAR")
            print(f"  Total gambar : {len(self.data):,}")
        else:
            # Mode Fallback
            self.mode       = 'fallback'
            self.all_trials = self._collect_trials()
            self.data       = [t for t in self.all_trials
                               if t[0] in self.target_subs]
            print(f"\n✓ EEGDataset [{split.upper()}] Fold {fold+1} "
                  f"— MODE FALLBACK")
            print(f"  Total trial  : {len(self.data):,}")

    # ----------------------------------------------------------
    def _collect_image_paths(self):
        """
        Kumpulkan path semua PNG dari img_dir
        yang subj_id-nya ada di target_subs.

        Konvensi nama file dari preprocess.py (format semua segmen):
          subj{id}_sesi{tanggal}_trial{idx}_seg{nomor}_label{label}.png
          Contoh: subj01_sesi20131027_trial07_seg03_label2.png

        Ekstraksi:
          - subj_id : parts[0] → 'subj01' → '1'
          - label   : parts[-1] → 'label2.png' → 2
        """
        all_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.png')
        ])

        result = []
        for fname in all_files:
            parts   = fname.split('_')

            # subj_id selalu di bagian pertama: 'subj01' → '1'
            subj_id = parts[0].replace('subj', '').lstrip('0') or '0'

            if subj_id not in self.target_subs:
                continue

            # label selalu di bagian terakhir: 'label2.png' → 2
            label_part = parts[-1].replace('label', '').replace('.png', '')
            label      = int(label_part)
            result.append((os.path.join(self.img_dir, fname), label))

        return result

    # ----------------------------------------------------------
    def _collect_trials(self):
        """Kumpulkan metadata semua trial dari semua file .mat."""
        trials    = []
        mat_files = list_mat_files(self.data_dir)
        for mat_file in mat_files:
            path    = os.path.join(self.data_dir, mat_file)
            subj_id = mat_file.split('_')[0]
            mat     = loadmat(path)
            keys    = detect_eeg_keys(mat)
            for i, key in enumerate(keys):
                label = self.labels[i] if i < len(self.labels) else None
                trials.append((subj_id, path, key, label))
        return trials

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'image':
            # Hanya baca PNG — ringan di CPU
            img_path, label = self.data[idx]
            img    = Image.open(img_path).convert('RGB')
            tensor = transforms.ToTensor()(img)
            return tensor, torch.tensor(label, dtype=torch.long)

        else:
            # Mode fallback — baca .mat → STFT
            subj_id, path, key, label = self.data[idx]
            eeg      = loadmat(path)[key]
            segments = segment_eeg(eeg)
            seg      = random.choice(segments)
            tensor   = segment_to_rgb_image(seg)
            return tensor, torch.tensor(label, dtype=torch.long)


# ==============================================================
# TESTING SCRIPT
# ==============================================================

if __name__ == "__main__":
    DATA_DIR   = "/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG"
    LABEL_PATH = "/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv"

    explore_dataset(DATA_DIR, LABEL_PATH)

    subject_ids = get_subject_ids(DATA_DIR)
    folds       = subject_wise_kfold(subject_ids)
    print_fold_info(folds)

    print("=== Uji EEGDataset (mode fallback) ===")
    train_ds = EEGDataset(DATA_DIR, LABEL_PATH, fold=0, split='train')
    val_ds   = EEGDataset(DATA_DIR, LABEL_PATH, fold=0, split='val')

    print(f"Train: {len(train_ds)} trial | Val: {len(val_ds)} trial")

    img_tensor, lbl = train_ds[0]
    label_names     = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    print(f"\nContoh output:")
    print(f"  Tensor shape : {img_tensor.shape}")
    print(f"  Label        : {lbl.item()} ({label_names[lbl.item()]})")
    print(f"  Nilai min/max: {img_tensor.min():.3f} / {img_tensor.max():.3f}")

    img_np = img_tensor.permute(1, 2, 0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.title(f"Spektrogram RGB — Label: {label_names[lbl.item()]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_spectrogram.png", dpi=100)
    plt.show()
    print("\n✓ Gambar spektrogram disimpan ke sample_spectrogram.png")
