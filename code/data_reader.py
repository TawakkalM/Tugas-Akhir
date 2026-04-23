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

# Parameter STFT 
SAMPLE_RATE    = 200      # Hz
WINDOW_SEC     = 5        # detik per segmen
SAMPLE_LEN     = SAMPLE_RATE * WINDOW_SEC   # = 1000 titik

STFT_NPERSEG   = 256      # window length STFT
STFT_NOVERLAP  = 128      # 50% overlap  →  hop = 128
STFT_NFFT      = 256      # ukuran FFT

IMG_SIZE       = 224      # ukuran akhir gambar untuk ResNet-18

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

# Indeks tiap zona dalam array 62 kanal
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

    labels = read_labels(label_path)
    mat_files = list_mat_files(data_dir)

    label_names = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    total_trials = 0
    total_segments_est = 0

    for mat_file in mat_files:
        path = os.path.join(data_dir, mat_file)
        subj_id = mat_file.split('_')[0]
        mat = loadmat(path)
        eeg_keys = detect_eeg_keys(mat)

        print(f"\n[Subjek {subj_id}] — {mat_file}")
        print(f"  Jumlah trial: {len(eeg_keys)}")

        for i, key in enumerate(eeg_keys):
            eeg = mat[key]
            n_ch, n_samp = eeg.shape
            dur_sec = n_samp / SAMPLE_RATE
            n_seg = int(dur_sec // WINDOW_SEC)
            lbl = labels[i] if i < len(labels) else '?'
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
    n_seg = n_samp // window_len
    segments = []
    for i in range(n_seg):
        start = i * window_len
        seg = eeg[:, start:start + window_len]
        segments.append(seg)
    return segments   # list of (62, 1000)


def compute_stft_spectrogram(channel_data):
    """
    Hitung STFT untuk satu kanal sinyal EEG (panjang 1000).
    Magnitude dikonversi ke skala logaritmik (dB) agar
    frekuensi tinggi dan rendah sama-sama terlihat jelas.

    Output shape: (n_freq_bins, time_frames)
    di mana n_freq_bins sesuai dengan 0–75 Hz
    """
    _, _, Zxx = stft(
        channel_data,
        fs=SAMPLE_RATE,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT
    )
    magnitude = np.abs(Zxx)

    # Konversi ke skala log (dB) — mencegah frekuensi rendah mendominasi
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

    Menggunakan persentil 2–98 sebagai batas bawah dan atas
    untuk memotong nilai ekstrem, sehingga gambar tidak
    terlalu gelap atau terlalu terang.
    """
    p_low  = np.percentile(spec, 2)
    p_high = np.percentile(spec, 98)

    # Hindari pembagian nol
    if p_high - p_low < 1e-10:
        return np.zeros_like(spec, dtype=np.uint8)

    # Clip nilai di luar rentang persentil lalu normalisasi
    spec_clipped = np.clip(spec, p_low, p_high)
    normalized   = (spec_clipped - p_low) / (p_high - p_low) * 255
    return normalized.astype(np.uint8)


def segment_to_rgb_image(segment):
    """
    Tahap 3 inti: Ubah satu segmen EEG (62 × 1000) menjadi
    citra RGB 224×224 dengan pemetaan zona otak:
      - Channel R (Merah)  = zona Frontal
      - Channel G (Hijau)  = zona Central & Temporal
      - Channel B (Biru)   = zona Parietal & Occipital

    Output: torch.Tensor (3, 224, 224) — siap masuk ResNet-18.
    """
    # Hitung spektrogram rata-rata tiap zona
    spec_r = zone_average_spectrogram(segment, IDX_FRONTAL)
    spec_g = zone_average_spectrogram(segment, IDX_CENTRAL_T)
    spec_b = zone_average_spectrogram(segment, IDX_PARIETAL)

    # Normalisasi masing-masing kanal ke [0, 255]
    r = normalize_spectrogram(spec_r)
    g = normalize_spectrogram(spec_g)
    b = normalize_spectrogram(spec_b)

    # Susun menjadi gambar RGB (H, W, 3)
    rgb = np.stack([r, g, b], axis=-1)   # (freq, time, 3)

    # Konversi ke PIL Image lalu resize ke 224×224
    img = Image.fromarray(rgb, mode='RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # Konversi ke tensor PyTorch (3, 224, 224), nilai [0.0, 1.0]
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)

    return tensor   # torch.Tensor (3, 224, 224)


# ==============================================================
# TAHAP 4 — PEMBAGIAN DATA (SUBJECT-WISE 5-FOLD)
# ==============================================================

def get_subject_ids(data_dir):
    """
    Ambil daftar ID subjek unik dari nama file .mat,
    urutkan secara numerik.
    Contoh: ['1', '2', ..., '15']
    """
    mat_files = list_mat_files(data_dir)
    subject_ids = []
    for f in mat_files:
        sid = f.split('_')[0]
        if sid not in subject_ids:
            subject_ids.append(sid)
    return subject_ids   # list of string


def subject_wise_kfold(subject_ids, n_splits=5, seed=RANDOM_SEED):
    """
    Bagi daftar subjek menjadi 5 fold.
    Setiap fold berisi: train_subjects, val_subjects.

    Dengan 15 subjek dan 5 fold → 3 subjek per fold untuk validasi.
    Pembagian bersifat deterministik (seed=1907).
    """
    rng = np.random.default_rng(seed)
    ids = np.array(subject_ids)
    shuffled = ids[rng.permutation(len(ids))]

    folds = np.array_split(shuffled, n_splits)
    result = []
    for i in range(n_splits):
        val_subjects   = list(folds[i])
        train_subjects = [s for j, f in enumerate(folds)
                          for s in f if j != i]
        result.append((train_subjects, val_subjects))

    return result   # list of (train_subjects, val_subjects)


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
# DATASET CLASS (Menggabungkan Tahap 2, 3, 4)
# ==============================================================

class EEGDataset(Dataset):
    """
    Dataset PyTorch untuk sinyal EEG SEED.

    Alur kerja:
    1. Baca semua file .mat dan metadata trial
    2. Filter trial berdasarkan subjek (train/val dari subject-wise fold)
    3. Saat __getitem__ dipanggil:
       - Muat sinyal EEG trial
       - Potong jadi segmen 5 detik (tanpa overlap)
       - Pilih satu segmen secara acak
       - Transformasi segmen → spektrogram RGB 224×224
       - Kembalikan tensor (3, 224, 224) + label
    """

    def __init__(self,
                 data_dir='/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG',
                 label_path='/kaggle/input/datasets/tawakkal19/kode/label.csv',
                 fold=0,
                 split='train',
                 n_splits=5):

        self.data_dir  = data_dir
        self.split     = split

        # Baca label
        self.labels = read_labels(label_path)

        # Bangun daftar semua (subject_id, path, eeg_key, label)
        self.all_trials = self._collect_trials()

        # Buat pembagian subject-wise fold
        subject_ids = get_subject_ids(data_dir)
        folds       = subject_wise_kfold(subject_ids, n_splits=n_splits)
        train_subs, val_subs = folds[fold]

        # Filter trial sesuai split
        target_subs = train_subs if split == 'train' else val_subs
        self.data = [t for t in self.all_trials if t[0] in target_subs]

        print(f"\n✓ EEGDataset [{split.upper()}] Fold {fold+1} "
              f"— {len(self.data)} trial "
              f"dari {len(target_subs)} subjek")

    # ----------------------------------------------------------
    def _collect_trials(self):
        """Kumpulkan metadata semua trial dari semua file .mat."""
        trials = []
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
        subj_id, path, key, label = self.data[idx]

        # Muat sinyal EEG (62 × N)
        eeg = loadmat(path)[key]

        # Potong menjadi segmen 5 detik
        segments = segment_eeg(eeg)

        # Pilih satu segmen secara acak
        seg = random.choice(segments)

        # Transformasi segmen → citra RGB 224×224
        tensor = segment_to_rgb_image(seg)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return tensor, label_tensor


# ==============================================================
# TESTING SCRIPT
# ==============================================================

if __name__ == "__main__":
    DATA_DIR   = "/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG"
    LABEL_PATH = "/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv"

    # --- Tahap 2: Eksplorasi dataset ---
    explore_dataset(DATA_DIR, LABEL_PATH)

    # --- Tahap 4: Cek pembagian fold ---
    subject_ids = get_subject_ids(DATA_DIR)
    folds = subject_wise_kfold(subject_ids)
    print_fold_info(folds)

    # --- Uji Dataset class (Fold 1) ---
    print("=== Uji EEGDataset ===")
    train_ds = EEGDataset(DATA_DIR, LABEL_PATH, fold=0, split='train')
    val_ds   = EEGDataset(DATA_DIR, LABEL_PATH, fold=0, split='val')

    print(f"Train: {len(train_ds)} trial | Val: {len(val_ds)} trial")

    # Ambil satu sampel dan periksa shape
    img_tensor, lbl = train_ds[0]
    label_names = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    print(f"\nContoh output:")
    print(f"  Tensor shape : {img_tensor.shape}  (harusnya: torch.Size([3, 224, 224]))")
    print(f"  Label        : {lbl.item()} ({label_names[lbl.item()]})")
    print(f"  Nilai min/max: {img_tensor.min():.3f} / {img_tensor.max():.3f}")

    # Visualisasi spektrogram RGB dari satu sampel
    img_np = img_tensor.permute(1, 2, 0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.title(f"Spektrogram RGB — Label: {label_names[lbl.item()]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_spectrogram.png", dpi=100)
    plt.show()
    print("\n✓ Gambar spektrogram disimpan ke sample_spectrogram.png")
