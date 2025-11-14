import os
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from scipy.io import loadmat
import matplotlib.pyplot as plt

# ====== Konfigurasi Global ======
RANDOM_SEED = 1907
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ====== Fungsi Utilitas ======
def detect_eeg_keys(mat):
    """
    Deteksi prefix variabel EEG (misal: 'djc_eeg', 'phl_eeg', 'ww_eeg').
    Kembalikan list nama field EEG (urut 1–15).
    """
    valid_keys = [k for k in mat.keys() if not k.startswith("__")]

    prefix = next(
        (k.split("_eeg1")[0] for k in valid_keys if "_eeg1" in k),
        next((k.split("_eeg")[0] for k in valid_keys if "_eeg" in k), None)
    )

    if prefix is None:
        print(f"Tidak ditemukan prefix EEG, dilewati. Keys: {valid_keys}")
        return []

    eeg_keys = [f"{prefix}_eeg{i}" for i in range(1, 16) if f"{prefix}_eeg{i}" in valid_keys]
    eeg_keys.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return eeg_keys


def read_labels(label_path):
    df = pd.read_csv(label_path, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if not {'filmname', 'label'}.issubset(df.columns):
        raise ValueError(f"Kolom CSV tidak sesuai. Ditemukan: {list(df.columns)}")
    return df['label'].astype(int).tolist()


def list_mat_files(data_dir):
    mat_files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.mat') and f[0].isdigit()
    ]
    mat_files.sort(key=lambda f: int(f.split('_')[0]))
    return mat_files


# ====== Dataset Class ======
class EEGDataset(Dataset):
    """
    - Tiap file/trial hanya diambil 1 window acak per epoch
    - Window diacak ulang setiap epoch 
    """

    def __init__(self,
                 data_dir='Preprocessed_EEG',
                 label_path='label.csv',
                 fold=0,
                 split='train',
                 n_splits=5,
                 window_sec=5,
                 sample_rate=200,
                 transform=None):

        self.data_dir = data_dir
        self.label_path = label_path
        self.fold = fold
        self.split = split
        self.n_splits = n_splits
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.transform = transform
        self.sample_len = window_sec * sample_rate

        # === Baca label ===
        self.labels = read_labels(label_path)

        # === Ambil semua file .mat ===
        mat_files = list_mat_files(data_dir)

        # === Kumpulkan metadata setiap trial ===
        self.data_info = self._collect_data_info(mat_files)

        # === Split Train/Val ===
        self.data = self._split_data()

        print(f"\n Dataset {split.upper()} siap digunakan ({len(self.data)} trial).")

    # --------------------------
    def _collect_data_info(self, mat_files):
        data_info = []
        for mat_file in mat_files:
            path = os.path.join(self.data_dir, mat_file)
            part_id = mat_file.split('_')[0]
            mat = loadmat(path)
            eeg_keys = detect_eeg_keys(mat)

            for i, key in enumerate(eeg_keys):
                label = self.labels[i] if i < len(self.labels) else None
                data_info.append((part_id, path, key, label))
        return data_info

    # --------------------------
    def _split_data(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=RANDOM_SEED)
        folds = list(kf.split(self.data_info))
        train_idx, val_idx = folds[self.fold]
        return [self.data_info[i] for i in (train_idx if self.split == 'train' else val_idx)]

    # --------------------------
    def __len__(self):
        # Panjang dataset = jumlah trial (1 window per trial)
        return len(self.data)

    def __getitem__(self, idx):
        """
        Di sini window diambil secara acak setiap kali dipanggil (dynamic windowing).
        """
        part_id, path, key, label = self.data[idx]
        eeg = loadmat(path)[key]  # (62, N)
        n_channels, n_samples = eeg.shape

        if n_samples <= self.sample_len:
            start = 0
        else:
            start = random.randint(0, n_samples - self.sample_len)

        data = eeg[:, start:start + self.sample_len]

        # === DEBUG: tampilkan info pemotongan acak ===
        print(f"[DEBUG] idx={idx:03d} | Partisipan={part_id} | Trial={key} | "
            f"Start index={start} | n_samples={n_samples}")

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y, part_id, key, start



# ====== Testing Script ======
if __name__ == "__main__":
    print("=== Memulai pembacaan dataset SEED EEG (Dynamic Window) ===")
    data_dir = "Preprocessed_EEG"
    label_path = "label.csv"

    train_dataset = EEGDataset(data_dir=data_dir, label_path=label_path, split='train', fold=0)
    # DEBUG: Print ke terminal untuk train 10 trial pertama
    print("\n--- DEBUG: Train Trials ---")
    for i in range(10):
        train_dataset[i]

    val_dataset = EEGDataset(data_dir=data_dir, label_path=label_path, split='val', fold=0)
    # DEBUG: Print ke terminal untuk validation 10 trial pertama
    print("\n--- DEBUG: Validation Trials ---")
    for i in range(10):
        val_dataset[i]

    print(f"\nTrain Trials: {len(train_dataset)} | Val Trials: {len(val_dataset)}")

    # === Ambil contoh acak ===
    idx = random.randint(0, len(train_dataset) - 1)
    signals, label, part_id, trial, start = train_dataset[idx]

    print(f"\n Contoh Data:")
    print(f"Partisipan: {part_id} | Trial: {trial} | Label: {label.item()}")
    print(f"Signals shape: {signals.shape} | Start index: {start}")

    # === Plot beberapa channel ===
    fs = train_dataset.sample_rate
    time_axis = np.arange(signals.shape[1]) / fs

    plt.figure(figsize=(12, 8))
    for i in range(5):
        plt.plot(time_axis, signals[i].numpy() + i * 200, label=f"Ch-{i+1}")
    plt.title(f"EEG Signals | Partisipan {part_id} | {trial} | Label: {label.item()}")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo (µV) [offset antar channel]")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
