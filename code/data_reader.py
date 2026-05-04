import os
import re
import random
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.signal import stft
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# ==============================================================
# 1. KONFIGURASI GLOBAL & PARAMETER PENELITIAN
# ==============================================================
RANDOM_SEED = 1907 # Sesuai ketetapan reproduktibilitas
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Parameter Sinyal & STFT (Sesuai Bab 3.2 Laporan Tugas Akhir)
SAMPLE_RATE   = 200    # Hz
WINDOW_SEC    = 5      # detik per segmen
SAMPLE_LEN    = SAMPLE_RATE * WINDOW_SEC  # 1000 titik sampel
MAX_SEGMENTS  = 45     # Maksimal segmen per trial (Target ~30.375 gambar)

STFT_NPERSEG  = 256    # Window length
STFT_NOVERLAP = 128    # Overlap 50%
IMG_SIZE      = 224    # Dimensi input ResNet-18

# ------------------------------------------------------------------
# [V1 - ORIGINAL PROPOSAL] Pemetaan RGB berdasarkan ZONA OTAK
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
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
    'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7',
    'P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2'
]

ALL_CH_IDX = list(range(len(SEED_62CH)))

def _zone_indices(zone_names):
    return [SEED_62CH.index(ch) for ch in zone_names if ch in SEED_62CH]

# Pemetaan indeks array berdasarkan zona
IDX_FRONTAL   = _zone_indices(FRONTAL_CH)
IDX_CENTRAL_T = _zone_indices(CENTRAL_TEMPORAL_CH)
IDX_PARIETAL  = _zone_indices(PARIETAL_OCCIPITAL_CH)

# ==============================================================
# 2. LOGIKA PEMROSESAN SPEKTROGRAM RGB
# ==============================================================

class SpectrogramProcessor:
    @staticmethod
    def normalize_spec(spec):
        """Normalisasi Min-Max dengan clipping persentil 2-98."""
        p_low, p_high = np.percentile(spec, [2, 98])
        if p_high - p_low < 1e-10:
            return np.zeros_like(spec, dtype=np.uint8)
        spec_clipped = np.clip(spec, p_low, p_high)
        return ((spec_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    def generate_rgb(self, segment):
        """
        Bangun citra RGB berdasarkan ZONA OTAK.
        1. Hitung STFT semua 62 kanal.
        2. Kelompokkan kanal berdasarkan zona spasial (Frontal, Central, Parietal).
        3. Rata-ratakan energi matriks STFT per zona.
        4. Susun ke dalam saluran (R, G, B).
        """
        # Kumpulkan spektrogram semua kanal
        all_specs = []
        for ch_idx in ALL_CH_IDX:
            _, _, Zxx = stft(segment[ch_idx], fs=SAMPLE_RATE,
                             nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
            # Konversi magnitude ke skala dB
            spec_db = 10 * np.log10(np.abs(Zxx) + 1e-10)
            all_specs.append(spec_db)

        # Ubah ke array numpy dengan shape (62_kanal, frekuensi, waktu)
        all_specs = np.array(all_specs)

        # Ekstraksi dan rata-rata matriks berdasarkan indeks ZONA OTAK
        spec_r = np.mean(all_specs[IDX_FRONTAL, :, :], axis=0)
        spec_g = np.mean(all_specs[IDX_CENTRAL_T, :, :], axis=0)
        spec_b = np.mean(all_specs[IDX_PARIETAL, :, :], axis=0)

        # Normalisasi dan resize masing-masing matriks zona menjadi gambar 224x224
        r_img = Image.fromarray(self.normalize_spec(spec_r)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        g_img = Image.fromarray(self.normalize_spec(spec_g)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        b_img = Image.fromarray(self.normalize_spec(spec_b)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # Gabung menjadi satu gambar RGB utuh
        rgb = Image.merge('RGB', [r_img, g_img, b_img])
        return rgb

# ==============================================================
# 3. OFFLINE PREPROCESSING
# ==============================================================

def run_offline_preprocessing(data_dir, output_dir, label_path):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    df_label = pd.read_csv(label_path, sep=';')
    y_labels = (df_label['label'].values + 1).tolist()
    
    proc = SpectrogramProcessor()
    
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat') and '_' in f])
    
    print(f"[INFO] Memulai Preprocessing Sinyal EEG ke Gambar PNG (Mode ZONA OTAK)...")
    for f in tqdm(mat_files):
        subj_id = f.split('_')[0]
        sesi_id = f.split('_')[1].split('.')[0]
        
        mat = loadmat(os.path.join(data_dir, f))
        eeg_keys = sorted([k for k in mat.keys() if 'eeg' in k.lower()], 
                          key=lambda x: int(re.findall(r'\d+', x)[-1]))
        
        for t_idx, key in enumerate(eeg_keys):
            eeg_signal = mat[key]
            label = y_labels[t_idx]
            
            total_pts = eeg_signal.shape[1]
            n_segments = min(total_pts // SAMPLE_LEN, MAX_SEGMENTS)
            
            for s_idx in range(n_segments):
                seg_data = eeg_signal[:, s_idx*SAMPLE_LEN : (s_idx+1)*SAMPLE_LEN]
                img = proc.generate_rgb(seg_data)
                
                fname = f"subj{subj_id}_sesi{sesi_id}_trial{t_idx+1:02d}_seg{s_idx:02d}_label{label}.png"
                img.save(os.path.join(output_dir, fname))

# ==============================================================
# 4. DATASET CLASS
# ==============================================================

class SEEDDataset(Dataset):
    def __init__(self, img_dir, subject_ids=None, transform=None):
        self.img_dir = img_dir
        self.target_subs = [str(sid).lstrip('0') for sid in subject_ids] if subject_ids else None
        
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.file_list = self._scan_files()
        print(f"[DATASET] Subjek: {self.target_subs if self.target_subs else 'ALL'} | Gambar: {len(self.file_list)}")

    def _scan_files(self):
        valid_samples = []
        for f in os.listdir(self.img_dir):
            if f.endswith('.png'):
                sid = f.split('_')[0].replace('subj', '').lstrip('0')
                if self.target_subs is None or sid in self.target_subs:
                    label = int(f.split('_label')[-1].split('.')[0])
                    valid_samples.append((os.path.join(self.img_dir, f), label))
        return valid_samples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), torch.tensor(label, dtype=torch.long)

# ==============================================================
# ENTRY POINT: EKSEKUSI PREPROCESS
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline Preprocessing EEG ke Spektrogram RGB (Berdasarkan Zona Otak)')
    parser.add_argument('--data-dir', type=str, default="/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG")
    parser.add_argument('--label-path', type=str, default="/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv")
    parser.add_argument('--output-dir', type=str, default="/kaggle/working/spectrogram_images")
    
    args = parser.parse_args()
    
    run_offline_preprocessing(args.data_dir, args.output_dir, args.label_path)