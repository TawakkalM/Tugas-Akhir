import os
import re
import random
import numpy as np
import pandas as pd
import torch
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
# [V2] Pemetaan RGB berdasarkan PITA FREKUENSI otak
# (bukan zona otak seperti sebelumnya)
#
# Dasar ilmiah: Zheng & Lu (2015) — paper dataset SEED — membuktikan
# bahwa pita beta dan gamma di area frontal paling diskriminatif
# untuk membedakan emosi positif, netral, dan negatif.
#
# Channel R = Theta + Alpha (4–13 Hz) — emosi, memori kerja
# Channel G = Beta           (13–30 Hz) — kognisi, regulasi emosi
# Channel B = Gamma          (30–45 Hz) — pemrosesan tingkat tinggi
#
# Seluruh 62 kanal digunakan untuk SEMUA saluran warna,
# namun dikompres berbeda berdasarkan pita frekuensi.
# ------------------------------------------------------------------

# Rentang frekuensi (Hz) per saluran warna
FREQ_THETA_ALPHA = (4,  13)   # R — Theta + Alpha
FREQ_BETA        = (13, 30)   # G — Beta
FREQ_GAMMA       = (30, 45)   # B — Gamma (batas atas 45 Hz, cukup untuk EEG emosi)

# Semua 62 kanal digunakan untuk ketiga saluran
SEED_62CH = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
    'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7',
    'P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2'
]

ALL_CH_IDX = list(range(len(SEED_62CH)))  # indeks 0–61

# ==============================================================
# 2. LOGIKA PEMROSESAN SPEKTROGRAM RGB
# ==============================================================

class SpectrogramProcessor:
    """Kelas untuk mentransformasi segmen sinyal menjadi citra spektrogram RGB."""

    @staticmethod
    def compute_stft_magnitude(channel_data):
        """Menghitung STFT Magnitude dengan skala logaritmik (dB)."""
        _, _, Zxx = stft(channel_data, fs=SAMPLE_RATE,
                         nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
        magnitude = np.abs(Zxx)
        return 10 * np.log10(magnitude + 1e-10)

    @staticmethod
    def normalize_spec(spec):
        """Normalisasi Min-Max dengan clipping persentil 2-98."""
        p_low, p_high = np.percentile(spec, [2, 98])
        if p_high - p_low < 1e-10:
            return np.zeros_like(spec, dtype=np.uint8)
        spec_clipped = np.clip(spec, p_low, p_high)
        return ((spec_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    @staticmethod
    def extract_freq_band(spec_db, freqs, freq_range):
        """
        Ekstrak dan rata-ratakan baris frekuensi dalam rentang tertentu.
        spec_db : (n_freq, n_time)
        freqs   : array frekuensi dari STFT
        freq_range : (f_low, f_high) dalam Hz
        """
        mask = (freqs >= freq_range[0]) & (freqs < freq_range[1])
        if not np.any(mask):
            return np.zeros(spec_db.shape[1])
        return np.mean(spec_db[mask, :], axis=0)   # rata-rata baris → (n_time,)

    def generate_rgb(self, segment):
        """
        [V2] Bangun citra RGB berdasarkan PITA FREKUENSI dari seluruh 62 kanal.

        Proses:
        1. Hitung STFT semua 62 kanal
        2. Untuk setiap kanal, ekstrak energi per pita frekuensi
        3. Rata-ratakan energi tiap pita dari semua kanal → 3 spektrogram 1D per waktu
        4. Susun sebagai gambar RGB 2D (frekuensi sudah terkompres ke sumbu warna)

        Output: PIL Image (224×224, mode RGB)
        """
        # Hitung STFT untuk semua 62 kanal sekaligus
        # freqs: array frekuensi, shape (n_freq,)
        freqs, _, Zxx_first = stft(segment[0], fs=SAMPLE_RATE,
                                   nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)

        # Kumpulkan spektrogram semua kanal
        all_specs = []
        for ch_idx in ALL_CH_IDX:
            _, _, Zxx = stft(segment[ch_idx], fs=SAMPLE_RATE,
                             nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
            spec_db = 10 * np.log10(np.abs(Zxx) + 1e-10)  # (n_freq, n_time)
            all_specs.append(spec_db)

        # all_specs: (62, n_freq, n_time)
        all_specs = np.array(all_specs)

        # Rata-rata energi per pita frekuensi dari semua 62 kanal
        # Hasil: (n_freq, n_time) rata-rata semua kanal
        mean_spec = np.mean(all_specs, axis=0)  # (n_freq, n_time)

        # Masking per pita frekuensi
        mask_r = (freqs >= FREQ_THETA_ALPHA[0]) & (freqs < FREQ_THETA_ALPHA[1])
        mask_g = (freqs >= FREQ_BETA[0])         & (freqs < FREQ_BETA[1])
        mask_b = (freqs >= FREQ_GAMMA[0])         & (freqs < FREQ_GAMMA[1])

        # Ambil baris frekuensi yang sesuai tiap pita
        spec_r = mean_spec[mask_r, :]   # (n_theta_alpha_bins, n_time)
        spec_g = mean_spec[mask_g, :]   # (n_beta_bins, n_time)
        spec_b = mean_spec[mask_b, :]   # (n_gamma_bins, n_time)

        # Normalisasi dan resize masing-masing menjadi gambar 224×224
        r_img = Image.fromarray(self.normalize_spec(spec_r)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        g_img = Image.fromarray(self.normalize_spec(spec_g)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        b_img = Image.fromarray(self.normalize_spec(spec_b)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # Gabung menjadi satu gambar RGB
        rgb = Image.merge('RGB', [r_img, g_img, b_img])
        return rgb

# ==============================================================
# 3. OFFLINE PREPROCESSING ( PENGGANTI PREPROCESS.PY )
# ==============================================================

def run_offline_preprocessing(data_dir, output_dir, label_path):
    """Fungsi utama untuk menghasilkan dataset gambar fisik di disk."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Mapping label: -1, 0, 1 -> 0 (Neg), 1 (Net), 2 (Pos)
    df_label = pd.read_csv(label_path, sep=';')
    y_labels = (df_label['label'].values + 1).tolist()
    
    proc = SpectrogramProcessor()
    
    # PERBAIKAN ERROR INDEX: Filter hanya file yang memiliki underscore (mengabaikan label.mat)
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat') and '_' in f])
    
    print(f"[INFO] Memulai Preprocessing Sinyal EEG ke Gambar PNG...")
    for f in tqdm(mat_files):
        subj_id = f.split('_')[0]
        # Deteksi sesi dari nama file (misal: 1_20131027.mat)
        sesi_id = f.split('_')[1].split('.')[0]
        
        mat = loadmat(os.path.join(data_dir, f))
        # Ambil key eeg1 - eeg15 secara urut
        eeg_keys = sorted([k for k in mat.keys() if 'eeg' in k.lower()], 
                          key=lambda x: int(re.findall(r'\d+', x)[-1]))
        
        for t_idx, key in enumerate(eeg_keys):
            eeg_signal = mat[key]
            label = y_labels[t_idx]
            
            # Batasi panjang segmen (Truncation)
            total_pts = eeg_signal.shape[1]
            n_segments = min(total_pts // SAMPLE_LEN, MAX_SEGMENTS)
            
            for s_idx in range(n_segments):
                seg_data = eeg_signal[:, s_idx*SAMPLE_LEN : (s_idx+1)*SAMPLE_LEN]
                img = proc.generate_rgb(seg_data)
                
                # Penamaan Unik: subj{id}_sesi{tgl}_trial{id}_seg{no}_label{id}.png
                fname = f"subj{subj_id}_sesi{sesi_id}_trial{t_idx+1:02d}_seg{s_idx:02d}_label{label}.png"
                img.save(os.path.join(output_dir, fname))

# ==============================================================
# 4. DATASET CLASS ( MURNI MODE GAMBAR )
# ==============================================================

class SEEDDataset(Dataset):
    """Dataset yang hanya membaca file PNG hasil precompute."""
    
    def __init__(self, img_dir, subject_ids=None, transform=None):
        self.img_dir = img_dir
        # Subject_ids dalam format list string, misal: ['1', '2', '3']
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
                # Ekstrak ID Subjek dari format nama 'subj01_...' -> '1'
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
    # Sesuaikan path direktori sesuai lingkungan kerja (Local/Kaggle)
    BASE_DIR   = "/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG"
    LABEL_PATH = "/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv"
    OUTPUT_DIR = "/kaggle/working/spectrogram_images"
    
    # Jalankan pembuatan gambar (Hanya perlu dilakukan satu kali)
    run_offline_preprocessing(BASE_DIR, OUTPUT_DIR, LABEL_PATH)