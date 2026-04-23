"""
preprocess.py
=============
Script pra-pemrosesan spektrogram RGB dari sinyal EEG SEED.

Tujuan:
- Memotong SEMUA segmen per trial (rata-rata ~45 segmen per trial)
- Mengubah setiap segmen menjadi gambar spektrogram RGB 224×224
- Menyimpan hasilnya sebagai file PNG ke disk

Dengan gambar yang sudah jadi, saat training:
- __getitem__ hanya membaca PNG (sangat ringan di CPU)
- CPU tidak lagi terbebani STFT setiap batch
- GPU bisa dimanfaatkan secara maksimal

Output:
  Setiap segmen disimpan sebagai file PNG unik:
    subj{id}_sesi{tanggal}_trial{idx}_seg{nomor}_label{label}.png

  Contoh: subj01_sesi20131027_trial07_seg03_label2.png

  Keterangan nama file:
    subj  = ID subjek (zero-padded, misal 01)
    sesi  = tanggal sesi (misal 20131027)
    trial = nomor trial dalam sesi (01-15)
    seg   = nomor urut segmen dalam trial (00, 01, 02, ...)
    label = kelas emosi (0=Negatif, 1=Netral, 2=Positif)

Total file yang dihasilkan: ~675 trial × 45 segmen = ~30.375 gambar

Cara menjalankan di Kaggle:
  !python preprocess.py \\
      --data-dir  /kaggle/input/.../Preprocessed_EEG \\
      --label-path /kaggle/input/.../label.csv \\
      --output-dir /kaggle/working/spectrograms
"""

import os
import argparse
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

# Import fungsi dari data_reader
from data_reader import (
    read_labels,
    list_mat_files,
    detect_eeg_keys,
    segment_eeg,
    segment_to_rgb_image,
    SAMPLE_LEN,
)


# ==============================================================
# FUNGSI PENGAMBILAN SEMUA SEGMEN
# ==============================================================

def get_all_segments(segments):
    """
    Kembalikan semua segmen dari satu trial beserta indeksnya.
    Setiap segmen diberi nama berdasarkan posisinya: seg00, seg01, dst.

    Dengan rata-rata 45 segmen per trial dan 675 trial total,
    estimasi output adalah ~30.375 gambar.
    """
    return [(f"seg{i:02d}", seg) for i, seg in enumerate(segments)]

def get_representative_segments(segments):
    """
    Ambil 3 segmen representatif:
    - awal
    - tengah
    - akhir

    Jika segmen < 3, ambil semua.
    """
    n = len(segments)

    if n == 0:
        return []

    if n <= 3:
        return [(f"seg{i:02d}", seg) for i, seg in enumerate(segments)]

    idxs = [
        0,          # awal
        n // 2,     # tengah
        n - 1       # akhir
    ]

    return [(f"seg{i:02d}", segments[i]) for i in idxs]


def save_rgb_image(tensor, save_path):
    """
    Simpan tensor (3, 224, 224) sebagai file PNG.
    Tensor bernilai [0.0, 1.0] dikonversi ke [0, 255].
    """
    img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img    = Image.fromarray(img_np, mode='RGB')
    img.save(save_path)


# ==============================================================
# FUNGSI PRECOMPUTE UTAMA
# ==============================================================

def precompute(data_dir, label_path, output_dir):
    """
    Jalankan pra-komputasi untuk seluruh dataset.

    Untuk setiap trial:
    1. Muat sinyal EEG dari file .mat
    2. Potong semua segmen 5 detik
    3. Pilih 3 segmen representatif (awal, tengah, akhir)
    4. Transformasi setiap segmen → spektrogram RGB 224×224
    5. Simpan sebagai PNG ke output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    labels    = read_labels(label_path)
    mat_files = list_mat_files(data_dir)

    total_saved   = 0
    total_skipped = 0

    label_names = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

    print(f"\n{'='*60}")
    print(f"  PRECOMPUTE — Spektrogram RGB (Semua Segmen)")
    print(f"{'='*60}")
    print(f"  Data dir        : {data_dir}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Segmen/trial    : SEMUA (~45 segmen per trial)")
    print(f"  Estimasi total  : ~{len(mat_files) * 15 * 45:,} gambar")
    print(f"{'='*60}\n")

    # Progress bar per file .mat
    for mat_file in tqdm(mat_files, desc="File .mat", unit="file"):
        path    = os.path.join(data_dir, mat_file)
        subj_id = mat_file.split('_')[0].zfill(2)  # zero-pad: '1' → '01'
        sesi_id = mat_file.replace('.mat', '').split('_')[1]  # tanggal sesi, misal '20131027'
        mat     = loadmat(path)
        keys    = detect_eeg_keys(mat)

        for trial_idx, key in enumerate(keys):
            label = labels[trial_idx] if trial_idx < len(labels) else None
            if label is None:
                total_skipped += 1
                continue

            eeg      = mat[key]          # (62, N)
            segments = segment_eeg(eeg)  # list of (62, 1000)

            if len(segments) == 0:
                total_skipped += 1
                continue

            all_segs = get_representative_segments(segments)

            for seg_name, seg in all_segs:
                # Nama file unik per subjek, sesi, trial, dan segmen
                # Format: subj01_sesi20131027_trial07_seg03_label2.png
                fname     = (f"subj{subj_id}"
                             f"_sesi{sesi_id}"
                             f"_trial{trial_idx+1:02d}"
                             f"_{seg_name}"
                             f"_label{label}.png")
                save_path = os.path.join(output_dir, fname)

                # Lewati jika file sudah ada (resume-able)
                if os.path.exists(save_path):
                    total_saved += 1
                    continue

                # Transformasi → simpan
                tensor = segment_to_rgb_image(seg)
                save_rgb_image(tensor, save_path)
                total_saved += 1

    print(f"\n{'='*60}")
    print(f"  Selesai!")
    print(f"  Total gambar tersimpan : {total_saved:,}")
    print(f"  Total trial dilewati   : {total_skipped}")
    print(f"  Output dir             : {output_dir}")
    print(f"{'='*60}\n")

    # Cetak distribusi kelas dari file yang tersimpan
    _print_output_summary(output_dir, label_names)


def _print_output_summary(output_dir, label_names):
    """Cetak ringkasan distribusi gambar per kelas."""
    from collections import Counter
    all_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

    label_count = Counter()
    for fname in all_files:
        parts      = fname.split('_')
        label_part = parts[-1].replace('label', '').replace('.png', '')
        label_count[int(label_part)] += 1

    print("  Distribusi gambar per kelas:")
    for lbl in sorted(label_count):
        print(f"    {label_names.get(lbl, lbl):<10}: "
              f"{label_count[lbl]:,} gambar")
    print(f"    {'TOTAL':<10}: {sum(label_count.values()):,} gambar\n")


# ==============================================================
# ARGUMENT PARSER & ENTRY POINT
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pra-komputasi spektrogram RGB dari EEG SEED'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG',
        help='Folder berisi file .mat dataset SEED'
    )
    parser.add_argument(
        '--label-path', type=str,
        default='/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv',
        help='Path ke file label.csv'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/kaggle/working/spectrograms',
        help='Folder output untuk menyimpan file PNG'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    precompute(
        data_dir   = args.data_dir,
        label_path = args.label_path,
        output_dir = args.output_dir,
    )
