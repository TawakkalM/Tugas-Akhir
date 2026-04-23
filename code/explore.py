"""
explore.py
==========
File eksplorasi data sebelum training.
Tujuan: memahami karakteristik data EEG dan kualitas
spektrogram RGB yang dihasilkan pipeline data_reader.py.

Output yang dihasilkan (disimpan ke folder SAVE_DIR):
  1. raw_signals_per_class.png     — sinyal EEG mentah per kelas emosi
  2. spectrogram_per_class.png     — spektrogram RGB 3 segmen per kelas
  3. spectrogram_channels.png      — spektrogram tiap kanal R/G/B terpisah
  4. segment_distribution.png      — distribusi jumlah segmen per trial
  5. dataset_summary.txt           — ringkasan statistik dataset
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat

# Import fungsi dari data_reader yang sudah ada
from data_reader import (
    read_labels,
    list_mat_files,
    detect_eeg_keys,
    segment_eeg,
    compute_stft_spectrogram,
    zone_average_spectrogram,
    normalize_spectrogram,
    segment_to_rgb_image,
    SAMPLE_RATE,
    WINDOW_SEC,
    SAMPLE_LEN,
    IDX_FRONTAL,
    IDX_CENTRAL_T,
    IDX_PARIETAL,
    SEED_62CH,
)

# ==============================================================
# KONFIGURASI PATH — sesuaikan dengan path Kaggle Anda
# ==============================================================
DATA_DIR   = "/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG"
LABEL_PATH = "/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv"
SAVE_DIR   = "/kaggle/working/explore_output"

# Nama kelas emosi
LABEL_NAMES  = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
LABEL_COLORS = {0: '#E74C3C', 1: '#95A5A6', 2: '#2ECC71'}

# Seed untuk reproduktibilitas
RANDOM_SEED = 1907
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================================================
# FUNGSI UTILITAS
# ==============================================================

def load_all_trials(data_dir, label_path):
    """
    Muat semua trial dari semua file .mat.
    Kembalikan list of dict berisi informasi tiap trial.
    """
    labels    = read_labels(label_path)
    mat_files = list_mat_files(data_dir)
    trials    = []

    for mat_file in mat_files:
        path    = os.path.join(data_dir, mat_file)
        subj_id = mat_file.split('_')[0]
        mat     = loadmat(path)
        keys    = detect_eeg_keys(mat)

        for i, key in enumerate(keys):
            eeg   = mat[key]               # (62, N)
            label = labels[i] if i < len(labels) else None
            trials.append({
                'subj_id' : subj_id,
                'path'    : path,
                'key'     : key,
                'label'   : label,
                'eeg'     : eeg,
                'n_samples': eeg.shape[1],
                'duration' : eeg.shape[1] / SAMPLE_RATE,
                'n_segments': eeg.shape[1] // SAMPLE_LEN,
            })

    return trials


def pick_one_trial_per_class(trials):
    """
    Pilih 1 trial secara deterministik untuk tiap kelas emosi (0, 1, 2).
    Kembalikan dict {label: trial}.
    """
    result = {}
    for label in [0, 1, 2]:
        candidates = [t for t in trials if t['label'] == label]
        # Ambil trial ke-3 dari tiap kelas agar lebih representatif
        result[label] = candidates[2]
    return result


# ==============================================================
# PLOT 1 — SINYAL EEG MENTAH PER KELAS EMOSI
# ==============================================================

def plot_raw_signals(trials_per_class, save_dir):
    """
    Plot sinyal EEG mentah dari 5 kanal representatif
    untuk masing-masing kelas emosi.
    Tujuan: melihat apakah ada perbedaan visual antar kelas.
    """
    # 5 kanal yang dipilih: FP1 (Frontal), CZ (Central),
    # T7 (Temporal), PZ (Parietal), OZ (Occipital)
    ch_names  = ['FP1', 'CZ', 'T7', 'PZ', 'OZ']
    ch_indices= [SEED_62CH.index(ch) for ch in ch_names]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Sinyal EEG Mentah — 5 Kanal Representatif per Kelas Emosi',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, label in enumerate([0, 1, 2]):
        trial   = trials_per_class[label]
        eeg     = trial['eeg']                # (62, N)
        # Ambil segmen pertama saja (5 detik pertama)
        seg     = eeg[:, :SAMPLE_LEN]
        time_ax = np.arange(SAMPLE_LEN) / SAMPLE_RATE

        ax = axes[row]
        ax.set_title(
            f"Kelas: {LABEL_NAMES[label]} | "
            f"Subjek {trial['subj_id']} | "
            f"Trial {trial['key']}",
            fontsize=11, color=LABEL_COLORS[label]
        )

        for j, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_names)):
            offset = j * 150   # offset vertikal antar kanal
            ax.plot(time_ax,
                    seg[ch_idx] + offset,
                    label=ch_name,
                    linewidth=0.8)

        ax.set_xlabel('Waktu (detik)')
        ax.set_ylabel('Amplitudo (µV) + offset')
        ax.set_xlim(0, WINDOW_SEC)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, 'raw_signals_per_class.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Disimpan → {path}")


# ==============================================================
# PLOT 2 — SPEKTROGRAM RGB PER KELAS EMOSI (3 SEGMEN)
# ==============================================================

def plot_spectrograms_per_class(trials_per_class, save_dir):
    """
    Untuk setiap kelas emosi, ambil 3 segmen berbeda dari trial
    yang sama, lalu tampilkan spektrogram RGB-nya.

    Layout: 3 baris (kelas) × 3 kolom (segmen)
    Tujuan: melihat konsistensi pola spektrogram dalam satu kelas
            dan perbedaan antar kelas.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle('Spektrogram RGB per Kelas Emosi (3 Segmen Berbeda)',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate([0, 1, 2]):
        trial    = trials_per_class[label]
        eeg      = trial['eeg']
        segments = segment_eeg(eeg)

        # Ambil segmen ke-0, ke-tengah, ke-akhir
        n_seg    = len(segments)
        seg_pick = [
            segments[0],
            segments[n_seg // 2],
            segments[-1]
        ]
        seg_labels = ['Segmen Awal', 'Segmen Tengah', 'Segmen Akhir']

        for col, (seg, seg_lbl) in enumerate(zip(seg_pick, seg_labels)):
            tensor = segment_to_rgb_image(seg)
            img_np = tensor.permute(1, 2, 0).numpy()

            ax = axes[row][col]
            ax.imshow(img_np, aspect='auto')
            ax.axis('off')

            title_color = LABEL_COLORS[label]
            if col == 0:
                ax.set_title(
                    f"{LABEL_NAMES[label]}\n{seg_lbl}",
                    fontsize=10, color=title_color, fontweight='bold'
                )
            else:
                ax.set_title(seg_lbl, fontsize=10, color=title_color)

    # Tambahkan label zona warna di bagian bawah
    fig.text(0.5, -0.01,
             'R (Merah) = Frontal  |  G (Hijau) = Central & Temporal  '
             '|  B (Biru) = Parietal & Occipital',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    path = os.path.join(save_dir, 'spectrogram_per_class.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Disimpan → {path}")


# ==============================================================
# PLOT 3 — SPEKTROGRAM TIAP KANAL R/G/B SECARA TERPISAH
# ==============================================================

def plot_spectrogram_channels(trials_per_class, save_dir):
    """
    Tampilkan kanal R, G, B secara terpisah (grayscale) untuk
    1 segmen dari masing-masing kelas emosi.

    Layout: 3 baris (kelas) × 3 kolom (R, G, B)
    Tujuan: melihat kontribusi tiap zona otak secara individual
            dan memastikan ketiga kanal sudah berisi informasi.
    """
    zone_names  = ['R — Frontal', 'G — Central & Temporal',
                   'B — Parietal & Occipital']
    zone_indices= [IDX_FRONTAL, IDX_CENTRAL_T, IDX_PARIETAL]
    cmaps       = ['Reds', 'Greens', 'Blues']

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('Spektrogram per Zona Otak (R / G / B) per Kelas Emosi',
                 fontsize=14, fontweight='bold')

    for row, label in enumerate([0, 1, 2]):
        trial    = trials_per_class[label]
        eeg      = trial['eeg']
        segments = segment_eeg(eeg)
        seg      = segments[len(segments) // 2]   # ambil segmen tengah

        for col, (z_name, z_idx, cmap) in enumerate(
                zip(zone_names, zone_indices, cmaps)):

            spec = zone_average_spectrogram(seg, z_idx)
            norm = normalize_spectrogram(spec)

            ax = axes[row][col]
            im = ax.imshow(norm, aspect='auto', origin='lower',
                           cmap=cmap,
                           extent=[0, WINDOW_SEC, 0, SAMPLE_RATE // 2])
            ax.set_xlabel('Waktu (s)', fontsize=8)
            ax.set_ylabel('Frekuensi (Hz)', fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(z_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(
                    f"{LABEL_NAMES[label]}\nFrekuensi (Hz)",
                    fontsize=9, color=LABEL_COLORS[label]
                )

    plt.tight_layout()
    path = os.path.join(save_dir, 'spectrogram_channels.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Disimpan → {path}")


# ==============================================================
# PLOT 4 — DISTRIBUSI JUMLAH SEGMEN PER TRIAL
# ==============================================================

def plot_segment_distribution(trials, save_dir):
    """
    Histogram jumlah segmen yang dihasilkan tiap trial,
    dikelompokkan per kelas emosi.
    Tujuan: memastikan distribusi segmen seimbang antar kelas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribusi Segmen Dataset SEED', fontsize=13,
                 fontweight='bold')

    # --- Kiri: histogram jumlah segmen per trial per kelas ---
    ax = axes[0]
    for label in [0, 1, 2]:
        segs = [t['n_segments'] for t in trials if t['label'] == label]
        ax.hist(segs, bins=15, alpha=0.6,
                label=LABEL_NAMES[label],
                color=LABEL_COLORS[label])
    ax.set_title('Distribusi Jumlah Segmen per Trial')
    ax.set_xlabel('Jumlah Segmen')
    ax.set_ylabel('Jumlah Trial')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # --- Kanan: bar chart total segmen per kelas ---
    ax2 = axes[1]
    totals = {
        lbl: sum(t['n_segments'] for t in trials if t['label'] == lbl)
        for lbl in [0, 1, 2]
    }
    bars = ax2.bar(
        [LABEL_NAMES[l] for l in [0, 1, 2]],
        [totals[l] for l in [0, 1, 2]],
        color=[LABEL_COLORS[l] for l in [0, 1, 2]],
        width=0.5, edgecolor='white'
    )
    for bar, val in zip(bars, [totals[l] for l in [0, 1, 2]]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f'{val:,}', ha='center', va='bottom', fontsize=11)
    ax2.set_title('Total Segmen per Kelas Emosi')
    ax2.set_ylabel('Total Segmen')
    ax2.set_ylim(0, max(totals.values()) * 1.15)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, 'segment_distribution.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[✓] Disimpan → {path}")


# ==============================================================
# RINGKASAN STATISTIK DATASET
# ==============================================================

def print_and_save_summary(trials, save_dir):
    """
    Cetak dan simpan ringkasan statistik dataset ke file teks.
    """
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 55)
    log("  RINGKASAN STATISTIK DATASET SEED")
    log("=" * 55)

    total_trials   = len(trials)
    total_segments = sum(t['n_segments'] for t in trials)
    subj_ids       = sorted(set(t['subj_id'] for t in trials), key=int)

    log(f"  Jumlah subjek       : {len(subj_ids)}")
    log(f"  Jumlah trial total  : {total_trials}")
    log(f"  Total segmen (est.) : {total_segments:,}")
    log()

    log(f"  {'Kelas':<10} {'Trial':>6} {'Segmen':>8} "
        f"{'Dur. min':>9} {'Dur. max':>9} {'Dur. rata':>10}")
    log(f"  {'-'*55}")

    for label in [0, 1, 2]:
        subset   = [t for t in trials if t['label'] == label]
        segs     = [t['n_segments'] for t in subset]
        durs     = [t['duration'] for t in subset]
        log(f"  {LABEL_NAMES[label]:<10} "
            f"{len(subset):>6} "
            f"{sum(segs):>8,} "
            f"{min(durs):>9.1f}s "
            f"{max(durs):>9.1f}s "
            f"{np.mean(durs):>9.1f}s")

    log()
    log(f"  Durasi trial rata-rata : "
        f"{np.mean([t['duration'] for t in trials]):.1f} detik")
    log(f"  Segmen per trial rata-rata : "
        f"{np.mean([t['n_segments'] for t in trials]):.1f}")
    log(f"  Window size  : {WINDOW_SEC} detik = {SAMPLE_LEN} titik")
    log(f"  Sample rate  : {SAMPLE_RATE} Hz")
    log("=" * 55)

    # Simpan ke file
    path = os.path.join(save_dir, 'dataset_summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[✓] Summary disimpan → {path}")


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    # Buat folder output
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nOutput akan disimpan ke: {SAVE_DIR}")
    print("=" * 55)

    # --- Muat semua trial ---
    print("\n[1/5] Memuat semua trial dari dataset...")
    trials = load_all_trials(DATA_DIR, LABEL_PATH)
    print(f"      Total trial dimuat: {len(trials)}")

    # --- Pilih 1 trial per kelas ---
    trials_per_class = pick_one_trial_per_class(trials)
    for lbl, t in trials_per_class.items():
        print(f"      Kelas {LABEL_NAMES[lbl]:<8} → "
              f"Subjek {t['subj_id']}, {t['key']}, "
              f"durasi {t['duration']:.0f}s, "
              f"{t['n_segments']} segmen")

    # --- Plot 1: Sinyal mentah ---
    print("\n[2/5] Plot sinyal EEG mentah per kelas...")
    plot_raw_signals(trials_per_class, SAVE_DIR)

    # --- Plot 2: Spektrogram RGB ---
    print("\n[3/5] Plot spektrogram RGB per kelas (3 segmen)...")
    plot_spectrograms_per_class(trials_per_class, SAVE_DIR)

    # --- Plot 3: Kanal R/G/B terpisah ---
    print("\n[4/5] Plot kanal R/G/B terpisah per kelas...")
    plot_spectrogram_channels(trials_per_class, SAVE_DIR)

    # --- Plot 4: Distribusi segmen ---
    print("\n[5/5] Plot distribusi segmen dataset...")
    plot_segment_distribution(trials, SAVE_DIR)

    # --- Ringkasan statistik ---
    print()
    print_and_save_summary(trials, SAVE_DIR)

    print("\n✓ Semua eksplorasi selesai.")
    print(f"  Silakan cek folder: {SAVE_DIR}")
