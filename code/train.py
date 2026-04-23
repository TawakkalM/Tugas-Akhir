import os
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import wandb
from tqdm import tqdm

from data_reader import EEGDataset, get_subject_ids, subject_wise_kfold
from models import build_resnet18, BaselineCNN2D, print_model_summary

# ==============================================================
# KONFIGURASI GLOBAL
# ==============================================================
LABEL_NAMES  = ['Negatif', 'Netral', 'Positif']
NUM_CLASSES  = 3


# ==============================================================
# FUNGSI UTILITAS WANDB
# ==============================================================

def init_wandb(args, fold=None, run_type='fold'):
    """
    Inisialisasi satu run wandb.

    run_type:
      - 'fold'    : run per fold (nama: resnet18_fold1, dst.)
      - 'summary' : run ringkasan akhir semua fold
    """
    if args.wandb_off:
        return None

    if run_type == 'fold':
        run_name = f"{args.model}_fold{fold+1}"
    else:
        run_name = f"{args.model}_summary"

    run = wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity if args.wandb_entity else None,
        name    = run_name,
        group   = args.model,
        job_type= run_type,
        config  = {
            "model"        : args.model,
            "fold"         : fold + 1 if fold is not None else "all",
            "n_splits"     : args.n_splits,
            "epochs"       : args.epochs,
            "batch_size"   : args.batch_size,
            "lr"           : args.lr,
            "weight_decay" : args.weight_decay,
            "patience"     : args.patience,
            "data_dir"     : args.data_dir,
            "num_classes"  : NUM_CLASSES,
        },
        reinit = True,
    )
    return run


def log_epoch(run, epoch, train_loss, val_loss, val_acc, val_f1):
    """Log metrik per epoch ke wandb."""
    if run is None:
        return
    wandb.log({
        "epoch"      : epoch,
        "train/loss" : train_loss,
        "val/loss"   : val_loss,
        "val/acc"    : val_acc,
        "val/f1"     : val_f1,
    }, step=epoch)


def log_fold_result(run, fold, acc, precision, recall, f1, cm_path):
    """Log hasil akhir satu fold ke wandb."""
    if run is None:
        return
    wandb.summary[f"fold{fold+1}/accuracy"]  = acc
    wandb.summary[f"fold{fold+1}/precision"] = precision
    wandb.summary[f"fold{fold+1}/recall"]    = recall
    wandb.summary[f"fold{fold+1}/f1"]        = f1
    wandb.log({
        f"confusion_matrix/fold{fold+1}": wandb.Image(cm_path)
    })


def log_summary(run, all_acc, all_prec, all_rec, all_f1, avg_cm_path):
    """Log ringkasan rata-rata seluruh fold ke wandb."""
    if run is None:
        return
    wandb.log({
        "summary/mean_accuracy"       : np.mean(all_acc),
        "summary/mean_precision"      : np.mean(all_prec),
        "summary/mean_recall"         : np.mean(all_rec),
        "summary/mean_f1"             : np.mean(all_f1),
        "summary/std_accuracy"        : np.std(all_acc),
        "summary/std_f1"              : np.std(all_f1),
        "summary/avg_confusion_matrix": wandb.Image(avg_cm_path),
    })
    wandb.summary["mean_accuracy"]  = np.mean(all_acc)
    wandb.summary["mean_precision"] = np.mean(all_prec)
    wandb.summary["mean_recall"]    = np.mean(all_rec)
    wandb.summary["mean_f1"]        = np.mean(all_f1)


# ==============================================================
# FUNGSI UTILITAS VISUALISASI
# ==============================================================

def plot_training_history(fold, epochs_ran, train_losses, val_losses,
                          val_accs, save_dir):
    """
    Simpan grafik training loss, validation loss, dan validation accuracy
    untuk satu fold.
    """
    ep = range(1, epochs_ran + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    axes[0].plot(ep, train_losses, 'b-o', label='Train Loss')
    axes[0].plot(ep, val_losses,   'r-o', label='Val Loss')
    axes[0].set_title(f'Fold {fold+1} — Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Accuracy ---
    axes[1].plot(ep, val_accs, 'g-o', label='Val Accuracy')
    axes[1].set_title(f'Fold {fold+1} — Validation Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(save_dir, f'fold{fold+1}_history.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  [Plot] History disimpan → {path}")


def plot_confusion_matrix(fold, cm, save_dir, model_name):
    """
    Simpan visualisasi confusion matrix untuk satu fold.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f'{model_name} — Fold {fold+1} Confusion Matrix')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    plt.tight_layout()
    path = os.path.join(save_dir, f'fold{fold+1}_confusion_matrix.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  [Plot] Confusion matrix disimpan → {path}")


def plot_average_confusion_matrix(avg_cm, save_dir, model_name):
    """
    Simpan confusion matrix rata-rata dari seluruh fold.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f'{model_name} — Rata-rata Confusion Matrix (5 Fold)')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    plt.tight_layout()
    path = os.path.join(save_dir, 'avg_confusion_matrix.png')
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  [Plot] Avg confusion matrix disimpan → {path}")


# ==============================================================
# FUNGSI EVALUASI LENGKAP
# ==============================================================

def evaluate(model, loader, criterion, device):
    """
    Jalankan evaluasi pada satu DataLoader.
    Kembalikan: loss, accuracy, precision, recall, f1, confusion matrix.
    """
    model.eval()
    total_loss = 0.0
    total_n    = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_n    += images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_n
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))

    precision = precision_score(all_labels, all_preds,
                                average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds,
                             average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds,
                         average='macro', zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds,
                                 labels=list(range(NUM_CLASSES)))

    return avg_loss, acc, precision, recall, f1, cm


# ==============================================================
# FUNGSI TRAINING SATU FOLD
# ==============================================================

def train_one_fold(fold, model, train_loader, val_loader,
                   args, device, save_dir, run=None):
    """
    Latih model untuk satu fold.
    Menerapkan:
    - Cross-Entropy Loss
    - Adam optimizer (lr=1e-4, weight_decay=1e-4)
    - Early stopping (patience=5) berdasarkan val loss
    - Simpan best model berdasarkan val loss terbaik
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_loss  = float('inf')
    patience_count = 0
    best_model_path = os.path.join(save_dir, f'fold{fold+1}_best_model.pth')

    train_losses, val_losses, val_accs = [], [], []

    print(f"\n{'='*60}")
    print(f"  FOLD {fold+1} — Mulai Training")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):

        # --- Training phase ---
        model.train()
        running_loss = 0.0
        running_n    = 0

        # Implementasi Progress Bar (tqdm)
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch:02d}/{args.epochs} [Train]", leave=False, ncols=90, unit="batch")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_n    += images.size(0)

            # Update tampilan loss di progress bar
            pbar.set_postfix(loss=f"{running_loss / running_n:.4f}")

        train_loss = running_loss / running_n

        # --- Validation phase ---
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"  Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"F1: {val_f1:.4f}")

        # --- Log ke wandb per epoch ---
        log_epoch(run, epoch, train_loss, val_loss, val_acc, val_f1)

        # --- Simpan model terbaik ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"           → Best model disimpan (val loss: {best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"           → Tidak ada perbaikan. "
                  f"Patience: {patience_count}/{args.patience}")

        # --- Early stopping ---
        if patience_count >= args.patience:
            print(f"\n  [Early Stopping] Val loss tidak membaik "
                  f"selama {args.patience} epoch. Training dihentikan.")
            break

    epochs_ran = len(train_losses)

    # --- Load best model untuk evaluasi akhir fold ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # --- Evaluasi akhir dengan metrik lengkap ---
    _, acc, precision, recall, f1, cm = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\n  Hasil Akhir Fold {fold+1}:")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")

    # --- Simpan visualisasi ---
    plot_training_history(fold, epochs_ran, train_losses,
                          val_losses, val_accs, save_dir)
    plot_confusion_matrix(fold, cm, save_dir, args.model)

    # --- Log hasil akhir fold ke wandb ---
    cm_path = os.path.join(save_dir, f'fold{fold+1}_confusion_matrix.png')
    log_fold_result(run, fold, acc, precision, recall, f1, cm_path)
    if run is not None:
        run.finish()

    return acc, precision, recall, f1, cm


# ==============================================================
# FUNGSI UTAMA — LOOP 5-FOLD
# ==============================================================

def run_all_folds(args):
    """
    Jalankan training dan evaluasi untuk semua 5 fold secara otomatis.
    Di akhir, cetak rata-rata metrik dari seluruh fold.
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    )
    print(f"\nDevice: {device}")

    # Buat folder output
    save_dir = os.path.join(args.checkpoint_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # Ambil pembagian fold
    subject_ids = get_subject_ids(args.data_dir)
    folds       = subject_wise_kfold(subject_ids, n_splits=args.n_splits)

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    all_cm = []

    # Tentukan fold mana saja yang akan dijalankan
    if args.fold_only is not None:
        # Validasi nilai --fold-only
        if not (0 <= args.fold_only < args.n_splits):
            raise ValueError(
                f"--fold-only harus antara 0 dan {args.n_splits - 1}, "
                f"bukan {args.fold_only}"
            )
        fold_range = [args.fold_only]
        print(f"[INFO] Mode --fold-only: hanya menjalankan Fold {args.fold_only + 1}")
    else:
        fold_range = range(args.n_splits)
        print(f"[INFO] Menjalankan semua {args.n_splits} fold")

    for fold in fold_range:
        train_subs, val_subs = folds[fold]

        # Dataset & DataLoader
        train_ds = EEGDataset(args.data_dir, args.label_path,
                              fold=fold, split='train',
                              n_splits=args.n_splits)
        val_ds   = EEGDataset(args.data_dir, args.label_path,
                              fold=fold, split='val',
                              n_splits=args.n_splits)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=device.type == 'cuda')
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=device.type == 'cuda')

        # Bangun model baru tiap fold
        if args.model == 'resnet18':
            model = build_resnet18(num_classes=NUM_CLASSES,
                                   pretrained=True)
        else:
            model = BaselineCNN2D(num_classes=NUM_CLASSES)
        model.to(device)

        # Cetak ringkasan model hanya di fold pertama
        if fold == 0:
            print_model_summary(model, args.model.upper())

        # Inisialisasi wandb untuk fold ini
        run = init_wandb(args, fold=fold, run_type='fold')

        # Training satu fold
        acc, prec, rec, f1, cm = train_one_fold(
            fold, model, train_loader, val_loader, args, device, save_dir,
            run=run
        )

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)
        all_cm.append(cm)

    # ==== Ringkasan Akhir Seluruh Fold ====
    avg_cm = np.mean(all_cm, axis=0)

    print(f"\n{'='*60}")
    if args.fold_only is not None:
        print(f"  HASIL — {args.model.upper()} — Fold {args.fold_only + 1} Saja")
    else:
        print(f"  HASIL AKHIR — {args.model.upper()} — 5-Fold CV")
    print(f"{'='*60}")
    print(f"  {'Fold':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*44}")
    for i, fold_idx in enumerate(fold_range):
        print(f"  Fold {fold_idx+1:<4} "
              f"{all_acc[i]:>8.4f} "
              f"{all_prec[i]:>8.4f} "
              f"{all_rec[i]:>8.4f} "
              f"{all_f1[i]:>8.4f}")

    # Hanya tampilkan rata-rata jika lebih dari 1 fold
    if len(all_acc) > 1:
        print(f"  {'-'*44}")
        print(f"  {'Rata-rata':<8} "
              f"{np.mean(all_acc):>8.4f} "
              f"{np.mean(all_prec):>8.4f} "
              f"{np.mean(all_rec):>8.4f} "
              f"{np.mean(all_f1):>8.4f}")
        print(f"  {'Std Dev':<8} "
              f"{np.std(all_acc):>8.4f} "
              f"{np.std(all_prec):>8.4f} "
              f"{np.std(all_rec):>8.4f} "
              f"{np.std(all_f1):>8.4f}")
    print(f"{'='*60}\n")

    # Simpan confusion matrix rata-rata
    plot_average_confusion_matrix(avg_cm, save_dir, args.model.upper())

    # --- Log ringkasan ke wandb (run terpisah khusus summary) ---
    avg_cm_path = os.path.join(save_dir, 'avg_confusion_matrix.png')
    summary_run = init_wandb(args, fold=None, run_type='summary')
    log_summary(summary_run, all_acc, all_prec, all_rec, all_f1, avg_cm_path)
    if summary_run is not None:
        summary_run.finish()

    # Simpan ringkasan ke file teks
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"{'='*44}\n")
        f.write(f"{'Fold':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}\n")
        
        # Penyesuaian agar tidak error index saat mode --fold-only
        for i, fold_idx in enumerate(fold_range):
            f.write(f"Fold {fold_idx+1:<4} "
                    f"{all_acc[i]:>8.4f} "
                    f"{all_prec[i]:>8.4f} "
                    f"{all_rec[i]:>8.4f} "
                    f"{all_f1[i]:>8.4f}\n")
        
        if len(all_acc) > 1:
            f.write(f"{'='*44}\n")
            f.write(f"{'Mean':<8} "
                    f"{np.mean(all_acc):>8.4f} "
                    f"{np.mean(all_prec):>8.4f} "
                    f"{np.mean(all_rec):>8.4f} "
                    f"{np.mean(all_f1):>8.4f}\n")
            f.write(f"{'Std':<8} "
                    f"{np.std(all_acc):>8.4f} "
                    f"{np.std(all_prec):>8.4f} "
                    f"{np.std(all_rec):>8.4f} "
                    f"{np.std(all_f1):>8.4f}\n")
    print(f"  [Summary] Disimpan → {summary_path}")


# ==============================================================
# ARGUMENT PARSER & ENTRY POINT
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training ResNet-18 / Baseline CNN untuk klasifikasi emosi EEG'
    )

    # Data
    parser.add_argument('--data-dir',    type=str,   default='/kaggle/input/datasets/tawakkal19/eeg-seed-200hz/Preprocessed_EEG')
    parser.add_argument('--label-path',  type=str,   default='/kaggle/input/datasets/tawakkal19/kode-eeg/label.csv')

    # Model
    parser.add_argument('--model',       type=str,   default='resnet18',
                        choices=['resnet18', 'baseline'],
                        help='Model yang digunakan: resnet18 atau baseline')

    # Training
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--weight-decay',type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=5,
                        help='Early stopping patience')

    # Cross validation
    parser.add_argument('--n-splits',    type=int,   default=5)
    parser.add_argument('--fold-only',   type=int,   default=None,
                        help='Jalankan hanya 1 fold tertentu saja. '
                             'Contoh: --fold-only 0 untuk fold pertama. '
                             'Jika tidak diisi, semua fold dijalankan.')

    # System
    parser.add_argument('--num-workers', type=int,   default=2)
    parser.add_argument('--no-cuda',     action='store_true')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # Wandb
    parser.add_argument('--wandb-project', type=str,
                        default='eeg-emotion-classification',
                        help='Nama project di wandb')
    parser.add_argument('--wandb-entity',  type=str, default=None,
                        help='Username atau nama tim di wandb (opsional)')
    parser.add_argument('--wandb-off',     action='store_true',
                        help='Matikan wandb (mode offline/tanpa logging)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Catat waktu mulai
    start_time_sec = time.time()
    start_date_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"  PROSES TRAINING DIMULAI PADA: {start_date_str}")
    print(f"{'='*60}\n")

    run_all_folds(args)

    # Catat waktu selesai & hitung durasi
    end_time_sec = time.time()
    end_date_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    elapsed          = end_time_sec - start_time_sec
    hours, rem       = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n{'='*60}")
    print(f"  TRAINING SELESAI PADA: {end_date_str}")
    print(f"  TOTAL WAKTU EKSEKUSI : {int(hours):02d} Jam, {int(minutes):02d} Menit, {int(seconds):02d} Detik")
    print(f"{'='*60}\n")