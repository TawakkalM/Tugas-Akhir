import os
import argparse
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['WANDB_SILENT'] = 'true'
import wandb
from tqdm import tqdm

from data_reader import SEEDDataset
from models import build_resnet18, BaselineCNN2D, print_model_summary, unfreeze_layers

# ==============================================================
# KONFIGURASI GLOBAL
# ==============================================================
LABEL_NAMES  = ['Negatif', 'Netral', 'Positif']
NUM_CLASSES  = 3

def get_wib_time():
    wib_tz = timezone(timedelta(hours=7))
    return datetime.now(wib_tz).strftime("%d-%m-%Y %H:%M:%S")

# ==============================================================
# FUNGSI UTILITAS WANDB
# ==============================================================

def init_wandb(args, fold=None, run_type='fold'):
    if args.wandb_off:
        return None

    run_name = f"{args.model}_fold{fold+1}" if run_type == 'fold' else f"{args.model}_summary"

    # Penentuan strategi otomatis untuk log WandB
    if args.model == 'resnet18' and args.pretrained:
        strategy_name = "progressive_unfreezing"
    elif args.model == 'resnet18' and not args.pretrained:
        strategy_name = "train_from_scratch"
    else:
        strategy_name = "standard"

    run = wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity if args.wandb_entity else None,
        name    = run_name,
        group   = args.model,
        job_type= run_type,
        config  = {
            "model"            : args.model,
            "pretrained"       : args.pretrained if args.model == 'resnet18' else False,
            "strategy"         : strategy_name,
            "phase2_epoch"     : args.phase2_epoch if strategy_name == "progressive_unfreezing" else None,
            "phase3_epoch"     : args.phase3_epoch if strategy_name == "progressive_unfreezing" else None,
            "lr_phase2_mult"   : args.lr_phase2_mult if strategy_name == "progressive_unfreezing" else None,
            "lr_phase3_mult"   : args.lr_phase3_mult if strategy_name == "progressive_unfreezing" else None,
            "fold"             : fold + 1 if fold is not None else "all",
            "n_splits"         : args.n_splits,
            "epochs"           : args.epochs,
            "batch_size"       : args.batch_size,
            "lr"               : args.lr,
            "weight_decay"     : args.weight_decay,
            "patience"         : args.patience,
            "img_dir"          : args.img_dir,
            "num_classes"      : NUM_CLASSES,
        },
        reinit = True,
    )
    return run

def log_epoch(run, epoch, train_loss, val_loss, val_acc, val_f1):
    if run:
        wandb.log({
            "epoch"      : epoch,
            "train/loss" : train_loss,
            "val/loss"   : val_loss,
            "val/acc"    : val_acc,
            "val/f1"     : val_f1,
        }, step=epoch)

def log_fold_result(run, fold, acc, precision, recall, f1, cm_path):
    if run:
        wandb.summary[f"fold{fold+1}/accuracy"]  = acc
        wandb.summary[f"fold{fold+1}/precision"] = precision
        wandb.summary[f"fold{fold+1}/recall"]    = recall
        wandb.summary[f"fold{fold+1}/f1"]        = f1
        wandb.log({f"confusion_matrix/fold{fold+1}": wandb.Image(cm_path)})

def log_summary(run, all_acc, all_prec, all_rec, all_f1, avg_cm_path):
    if run:
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

def plot_training_history(fold, epochs_ran, train_losses, val_losses, val_accs, save_dir):
    ep = range(1, epochs_ran + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, train_losses, 'b-o', label='Train Loss')
    axes[0].plot(ep, val_losses,   'r-o', label='Val Loss')
    axes[0].set_title(f'Fold {fold+1} — Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

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

def plot_confusion_matrix(fold, cm, save_dir, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f'{model_name} — Fold {fold+1} Confusion Matrix')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    plt.tight_layout()
    path = os.path.join(save_dir, f'fold{fold+1}_confusion_matrix.png')
    plt.savefig(path, dpi=100)
    plt.close()

def plot_average_confusion_matrix(avg_cm, save_dir, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f'{model_name} — Rata-rata Confusion Matrix (5 Fold)')
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    plt.tight_layout()
    path = os.path.join(save_dir, 'avg_confusion_matrix.png')
    plt.savefig(path, dpi=100)
    plt.close()

# ==============================================================
# FUNGSI EVALUASI LENGKAP
# ==============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_n    += images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / total_n
    acc       = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    return avg_loss, acc, precision, recall, f1, cm

# ==============================================================
# FUNGSI TRAINING SATU FOLD
# ==============================================================

def train_one_fold(fold, model, train_loader, val_loader, args, device, save_dir, run=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss   = float('inf')
    patience_count  = 0
    best_model_path = os.path.join(save_dir, f'fold{fold+1}_best_model.pth')
    train_losses, val_losses, val_accs = [], [], []

    print(f"\n{'='*60}")
    print(f"  [{get_wib_time()}] FOLD {fold+1} — Mulai Training")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):

        # Hanya lakukan unfreezing bertahap JIKA model Pretrained
        if args.model == 'resnet18' and args.pretrained:
            if epoch == args.phase2_epoch:
                print(f"\n  [Fase 2] Epoch {epoch} — membuka layer4")
                unfreeze_layers(model, phase=2)
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr * args.lr_phase2_mult,
                    weight_decay=args.weight_decay
                )

            elif epoch == args.phase3_epoch:
                print(f"\n  [Fase 3] Epoch {epoch} — membuka layer3")
                unfreeze_layers(model, phase=3)
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr * args.lr_phase3_mult,
                    weight_decay=args.weight_decay
                )
            
        model.train()
        running_loss, running_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch:02d}/{args.epochs} [Train]", leave=False, ncols=90, unit="batch")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_n    += images.size(0)
            pbar.set_postfix(loss=f"{running_loss / running_n:.4f}")

        train_loss = running_loss / running_n
        val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"  [{get_wib_time()}] Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        log_epoch(run, epoch, train_loss, val_loss, val_acc, val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"           → Best model disimpan (val loss: {best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"           → Tidak ada perbaikan. Patience: {patience_count}/{args.patience}")

        if patience_count >= args.patience:
            print(f"\n  [{get_wib_time()}] Early Stopping! Val loss tidak membaik selama {args.patience} epoch.")
            break

    epochs_ran = len(train_losses)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, acc, precision, recall, f1, cm = evaluate(model, val_loader, criterion, device)

    print(f"\n  [{get_wib_time()}] Hasil Akhir Fold {fold+1}:")
    print(f"    Accuracy  : {acc:.4f}\n    Precision : {precision:.4f}\n    Recall    : {recall:.4f}\n    F1-Score  : {f1:.4f}")

    plot_training_history(fold, epochs_ran, train_losses, val_losses, val_accs, save_dir)
    plot_confusion_matrix(fold, cm, save_dir, args.model)
    cm_path = os.path.join(save_dir, f'fold{fold+1}_confusion_matrix.png')
    
    log_fold_result(run, fold, acc, precision, recall, f1, cm_path)
    if run: run.finish()

    return acc, precision, recall, f1, cm

# ==============================================================
# FUNGSI UTAMA — LOOP 5-FOLD
# ==============================================================

def run_all_folds(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\nDevice: {device}")
    print(f"[INFO] Mode Dataset : GAMBAR PNG (Precomputed)")
    print(f"[INFO] img_dir      : {args.img_dir}")

    save_dir = os.path.join(args.checkpoint_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    all_subjects = np.arange(1, 16)
    kf = KFold(n_splits=args.n_splits, shuffle=False)
    folds = list(kf.split(all_subjects))

    all_acc, all_prec, all_rec, all_f1, all_cm = [], [], [], [], []

    fold_range = [args.fold_only] if args.fold_only is not None else range(args.n_splits)

    for fold in fold_range:
        train_idx, val_idx = folds[fold]
        train_subs = all_subjects[train_idx].tolist()
        val_subs   = all_subjects[val_idx].tolist()

        train_ds = SEEDDataset(img_dir=args.img_dir, subject_ids=train_subs)
        val_ds   = SEEDDataset(img_dir=args.img_dir, subject_ids=val_subs)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=device.type == 'cuda')

        model = build_resnet18(num_classes=NUM_CLASSES, pretrained=args.pretrained) if args.model == 'resnet18' else BaselineCNN2D(num_classes=NUM_CLASSES)
        model.to(device)

        if fold == 0: print_model_summary(model, args.model.upper())

        run = init_wandb(args, fold=fold, run_type='fold')
        acc, prec, rec, f1, cm = train_one_fold(fold, model, train_loader, val_loader, args, device, save_dir, run=run)

        all_acc.append(acc); all_prec.append(prec); all_rec.append(rec); all_f1.append(f1); all_cm.append(cm)

    avg_cm = np.mean(all_cm, axis=0)

    print(f"\n{'='*60}")
    print(f"  HASIL AKHIR — {args.model.upper()} — {'Fold ' + str(args.fold_only+1) if args.fold_only is not None else '5-Fold CV'}")
    print(f"{'='*60}")
    print(f"  {'Fold':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*44}")
    for i, fold_idx in enumerate(fold_range):
        print(f"  Fold {fold_idx+1:<4} {all_acc[i]:>8.4f} {all_prec[i]:>8.4f} {all_rec[i]:>8.4f} {all_f1[i]:>8.4f}")

    if len(all_acc) > 1:
        print(f"  {'-'*44}")
        print(f"  {'Mean':<8} {np.mean(all_acc):>8.4f} {np.mean(all_prec):>8.4f} {np.mean(all_rec):>8.4f} {np.mean(all_f1):>8.4f}")
        print(f"  {'Std Dev':<8} {np.std(all_acc):>8.4f} {np.std(all_prec):>8.4f} {np.std(all_rec):>8.4f} {np.std(all_f1):>8.4f}")
    print(f"{'='*60}\n")

    plot_average_confusion_matrix(avg_cm, save_dir, args.model.upper())
    avg_cm_path = os.path.join(save_dir, 'avg_confusion_matrix.png')
    
    summary_run = init_wandb(args, fold=None, run_type='summary')
    log_summary(summary_run, all_acc, all_prec, all_rec, all_f1, avg_cm_path)
    if summary_run: summary_run.finish()

# ==============================================================
# ARGUMENT PARSER & ENTRY POINT
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Training ResNet-18 / Baseline CNN untuk klasifikasi emosi EEG')

    parser.add_argument('--img-dir', type=str, default='/kaggle/working/spectrograms',
                        help='Path ke folder PNG hasil data_reader.py.')
    parser.add_argument('--model',        type=str,   default='resnet18', choices=['resnet18', 'baseline'])
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=15)

    # Argumen Opsi Pretrained (Aktif secara default)
    parser.add_argument('--pretrained', action='store_true', default=True, help='Gunakan bobot ImageNet')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', help='Latih ResNet dari awal')

    # Argumen Progressive Unfreezing
    parser.add_argument('--phase2-epoch', type=int, default=11)
    parser.add_argument('--phase3-epoch', type=int, default=56)
    parser.add_argument('--lr-phase2-mult', type=float, default=0.01, help='Pengali LR Fase 2 (default: 0.01)')
    parser.add_argument('--lr-phase3-mult', type=float, default=0.001, help='Pengali LR Fase 3 (default: 0.001)')

    parser.add_argument('--n-splits',     type=int,   default=5)
    parser.add_argument('--fold-only',    type=int,   default=None)
    parser.add_argument('--num-workers',  type=int,   default=2)
    parser.add_argument('--no-cuda',      action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    # Wandb
    parser.add_argument('--wandb-project', type=str, default='eeg-emotion-classification')
    parser.add_argument('--wandb-entity',  type=str, default=None)
    parser.add_argument('--wandb-off',     action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    start_time_sec = time.time()
    start_date_str = get_wib_time()

    print(f"\n{'='*60}")
    print(f"  PROSES TRAINING DIMULAI PADA: {start_date_str}")
    print(f"{'='*60}\n")

    run_all_folds(args)

    end_time_sec = time.time()
    end_date_str = get_wib_time()

    elapsed          = end_time_sec - start_time_sec
    hours, rem       = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n{'='*60}")
    print(f"  TRAINING SELESAI PADA: {end_date_str}")
    print(f"  TOTAL WAKTU EKSEKUSI : {int(hours):02d} Jam, {int(minutes):02d} Menit, {int(seconds):02d} Detik")
    print(f"{'='*60}\n")