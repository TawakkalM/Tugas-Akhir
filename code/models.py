import torch
import torch.nn as nn
import torchvision.models as tv_models

# ==============================================================
# KONFIGURASI
# ==============================================================
NUM_CLASSES = 3   # Negatif, Netral, Positif

# ==============================================================
# TAHAP 5A — MODEL UTAMA: ResNet-18 
# ==============================================================

def build_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = tv_models.resnet18(weights=weights)

    # Eksperimen 14: Bekukan secara permanen bagian awal model (Feature Extractor)
    if pretrained:
        # Langkah 1: Matikan semua gembok parameter
        for param in model.parameters():
            param.requires_grad = False
        
        # Langkah 2: Buka gembok hanya untuk layer4 (blok residual terakhir)
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        print("  [Init] Eksperimen 14 Aktif: Layer 1-3 dibekukan permanen. Hanya Layer 4 & FC yang dilatih.")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("  [Init] Pretrained=False: Semua parameter dilatih dari nol (Scratch).")

    # Lapisan Klasifikasi Akhir dengan Dropout 0.3
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    # FC Layer harus selalu bisa dilatih
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

# Fungsi unfreeze_layers dihilangkan karena kita menggunakan pembekuan permanen