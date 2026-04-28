import torch
import torch.nn as nn
import torchvision.models as tv_models

# ==============================================================
# KONFIGURASI
# ==============================================================
NUM_CLASSES = 3   # Negatif, Netral, Positif


# ==============================================================
# TAHAP 5A — MODEL UTAMA: ResNet-18 dengan Transfer Learning
# ==============================================================

def build_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    """
    Membangun ResNet-18 dengan strategi Progressive Unfreezing.

    Fase 1 (awal): Semua layer konvolusi DIBEKUKAN.
                   Hanya FC layer yang dilatih.
                   Parameter aktif: ~1.500 dari 11 juta.

    Fase 2 & 3   : Layer dibuka bertahap via unfreeze_layers().

    Input  : (batch, 3, 224, 224)
    Output : (batch, 3)
    """
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = tv_models.resnet18(weights=weights)

    # Bekukan SEMUA layer terlebih dahulu
    for param in model.parameters():
        param.requires_grad = False

    # Ganti FC layer dengan Dropout + Linear, selalu trainable
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    # FC layer selalu trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_layers(model, phase):
    """
    Buka layer ResNet-18 secara bertahap sesuai fase training.

    Phase 1 : hanya FC (sudah diset di build_resnet18)
    Phase 2 : buka layer4 + FC
    Phase 3 : buka layer3 + layer4 + FC
    """
    if phase == 2:
        for param in model.layer4.parameters():
            param.requires_grad = True
        print("  [Unfreeze] layer4 dibuka untuk dilatih")

    elif phase == 3:
        for param in model.layer3.parameters():
            param.requires_grad = True
        print("  [Unfreeze] layer3 dibuka untuk dilatih")

    # Hitung parameter yang aktif
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [Unfreeze] Trainable: {trainable:,} / {total:,} parameter")


# ==============================================================
# TAHAP 5B — MODEL BASELINE: CNN-2D Standar (tanpa Residual)
# ==============================================================

class BaselineCNN2D(nn.Module):
    """
    CNN-2D standar sebagai model pembanding (baseline).
    Tidak menggunakan mekanisme residual connection.

    Arsitektur:
    - 3 blok konvolusi (Conv2d → BatchNorm → ReLU → MaxPool)
    - Global Average Pooling
    - Fully Connected → 3 kelas

    Input  : (batch, 3, 224, 224)
    Output : (batch, 3)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(BaselineCNN2D, self).__init__()

        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → (32, 112, 112)

            # Blok 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → (64, 56, 56)

            # Blok 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # → (128, 28, 28)
        )

        # Global Average Pooling → (128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==============================================================
# FUNGSI UTILITAS MODEL
# ==============================================================

def count_parameters(model):
    """Hitung jumlah total dan trainable parameter."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model, model_name="Model"):
    """Cetak ringkasan sederhana model."""
    total, trainable = count_parameters(model)
    frozen = total - trainable
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Total parameter    : {total:,}")
    print(f"  Trainable parameter: {trainable:,}")
    print(f"  Frozen parameter   : {frozen:,}")
    pct = trainable / total * 100
    print(f"  % Dilatih          : {pct:.1f}%")
    print(f"{'='*50}\n")


# ==============================================================
# TESTING SCRIPT
# ==============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Simulasi input spektrogram RGB 224×224
    dummy_input = torch.randn(4, 3, 224, 224).to(device)

    # --- Uji ResNet-18 ---
    print("\n=== ResNet-18 (Transfer Learning) ===")
    resnet = build_resnet18(pretrained=True).to(device)
    out    = resnet(dummy_input)
    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {out.shape}  (harusnya: [4, 3])")
    print_model_summary(resnet, "ResNet-18 + Transfer Learning")

    # --- Uji Baseline CNN-2D ---
    print("=== Baseline CNN-2D ===")
    baseline = BaselineCNN2D().to(device)
    out_b    = baseline(dummy_input)
    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {out_b.shape}  (harusnya: [4, 3])")
    print_model_summary(baseline, "Baseline CNN-2D")