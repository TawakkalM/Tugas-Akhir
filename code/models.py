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

    # Ganti FC layer terakhir dan tambahkan Dropout 0.5
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Eksperimen 16: Label Smoothing + Dropout 0.5
        nn.Linear(in_features, num_classes)
    )

    # Pastikan SEMUA parameter bisa dilatih (Full Fine-Tuning) sejak awal
    for param in model.parameters():
        param.requires_grad = True
        
    print(f"  [Init] ResNet-18 dimuat. Pretrained={pretrained}. Full Fine-Tuning aktif.")

    return model

# ==============================================================
# TAHAP 5B — MODEL BASELINE: CNN-2D Standar
# ==============================================================

class BaselineCNN2D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(BaselineCNN2D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
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
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_summary(model, model_name="Model"):
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