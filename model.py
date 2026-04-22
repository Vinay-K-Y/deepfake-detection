"""
Deepfake Detector Model using Transfer Learning
Base: EfficientNet-B0 pre-trained on ImageNet
"""
import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=1, dropout=0.5):
        super(DeepfakeDetector, self).__init__()

        # Load pre-trained EfficientNet B0
        self.base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Freeze early layers (transfer learning)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze last 3 blocks for fine-tuning
        for param in self.base_model.features[-3:].parameters():
            param.requires_grad = True

        # Replace classifier head for binary classification
        in_features = self.base_model.classifier[1].in_features

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        print("✅ All layers unfrozen for fine-tuning")


def get_model(device):
    """Create model and move to device"""
    model = DeepfakeDetector()
    model = model.to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Model Parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total    : {total:,}")
    print(f"  Frozen   : {total - trainable:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    model = get_model(device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape : {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.4f}")
