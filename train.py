"""
Training Script for Deepfake Detection Model
Uses transfer learning with EfficientNet-B0
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import multiprocessing

from model import get_model
from dataloader import get_dataloaders


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    'data_dir': 'processed_data',
    'batch_size': 32,           # Adjust based on your GPU VRAM
    'epochs': 20,
    'lr': 0.0001,
    'save_dir': 'saved_models',
    'num_workers': 0,           # Use 0 on Windows to avoid errors
    'unfreeze_epoch': 10        # Epoch to unfreeze all layers
}


# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════
def setup():
    # Create save directory
    Path(CONFIG['save_dir']).mkdir(exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("DEEPFAKE DETECTION - TRAINING")
    print("=" * 60)
    print(f"Device      : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name    : {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Batch Size  : {CONFIG['batch_size']}")
    print(f"Epochs      : {CONFIG['epochs']}")
    print(f"Learning Rate: {CONFIG['lr']}")
    print("=" * 60)
    
    return device


# ═══════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot([x*100 for x in history['train_acc']], label='Train Acc', marker='o')
    ax2.plot([x*100 for x in history['val_acc']], label='Val Acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════
def main():
    # Windows multiprocessing fix
    multiprocessing.freeze_support()
    
    # Setup
    device = setup()
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        CONFIG['data_dir'],
        CONFIG['batch_size'],
        CONFIG['num_workers']
    )
    
    if val_loader is None:
        print("⚠️  Using test set as validation set")
        val_loader = test_loader
    
    # Create model
    print("\nCreating model...")
    model = get_model(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # Training loop
    print("\n🚀 Starting training...\n")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, f"{CONFIG['save_dir']}/best_model.pth")
            print(f"  💾 Best model saved! (Val Loss: {val_loss:.4f})")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f}   | Val Acc  : {val_acc*100:.2f}%")
        print(f"  Best Val Acc: {best_val_acc*100:.2f}%")
        
        # Unfreeze all layers for fine-tuning
        if epoch + 1 == CONFIG['unfreeze_epoch']:
            print("\n" + "=" * 60)
            print("🔓 UNFREEZING ALL LAYERS FOR FINE-TUNING")
            print("=" * 60)
            model.unfreeze_all()
            optimizer = Adam(model.parameters(), lr=CONFIG['lr'] / 10)
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', patience=3, factor=0.5, verbose=True
            )
    
    # Save final model
    torch.save(model.state_dict(), f"{CONFIG['save_dir']}/final_model.pth")
    
    # Save training history
    with open(f"{CONFIG['save_dir']}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_training_curves(history, f"{CONFIG['save_dir']}/training_curves.png")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Validation Loss    : {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"\nModels saved in: {CONFIG['save_dir']}/")
    print("  - best_model.pth    (best validation loss)")
    print("  - final_model.pth   (after all epochs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
