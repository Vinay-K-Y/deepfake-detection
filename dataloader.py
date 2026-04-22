"""
Data Loader for Deepfake Detection
Handles data loading, augmentation, and batching
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Root directory (should contain train/test/valid folders)
            split: 'train', 'test', or 'valid'
            transform: Augmentation pipeline
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []

        # Check if folder exists
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")

        # Load real images → label 0
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                self.samples.append((img_path, 0))
            for img_path in real_dir.glob('*.png'):
                self.samples.append((img_path, 0))

        # Load fake images → label 1
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                self.samples.append((img_path, 1))
            for img_path in fake_dir.glob('*.png'):
                self.samples.append((img_path, 1))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

        # Count distribution
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        
        print(f"{split.upper():5s}: {len(self.samples):6,} images (Real: {real_count:6,}, Fake: {fake_count:6,})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    """
    Create augmentation pipelines for training and validation
    """
    # ImageNet normalization (required for EfficientNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Path to processed_data folder
        batch_size: Batch size
        num_workers: Number of worker processes (use 0 on Windows)
    """
    train_transform, val_transform = get_transforms()

    # Create datasets
    train_dataset = DeepfakeDataset(data_dir, split='train', transform=train_transform)
    
    # Try to load validation set
    try:
        val_dataset = DeepfakeDataset(data_dir, split='valid', transform=val_transform)
    except:
        print("⚠️  No validation set found, will use test set for validation")
        val_dataset = None
    
    # Try to load test set
    try:
        test_dataset = DeepfakeDataset(data_dir, split='test', transform=val_transform)
    except:
        print("⚠️  No test set found")
        test_dataset = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataloader
    print("Testing DataLoader...")
    print("=" * 50)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir='processed_data',
        batch_size=32
    )
    
    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Valid batches: {len(val_loader)}")
    if test_loader:
        print(f"  Test batches : {len(test_loader)}")
    
    # Test one batch
    print("\nTesting first batch...")
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label distribution: Real={sum(labels==0).item()}, Fake={sum(labels==1).item()}")
