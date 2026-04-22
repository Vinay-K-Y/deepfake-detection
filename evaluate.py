import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, roc_curve, confusion_matrix
)
from model import DeepfakeDetector
from dataloader import get_dataloaders
import multiprocessing

def calculate_eer(y_true, y_prob):
    """Calculates the Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    # Find the threshold where False Positive Rate == False Negative Rate
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    return fpr[eer_index]

def evaluate_model():
    print("=" * 50)
    print("DEEPFAKE MODEL EVALUATION")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("\nLoading validation dataset...")
    # Assuming 'processed_data' is your data folder as per train.py
    _, val_loader, test_loader = get_dataloaders('processed_data', batch_size=32, num_workers=0)
    
    # Use test loader if available, otherwise fallback to val loader
    eval_loader = test_loader if test_loader else val_loader
    if eval_loader is None:
        print("❌ Error: No validation or test data found.")
        return

    # 2. Load Model
    print("Loading best_model.pth...")
    model = DeepfakeDetector()
    checkpoint = torch.load("saved_models/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    # 3. Run Inference
    print("Running inference on dataset... This may take a minute.")
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Forward pass
            outputs = model(images).squeeze()
            probs = outputs.cpu().numpy()
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            preds = (probs > 0.5).astype(int)

            # Store results
            all_labels.extend(labels)
            
            # Handle batch size of 1 edge case
            if probs.ndim == 0:
                all_probs.append(probs.item())
                all_preds.append(preds.item())
            else:
                all_probs.extend(probs)
                all_preds.extend(preds)

    # 4. Calculate Metrics
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    loss_log = log_loss(y_true, y_prob)
    eer = calculate_eer(y_true, y_prob)

    # 5. Print Results
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy   : {accuracy * 100:.2f}%")
    print(f"Precision  : {precision * 100:.2f}%")
    print(f"Recall     : {recall * 100:.2f}%")
    print(f"F1-Score   : {f1 * 100:.2f}%")
    print(f"AUC-ROC    : {auc_roc:.4f}")
    print(f"Log Loss   : {loss_log:.4f}")
    print(f"EER        : {eer * 100:.2f}%")
    print("=" * 50)

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Saved 'confusion_matrix.png' and 'roc_curve.png' to your folder!")

if __name__ == '__main__':
    # Required for Windows multi-processing safety
    multiprocessing.freeze_support()
    evaluate_model()