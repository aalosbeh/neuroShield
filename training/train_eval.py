import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import time
import os
from tqdm import tqdm

def train_model(model, data_loader, profiler=None, epochs=50, lr=1e-3, weight_decay=1e-5, 
                patience=10, checkpoint_dir='checkpoints', device=None):
    """
    Train the NeuroShield++ model
    
    Args:
        model (nn.Module): NeuroShield++ model
        data_loader (MedFuseDataLoader): Data loader object
        profiler (AttackerProfiler, optional): Psychometric profiler
        epochs (int): Number of training epochs
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
        patience (int): Patience for early stopping
        checkpoint_dir (str): Directory to save checkpoints
        device (torch.device, optional): Device to use for training
        
    Returns:
        dict: Training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on {device}")
    model = model.to(device)
    
    # Get data loaders
    train_loader, val_loader, _ = data_loader.get_loaders()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_bar:
            # Move data to device
            sequences = batch['sequence'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            data = {'sequence': sequences, 'features': features}
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            train_bar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_bar:
                # Move data to device
                sequences = batch['sequence'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                data = {'sequence': sequences, 'features': features}
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Update progress bar
                val_bar.set_postfix({'loss': loss.item(), 'acc': val_correct / val_total})
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            
            print(f"Saved checkpoint at epoch {epoch+1}")
        else:
            no_improve += 1
            
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Training completed. Best model from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    return history


def evaluate_model(model, data_loader, profiler=None, device=None, class_names=None):
    """
    Evaluate the NeuroShield++ model
    
    Args:
        model (nn.Module): NeuroShield++ model
        data_loader (MedFuseDataLoader): Data loader object
        profiler (AttackerProfiler, optional): Psychometric profiler
        device (torch.device, optional): Device to use for evaluation
        class_names (list, optional): List of class names
        
    Returns:
        dict: Evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating on {device}")
    model = model.to(device)
    
    # Get test loader
    _, _, test_loader = data_loader.get_loaders()
    
    # Get class names if not provided
    if class_names is None:
        class_names = data_loader.get_class_names()
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            sequences = batch['sequence'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            data = {'sequence': sequences, 'features': features}
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate ROC curve and AUC for each class
    n_classes = len(class_names)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            (all_labels == i).astype(int), 
            all_probabilities[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    macro_roc_auc = auc(all_fpr, mean_tpr)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print(f"ROC AUC (macro): {macro_roc_auc:.4f}")
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={per_class_precision[i]:.4f}, "
              f"Recall={per_class_recall[i]:.4f}, F1={per_class_f1[i]:.4f}, "
              f"AUC={roc_auc[i]:.4f}")
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'roc_auc_macro': macro_roc_auc,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_roc_auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'all_fpr': all_fpr,
        'mean_tpr': mean_tpr,
        'class_names': class_names
    }
    
    return metrics


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history (dict): Training history
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {save_path}")
    
    plt.show()


def plot_roc_curves(metrics, save_path=None):
    """
    Plot ROC curves
    
    Args:
        metrics (dict): Evaluation metrics
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(metrics['class_names']):
        plt.plot(metrics['fpr'][i], metrics['tpr'][i],
                 label=f'{class_name} (AUC = {metrics["per_class_roc_auc"][i]:.2f})')
    
    # Plot macro-average ROC curve
    plt.plot(metrics['all_fpr'], metrics['mean_tpr'],
             label=f'Macro-average (AUC = {metrics["roc_auc_macro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves plot to {save_path}")
    
    plt.show()


def measure_inference_time(model, data_loader, num_samples=100, device=None):
    """
    Measure inference time
    
    Args:
        model (nn.Module): NeuroShield++ model
        data_loader (MedFuseDataLoader): Data loader object
        num_samples (int): Number of samples to measure
        device (torch.device, optional): Device to use for inference
        
    Returns:
        float: Average inference time per sample (ms)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Get test loader
    _, _, test_loader = data_loader.get_loaders()
    
    # Get a batch
    for batch in test_loader:
        sequences = batch['sequence'].to(device)
        features = batch['features'].to(device)
        break
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            data = {'sequence': sequences, 'features': features}
            _ = model(data)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_samples):
            data = {'sequence': sequences, 'features': features}
            _ = model(data)
    
    end_time = time.time()
    
    # Calculate average inference time
    total_time = (end_time - start_time) * 1000  # Convert to ms
    avg_time = total_time / num_samples
    
    print(f"Average inference time: {avg_time:.2f} ms per sample")
    
    return avg_time


def evaluate_with_profiling(model, data_loader, profiler, device=None):
    """
    Evaluate model with psychometric profiling
    
    Args:
        model (nn.Module): NeuroShield++ model
        data_loader (MedFuseDataLoader): Data loader object
        profiler (AttackerProfiler): Psychometric profiler
        device (torch.device, optional): Device to use for evaluation
        
    Returns:
        dict: Evaluation metrics with profiling
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Get test loader and class names
    _, _, test_loader = data_loader.get_loaders()
    class_names = data_loader.get_class_names()
    
    # Initialize metrics
    detection_times = {
        'with_profiling': [],
        'without_profiling': []
    }
    
    # Simulate attack sequences
    attack_sequences = [
        # Sequence 1: Reconnaissance -> MITM
        [
            {'attack_type': 'Reconnaissance-Scan', 'packet_count': 20, 'target_count': 10,
             'critical_target': False, 'evasion_techniques': ['slow_scan'], 
             'payload_complexity': 0.3, 'timing_pattern': 'irregular'},
            {'attack_type': 'MITM-ARPspoof', 'packet_count': 50, 'target_count': 2,
             'critical_target': True, 'evasion_techniques': ['mac_spoofing', 'timing_evasion'], 
             'payload_complexity': 0.7, 'timing_pattern': 'regular'}
        ],
        # Sequence 2: DDoS -> Malware
        [
            {'attack_type': 'DDoS-UDPFlood', 'packet_count': 1000, 'target_count': 5,
             'critical_target': False, 'evasion_techniques': [], 
             'payload_complexity': 0.2, 'timing_pattern': 'regular'},
            {'attack_type': 'Malware-Ransomware', 'packet_count': 30, 'target_count': 3,
             'critical_target': True, 'evasion_techniques': ['encryption', 'obfuscation'], 
             'payload_complexity': 0.9, 'timing_pattern': 'irregular'}
        ]
    ]
    
    # Process each attack sequence
    for sequence_idx, attack_sequence in enumerate(attack_sequences):
        print(f"\nProcessing attack sequence {sequence_idx+1}:")
        
        # Reset profiler for each sequence
        profiler_active = AttackerProfiler()
        profiler_inactive = AttackerProfiler()
        
        for attack_idx, attack_data in enumerate(attack_sequence):
            print(f"  Attack {attack_idx+1}: {attack_data['attack_type']}")
            
            # With profiling
            if attack_idx > 0:  # Not the first attack in sequence
                # Get base threshold
                base_threshold = 0.5
                
                # Adjust threshold based on profile
                attack_class = attack_data['attack_type'].split('-')[0]
                adjusted_threshold = profiler_active.adjust_detection_threshold(base_threshold, attack_class)
                
                print(f"    Base threshold: {base_threshold:.4f}, Adjusted: {adjusted_threshold:.4f}")
                
                # Simulate detection with adjusted threshold
                detection_time_with = simulate_detection_time(
                    model, test_loader, attack_data['attack_type'], adjusted_threshold, device
                )
                detection_times['with_profiling'].append(detection_time_with)
                
                # Simulate detection without adjustment
                detection_time_without = simulate_detection_time(
                    model, test_loader, attack_data['attack_type'], base_threshold, device
                )
                detection_times['without_profiling'].append(detection_time_without)
                
                print(f"    Detection time with profiling: {detection_time_with}")
                print(f"    Detection time without profiling: {detection_time_without}")
                print(f"    Improvement: {(detection_time_without - detection_time_with) / detection_time_without * 100:.2f}%")
            
            # Update profiles
            profiler_active.update_profile(attack_data)
            
            # Print profile
            if attack_idx == len(attack_sequence) - 1:
                print("\n  Final attacker profile:")
                profile = profiler_active.get_profile()
                for trait, value in profile.items():
                    print(f"    {trait}: {value:.4f}")
                
                print("\n  " + profiler_active.get_profile_description())
    
    # Calculate average improvement
    if detection_times['with_profiling']:
        avg_with = sum(detection_times['with_profiling']) / len(detection_times['with_profiling'])
        avg_without = sum(detection_times['without_profiling']) / len(detection_times['without_profiling'])
        improvement = (avg_without - avg_with) / avg_without * 100
        
        print(f"\nAverage detection time with profiling: {avg_with:.2f}")
        print(f"Average detection time without profiling: {avg_without:.2f}")
        print(f"Average improvement: {improvement:.2f}%")
    
    # Return metrics
    metrics = {
        'detection_times': detection_times,
        'avg_with_profiling': avg_with if detection_times['with_profiling'] else None,
        'avg_without_profiling': avg_without if detection_times['without_profiling'] else None,
        'improvement_percentage': improvement if detection_times['with_profiling'] else None
    }
    
    return metrics


def simulate_detection_time(model, data_loader, attack_type, threshold, device):
    """
    Simulate detection time for an attack
    
    Args:
        model (nn.Module): NeuroShield++ model
        data_loader (DataLoader): Test data loader
        attack_type (str): Attack type to detect
        threshold (float): Detection threshold
        device (torch.device): Device to use
        
    Returns:
        int: Number of packets needed for detection
    """
    # This is a simplified simulation
    # In a real implementation, this would use actual sequential data
    
    # Find samples of the specified attack type
    attack_samples = []
    attack_class_idx = None
    
    for batch in data_loader:
        sequences = batch['sequence']
        labels = batch['label']
        
        for i, label in enumerate(labels):
            label_name = data_loader.dataset.label_encoder.inverse_transform([label.item()])[0]
            if label_name == attack_type:
                attack_samples.append(sequences[i])
                attack_class_idx = label.item()
                if len(attack_samples) >= 10:  # Get a few samples
                    break
        
        if len(attack_samples) >= 10:
            break
    
    if not attack_samples:
        print(f"Warning: No samples found for attack type {attack_type}")
        return 10  # Default value
    
    # Simulate detection process
    # We'll use a random sample and check how many steps it takes to exceed threshold
    sample = attack_samples[0].to(device)
    
    # Simulate detection with increasing context
    for packets in range(1, sample.size(0) + 1):
        # Use partial sequence
        partial_sequence = sample[:packets].unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            data = {'sequence': partial_sequence, 'features': sample[packets-1].unsqueeze(0)}
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Check if probability exceeds threshold
        if probabilities[0, attack_class_idx].item() > threshold:
            return packets
    
    # If not detected, return sequence length
    return sample.size(0)
