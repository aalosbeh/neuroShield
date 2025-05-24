# PyTorch Implementation of NeuroShield++

This directory contains the complete PyTorch implementation of the NeuroShield++ framework as described in the paper. The code is organized into several modules:

1. `data/` - Data loading and preprocessing utilities
2. `models/` - Neural network architecture implementations
3. `psychometric/` - AI-psychometric profiling components
4. `training/` - Training and evaluation scripts
5. `utils/` - Helper functions and utilities

## Requirements

```
torch>=1.10.0
torchvision>=0.11.0
pytorch-geometric>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## Quick Start

```python
from neuroshield.models import NeuroShieldPlusPlus
from neuroshield.data import MedFuseDataLoader
from neuroshield.psychometric import AttackerProfiler

# Load the dataset
dataloader = MedFuseDataLoader(
    ciciomt_path="path/to/CICIoMT2024",
    iot_icu_path="path/to/IoT-ICU",
    wustl_path="path/to/WUSTL-EHMS-2020",
    batch_size=512,
    use_gnn=True
)

# Initialize the model
model = NeuroShieldPlusPlus(
    input_dim=30,
    num_classes=26,
    use_transformer=True,
    use_cnn=True,
    use_gnn=True
)

# Initialize the profiler
profiler = AttackerProfiler()

# Train the model
from neuroshield.training import train_model
train_model(model, dataloader, profiler, epochs=50)

# Evaluate the model
from neuroshield.training import evaluate_model
metrics = evaluate_model(model, dataloader, profiler)
print(f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_macro']:.4f}")
```
