# Dataset Documentation and Setup Instructions

## Overview

The NeuroShield++ framework utilizes a unified dataset called MedFuse-IDS24, which combines three state-of-the-art datasets for IoMT security:

1. **CICIoMT2024**: Contains IoMT network traffic with 18 attack types
2. **IoT-ICU Dataset**: Focuses on normal vs. malicious ICU traffic
3. **WUSTL-EHMS-2020**: Includes network and biometric features for MITM, spoofing, and injection attacks

This document provides instructions for downloading, setting up, and preprocessing these datasets for use with the NeuroShield++ framework.

## Dataset Sources and Access

### CICIoMT2024 Dataset

The Canadian Institute for Cybersecurity's IoMT dataset contains network traffic from 40 IoMT devices (25 real, 15 simulated) with various attack types.

**Download Instructions:**
1. Visit the CIC website: https://www.unb.ca/cic/datasets/iomt-2024.html
2. Register with your institutional email
3. Download the dataset files (approximately 15GB total)
4. Extract the files to a directory named `CICIoMT2024`

**Structure:**
```
CICIoMT2024/
├── Benign/
│   ├── Monday-WorkingHours.pcap
│   ├── Tuesday-WorkingHours.pcap
│   └── ...
├── DDoS/
│   ├── DDoS-UDPFlood.pcap
│   ├── DDoS-TCPFlood.pcap
│   └── ...
├── Reconnaissance/
│   ├── Recon-PortScan.pcap
│   └── ...
└── ...
```

### IoT-ICU Dataset

This dataset models a hospital ICU with smart beds and patient monitoring sensors, focusing on MQTT-based communications.

**Download Instructions:**
1. Access the IEEE DataPort: https://ieee-dataport.org/documents/iot-healthcare-security-dataset
2. Download the CSV files (approximately 2GB)
3. Extract to a directory named `IoT-ICU`

**Structure:**
```
IoT-ICU/
├── benign_traffic.csv
├── mqtt_exploit.csv
├── dos_attack.csv
├── scan_attack.csv
└── metadata.json
```

### WUSTL-EHMS-2020 Dataset

This dataset from Washington University's Enhanced Healthcare Monitoring System testbed includes both network flow features and patient biometric readings.

**Download Instructions:**
1. Visit the WUSTL repository: https://github.com/WUSTL-EHMS/EHMS-Dataset
2. Clone or download the repository
3. Extract to a directory named `WUSTL-EHMS-2020`

**Structure:**
```
WUSTL-EHMS-2020/
├── normal_traffic.csv
├── mitm_arpspoof.csv
├── mitm_datainjection.csv
└── dataset_description.txt
```

## Dataset Preprocessing

The NeuroShield++ framework includes a comprehensive data preprocessing pipeline that handles:

1. Feature harmonization across datasets
2. Handling of missing values
3. Encoding of categorical features
4. Normalization of numerical features
5. Creation of device graphs for GNN processing

### Manual Preprocessing (if needed)

If you need to perform any preprocessing steps manually before using our data loader:

1. **Convert PCAP files to CSV** (for CICIoMT2024):
   ```bash
   # Install required tools
   sudo apt-get install tshark

   # Convert PCAP to CSV
   tshark -r input.pcap -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e ip.proto -e frame.len -E header=y -E separator=, > output.csv
   ```

2. **Harmonize column names** across datasets:
   - Rename columns to match our expected format
   - Ensure consistent data types

3. **Label encoding** for categorical features:
   ```python
   from sklearn.preprocessing import LabelEncoder
   
   le = LabelEncoder()
   df['protocol_encoded'] = le.fit_transform(df['protocol'])
   ```

## Using the Data Loader

Our `MedFuseDataLoader` class handles all the preprocessing steps automatically. Here's how to use it:

```python
from neuroshield.data import MedFuseDataLoader

# Initialize the data loader
data_loader = MedFuseDataLoader(
    ciciomt_path="/path/to/CICIoMT2024",
    iot_icu_path="/path/to/IoT-ICU",
    wustl_path="/path/to/WUSTL-EHMS-2020",
    batch_size=512,
    sequence_length=10,
    use_gnn=True,
    feature_selection=True
)

# Get data loaders for training, validation, and testing
train_loader, val_loader, test_loader = data_loader.get_loaders()

# Get GNN data if using graph neural networks
if data_loader.use_gnn:
    gnn_data = data_loader.get_gnn_data()

# Get number of classes and feature dimensions
num_classes = data_loader.get_num_classes()
feature_dim = data_loader.get_feature_dim()

# Get class names
class_names = data_loader.get_class_names()
```

## Dataset Statistics

### Class Distribution

The unified MedFuse-IDS24 dataset contains approximately 2.3 million records with the following distribution:

- **Normal traffic**: ~1.6 million records (70%)
- **Attack traffic**: ~700,000 records (30%)
  - DDoS/DoS: ~300,000 records
  - Reconnaissance: ~50,000 records
  - Spoofing/MITM: ~80,000 records
  - Malware/Ransomware: ~50,000 records
  - Injection: ~20,000 records
  - Others (phishing, rogue device): ~200,000 records

### Feature Information

The dataset includes the following feature categories:

1. **Network flow features** (25 standardized features):
   - Source/destination IPs and ports
   - Protocol information
   - Packet counts and byte volumes
   - Flow duration
   - IoT-specific fields (MQTT topics, BLE signal strength)

2. **Device metadata**:
   - Device type (e.g., ECG Sensor, InfusionPump)
   - Network segment (e.g., WardA VLAN, ICU Subnet)

3. **Patient biometrics** (from WUSTL-EHMS):
   - Heart rate
   - Oxygen saturation (SpO2)
   - Blood pressure (systolic/diastolic)
   - Temperature
   - Respiratory rate
   - Glucose levels

## Data Augmentation

For handling class imbalance, we apply:

1. **SMOTE** (Synthetic Minority Over-sampling Technique) to create synthetic samples for minority classes
2. **Time-series augmentation** for sequential models by injecting noise or duplicating pattern segments

## Troubleshooting

Common issues and solutions:

1. **Missing files or corrupt downloads**:
   - Verify checksums against those provided on the dataset websites
   - Re-download problematic files

2. **Memory errors during loading**:
   - Reduce batch size
   - Process datasets in chunks
   - Use a machine with more RAM

3. **Class imbalance issues**:
   - Adjust class weights in the loss function
   - Modify the SMOTE sampling rate

4. **Feature compatibility**:
   - If adding new features, ensure they are properly normalized and encoded

## Citation Information

If you use these datasets in your research, please cite the original sources:

```
@inproceedings{ciciomt2024,
  title={CICIoMT2024: A Diverse Dataset for Internet of Medical Things Security},
  author={Canadian Institute for Cybersecurity},
  booktitle={Proceedings of the IEEE Conference on Communications and Network Security},
  year={2024}
}

@article{hussain2021,
  title={IoT-Flock: An Open-source Framework for IoT Traffic Generation},
  author={Hussain, F. and Hussain, R. and Hassan, S. A. and Hossain, E.},
  journal={IEEE Access},
  volume={9},
  pages={29631--29647},
  year={2021}
}

@inproceedings{wustl2020,
  title={WUSTL-EHMS-2020: A Comprehensive Dataset for Security Assessment of Enhanced Healthcare Monitoring Systems},
  author={Washington University in St. Louis},
  booktitle={Proceedings of the ACM Conference on Computer and Communications Security},
  year={2020}
}
```
