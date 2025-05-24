import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch_geometric.data as gdata

class MedFuseDataset(Dataset):
    """
    Unified dataset for NeuroShield++ that combines CICIoMT2024, IoT-ICU, and WUSTL-EHMS-2020
    """
    def __init__(self, data, labels, device_ids=None, sequence_length=10, transform=None):
        """
        Initialize the dataset
        
        Args:
            data (np.ndarray): Feature data
            labels (np.ndarray): Labels
            device_ids (np.ndarray, optional): Device IDs for GNN construction
            sequence_length (int): Length of sequences for Transformer
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.device_ids = torch.LongTensor(device_ids) if device_ids is not None else None
        self.sequence_length = sequence_length
        self.transform = transform
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
        
    def __getitem__(self, idx):
        # For Transformer: create a sequence of consecutive samples
        if idx + self.sequence_length > len(self.data):
            idx = len(self.data) - self.sequence_length
            
        sequence = self.data[idx:idx+self.sequence_length]
        label = self.labels[idx+self.sequence_length-1]  # Label of the last element
        
        sample = {
            'sequence': sequence,
            'features': self.data[idx+self.sequence_length-1],  # Current features for CNN
            'label': label
        }
        
        if self.device_ids is not None:
            sample['device_id'] = self.device_ids[idx+self.sequence_length-1]
            
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class MedFuseDataLoader:
    """
    Data loader for the unified MedFuse-IDS24 dataset
    """
    def __init__(self, ciciomt_path, iot_icu_path, wustl_path, batch_size=512, 
                 sequence_length=10, use_gnn=False, feature_selection=True):
        """
        Initialize the data loader
        
        Args:
            ciciomt_path (str): Path to CICIoMT2024 dataset
            iot_icu_path (str): Path to IoT-ICU dataset
            wustl_path (str): Path to WUSTL-EHMS-2020 dataset
            batch_size (int): Batch size for data loader
            sequence_length (int): Length of sequences for Transformer
            use_gnn (bool): Whether to use GNN features
            feature_selection (bool): Whether to use feature selection
        """
        self.ciciomt_path = ciciomt_path
        self.iot_icu_path = iot_icu_path
        self.wustl_path = wustl_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.use_gnn = use_gnn
        self.feature_selection = feature_selection
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess all datasets
        """
        print("Loading CICIoMT2024 dataset...")
        ciciomt_data = self._load_ciciomt()
        
        print("Loading IoT-ICU dataset...")
        iot_icu_data = self._load_iot_icu()
        
        print("Loading WUSTL-EHMS-2020 dataset...")
        wustl_data = self._load_wustl()
        
        print("Merging datasets...")
        self.merged_data = self._merge_datasets(ciciomt_data, iot_icu_data, wustl_data)
        
        print("Preprocessing merged dataset...")
        self._preprocess_merged_data()
        
        print("Creating train/val/test splits...")
        self._create_data_splits()
        
        print("Data loading complete!")
        
    def _load_ciciomt(self):
        """
        Load CICIoMT2024 dataset
        
        Returns:
            pd.DataFrame: Loaded and preprocessed CICIoMT data
        """
        # In a real implementation, this would load the actual dataset files
        # For this example, we'll create a synthetic version
        
        # Simulate loading multiple CSV files from the dataset
        df_list = []
        
        # Benign traffic
        benign_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10000, freq='1s'),
            'src_ip': np.random.choice(['192.168.1.' + str(i) for i in range(1, 50)], 10000),
            'dst_ip': np.random.choice(['192.168.1.' + str(i) for i in range(50, 100)], 10000),
            'src_port': np.random.randint(1024, 65535, 10000),
            'dst_port': np.random.choice([80, 443, 8080, 22, 23], 10000),
            'protocol': np.random.choice(['TCP', 'UDP', 'MQTT', 'BLE'], 10000),
            'packet_count': np.random.randint(1, 100, 10000),
            'byte_volume': np.random.randint(100, 10000, 10000),
            'flow_duration': np.random.uniform(0.1, 10.0, 10000),
            'device_type': np.random.choice(['ECGSensor', 'InfusionPump', 'PatientMonitor'], 10000),
            'network_segment': np.random.choice(['WardA', 'ICU', 'NursesStation'], 10000),
            'label': 'Normal'
        })
        df_list.append(benign_df)
        
        # DDoS attack
        ddos_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-02', periods=2000, freq='100ms'),
            'src_ip': np.random.choice(['10.0.0.' + str(i) for i in range(1, 20)], 2000),
            'dst_ip': np.random.choice(['192.168.1.' + str(i) for i in range(50, 60)], 2000),
            'src_port': np.random.randint(1024, 65535, 2000),
            'dst_port': np.random.choice([80, 443], 2000),
            'protocol': np.random.choice(['TCP', 'UDP'], 2000),
            'packet_count': np.random.randint(100, 1000, 2000),
            'byte_volume': np.random.randint(10000, 100000, 2000),
            'flow_duration': np.random.uniform(0.01, 0.5, 2000),
            'device_type': np.random.choice(['Router', 'Gateway'], 2000),
            'network_segment': np.random.choice(['WardA', 'ICU'], 2000),
            'label': 'DDoS-UDPFlood'
        })
        df_list.append(ddos_df)
        
        # Reconnaissance attack
        recon_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-03', periods=1000, freq='2s'),
            'src_ip': np.random.choice(['172.16.0.' + str(i) for i in range(1, 10)], 1000),
            'dst_ip': np.random.choice(['192.168.1.' + str(i) for i in range(1, 100)], 1000),
            'src_port': np.random.randint(1024, 65535, 1000),
            'dst_port': np.random.choice(range(1, 1024), 1000),
            'protocol': np.random.choice(['TCP'], 1000),
            'packet_count': np.random.randint(1, 10, 1000),
            'byte_volume': np.random.randint(40, 100, 1000),
            'flow_duration': np.random.uniform(0.01, 0.1, 1000),
            'device_type': np.random.choice(['Unknown'], 1000),
            'network_segment': np.random.choice(['WardA', 'ICU', 'NursesStation'], 1000),
            'label': 'Reconnaissance-Scan'
        })
        df_list.append(recon_df)
        
        # Combine all dataframes
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Add CICIoMT-specific features
        combined_df['mqtt_topic'] = np.where(
            combined_df['protocol'] == 'MQTT',
            np.random.choice(['patient/vitals', 'device/status', 'alerts'], combined_df.shape[0]),
            None
        )
        
        combined_df['ble_signal_strength'] = np.where(
            combined_df['protocol'] == 'BLE',
            np.random.uniform(-90, -30, combined_df.shape[0]),
            None
        )
        
        # Add dataset source column
        combined_df['source'] = 'CICIoMT2024'
        
        return combined_df
    
    def _load_iot_icu(self):
        """
        Load IoT-ICU dataset
        
        Returns:
            pd.DataFrame: Loaded and preprocessed IoT-ICU data
        """
        # Simulate loading the IoT-ICU dataset
        # Create synthetic data for demonstration
        
        # Benign ICU traffic
        benign_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5000, freq='2s'),
            'src_ip': np.random.choice(['192.168.2.' + str(i) for i in range(1, 20)], 5000),
            'dst_ip': np.random.choice(['192.168.2.' + str(i) for i in range(20, 30)], 5000),
            'src_port': np.random.randint(1024, 65535, 5000),
            'dst_port': np.random.choice([1883, 8883], 5000),  # MQTT ports
            'protocol': np.random.choice(['MQTT', 'TCP'], 5000),
            'packet_count': np.random.randint(1, 50, 5000),
            'byte_volume': np.random.randint(100, 5000, 5000),
            'flow_duration': np.random.uniform(0.1, 5.0, 5000),
            'device_type': np.random.choice(['ECGSensor', 'SpO2Sensor', 'BPMonitor', 'BedControl'], 5000),
            'network_segment': np.random.choice(['ICU'], 5000),
            'mqtt_topic': np.random.choice(['patient/vitals/ecg', 'patient/vitals/spo2', 'patient/vitals/bp', 'bed/control'], 5000),
            'label': 'Normal'
        })
        
        # MQTT exploit attack
        mqtt_attack_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-02', periods=1000, freq='1s'),
            'src_ip': np.random.choice(['10.0.0.' + str(i) for i in range(1, 10)], 1000),
            'dst_ip': np.random.choice(['192.168.2.' + str(i) for i in range(20, 30)], 1000),
            'src_port': np.random.randint(1024, 65535, 1000),
            'dst_port': np.random.choice([1883, 8883], 1000),  # MQTT ports
            'protocol': np.random.choice(['MQTT'], 1000),
            'packet_count': np.random.randint(1, 20, 1000),
            'byte_volume': np.random.randint(100, 2000, 1000),
            'flow_duration': np.random.uniform(0.1, 2.0, 1000),
            'device_type': np.random.choice(['Unknown'], 1000),
            'network_segment': np.random.choice(['ICU'], 1000),
            'mqtt_topic': np.random.choice(['patient/vitals/ecg', 'patient/vitals/spo2', 'patient/vitals/bp', 'bed/control'], 1000),
            'label': 'MQTT-slowite'
        })
        
        # Combine all dataframes
        combined_df = pd.concat([benign_df, mqtt_attack_df], ignore_index=True)
        
        # Add dataset source column
        combined_df['source'] = 'IoT-ICU'
        
        return combined_df
    
    def _load_wustl(self):
        """
        Load WUSTL-EHMS-2020 dataset
        
        Returns:
            pd.DataFrame: Loaded and preprocessed WUSTL-EHMS data
        """
        # Simulate loading the WUSTL-EHMS dataset
        # Create synthetic data for demonstration
        
        # Benign traffic with biometrics
        benign_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=8000, freq='1s'),
            'src_ip': np.random.choice(['192.168.3.' + str(i) for i in range(1, 20)], 8000),
            'dst_ip': np.random.choice(['192.168.3.' + str(i) for i in range(20, 30)], 8000),
            'src_port': np.random.randint(1024, 65535, 8000),
            'dst_port': np.random.choice([80, 443, 8080], 8000),
            'protocol': np.random.choice(['TCP', 'UDP'], 8000),
            'packet_count': np.random.randint(1, 100, 8000),
            'byte_volume': np.random.randint(100, 10000, 8000),
            'flow_duration': np.random.uniform(0.1, 10.0, 8000),
            'device_type': np.random.choice(['ECGSensor', 'SpO2Sensor', 'BPMonitor'], 8000),
            'network_segment': np.random.choice(['WardB'], 8000),
            'heart_rate': np.random.randint(60, 100, 8000),
            'spo2': np.random.randint(95, 100, 8000),
            'systolic_bp': np.random.randint(110, 140, 8000),
            'diastolic_bp': np.random.randint(60, 90, 8000),
            'temperature': np.random.uniform(36.5, 37.5, 8000),
            'respiratory_rate': np.random.randint(12, 20, 8000),
            'glucose': np.random.randint(70, 120, 8000),
            'label': 'Normal'
        })
        
        # MITM attack
        mitm_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-02', periods=1000, freq='1s'),
            'src_ip': np.random.choice(['10.0.0.' + str(i) for i in range(1, 10)], 1000),
            'dst_ip': np.random.choice(['192.168.3.' + str(i) for i in range(20, 30)], 1000),
            'src_port': np.random.randint(1024, 65535, 1000),
            'dst_port': np.random.choice([80, 443, 8080], 1000),
            'protocol': np.random.choice(['TCP', 'UDP'], 1000),
            'packet_count': np.random.randint(1, 100, 1000),
            'byte_volume': np.random.randint(100, 10000, 1000),
            'flow_duration': np.random.uniform(0.1, 10.0, 1000),
            'device_type': np.random.choice(['ECGSensor', 'SpO2Sensor', 'BPMonitor'], 1000),
            'network_segment': np.random.choice(['WardB'], 1000),
            'heart_rate': np.random.randint(60, 100, 1000),
            'spo2': np.random.randint(95, 100, 1000),
            'systolic_bp': np.random.randint(110, 140, 1000),
            'diastolic_bp': np.random.randint(60, 90, 1000),
            'temperature': np.random.uniform(36.5, 37.5, 1000),
            'respiratory_rate': np.random.randint(12, 20, 1000),
            'glucose': np.random.randint(70, 120, 1000),
            'label': 'MITM-ARPspoof'
        })
        
        # Data injection attack
        injection_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-03', periods=500, freq='1s'),
            'src_ip': np.random.choice(['10.0.0.' + str(i) for i in range(1, 10)], 500),
            'dst_ip': np.random.choice(['192.168.3.' + str(i) for i in range(20, 30)], 500),
            'src_port': np.random.randint(1024, 65535, 500),
            'dst_port': np.random.choice([80, 443, 8080], 500),
            'protocol': np.random.choice(['TCP', 'UDP'], 500),
            'packet_count': np.random.randint(1, 100, 500),
            'byte_volume': np.random.randint(100, 10000, 500),
            'flow_duration': np.random.uniform(0.1, 10.0, 500),
            'device_type': np.random.choice(['ECGSensor', 'SpO2Sensor', 'BPMonitor'], 500),
            'network_segment': np.random.choice(['WardB'], 500),
            # Anomalous biometric values
            'heart_rate': np.random.randint(30, 200, 500),  # Abnormal range
            'spo2': np.random.randint(70, 100, 500),        # Abnormal range
            'systolic_bp': np.random.randint(80, 200, 500), # Abnormal range
            'diastolic_bp': np.random.randint(40, 120, 500),# Abnormal range
            'temperature': np.random.uniform(35.0, 40.0, 500), # Abnormal range
            'respiratory_rate': np.random.randint(8, 30, 500), # Abnormal range
            'glucose': np.random.randint(40, 300, 500),     # Abnormal range
            'label': 'MITM-dataInject'
        })
        
        # Combine all dataframes
        combined_df = pd.concat([benign_df, mitm_df, injection_df], ignore_index=True)
        
        # Add dataset source column
        combined_df['source'] = 'WUSTL-EHMS-2020'
        
        return combined_df
    
    def _merge_datasets(self, ciciomt_df, iot_icu_df, wustl_df):
        """
        Merge the three datasets with schema alignment
        
        Args:
            ciciomt_df (pd.DataFrame): CICIoMT dataset
            iot_icu_df (pd.DataFrame): IoT-ICU dataset
            wustl_df (pd.DataFrame): WUSTL-EHMS dataset
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        # Identify common network flow features across all datasets
        common_features = [
            'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 
            'protocol', 'packet_count', 'byte_volume', 'flow_duration',
            'device_type', 'network_segment', 'label', 'source'
        ]
        
        # Ensure all datasets have these columns
        for df in [ciciomt_df, iot_icu_df, wustl_df]:
            for feature in common_features:
                if feature not in df.columns:
                    df[feature] = None
        
        # For biometric features from WUSTL, add them to other datasets as null
        biometric_features = [
            'heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp',
            'temperature', 'respiratory_rate', 'glucose'
        ]
        
        for feature in biometric_features:
            if feature not in ciciomt_df.columns:
                ciciomt_df[feature] = None
            if feature not in iot_icu_df.columns:
                iot_icu_df[feature] = None
        
        # For IoT-specific features, add them to other datasets as null
        iot_features = ['mqtt_topic', 'ble_signal_strength']
        
        for feature in iot_features:
            if feature not in ciciomt_df.columns:
                ciciomt_df[feature] = None
            if feature not in iot_icu_df.columns:
                iot_icu_df[feature] = None
            if feature not in wustl_df.columns:
                wustl_df[feature] = None
        
        # Concatenate all datasets
        merged_df = pd.concat([ciciomt_df, iot_icu_df, wustl_df], ignore_index=True)
        
        # Harmonize attack labels
        attack_mapping = {
            'Normal': 'Normal',
            'DDoS-UDPFlood': 'DDoS-UDPFlood',
            'Reconnaissance-Scan': 'Reconnaissance-Scan',
            'MQTT-slowite': 'MQTT-slowite',
            'MITM-ARPspoof': 'MITM-ARPspoof',
            'MITM-dataInject': 'Injection-FakeVitals'
        }
        
        merged_df['label'] = merged_df['label'].map(attack_mapping)
        
        return merged_df
    
    def _preprocess_merged_data(self):
        """
        Preprocess the merged dataset
        """
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.merged_data['timestamp']):
            self.merged_data['timestamp'] = pd.to_datetime(self.merged_data['timestamp'])
        
        # Extract time-based features
        self.merged_data['hour'] = self.merged_data['timestamp'].dt.hour
        self.merged_data['day_of_week'] = self.merged_data['timestamp'].dt.dayofweek
        
        # Convert categorical features using label encoding
        categorical_features = ['protocol', 'device_type', 'network_segment', 'src_ip', 'dst_ip']
        self.label_encoders = {}
        
        for feature in categorical_features:
            le = LabelEncoder()
            self.merged_data[feature] = le.fit_transform(self.merged_data[feature].astype(str))
            self.label_encoders[feature] = le
        
        # Handle MQTT and BLE specific features
        self.merged_data['has_mqtt'] = self.merged_data['mqtt_topic'].notna().astype(int)
        self.merged_data['has_ble'] = self.merged_data['ble_signal_strength'].notna().astype(int)
        
        if 'mqtt_topic' in self.merged_data.columns:
            mqtt_le = LabelEncoder()
            self.merged_data['mqtt_topic_encoded'] = mqtt_le.fit_transform(
                self.merged_data['mqtt_topic'].fillna('none').astype(str)
            )
            self.label_encoders['mqtt_topic'] = mqtt_le
        
        # Fill missing values in numeric columns
        numeric_features = [
            'packet_count', 'byte_volume', 'flow_duration', 
            'ble_signal_strength', 'heart_rate', 'spo2', 
            'systolic_bp', 'diastolic_bp', 'temperature', 
            'respiratory_rate', 'glucose'
        ]
        
        for feature in numeric_features:
            if feature in self.merged_data.columns:
                # Fill with median of normal traffic for that feature
                normal_median = self.merged_data.loc[
                    self.merged_data['label'] == 'Normal', feature
                ].median()
                
                self.merged_data[feature] = self.merged_data[feature].fillna(normal_median)
        
        # Encode cyclical time features
        self.merged_data['hour_sin'] = np.sin(2 * np.pi * self.merged_data['hour'] / 24)
        self.merged_data['hour_cos'] = np.cos(2 * np.pi * self.merged_data['hour'] / 24)
        self.merged_data['day_sin'] = np.sin(2 * np.pi * self.merged_data['day_of_week'] / 7)
        self.merged_data['day_cos'] = np.cos(2 * np.pi * self.merged_data['day_of_week'] / 7)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.merged_data['label_encoded'] = self.label_encoder.fit_transform(self.merged_data['label'])
        self.num_classes = len(self.label_encoder.classes_)
        
        # Select features for model training
        self.selected_features = [
            'src_port', 'dst_port', 'protocol', 'packet_count', 'byte_volume', 
            'flow_duration', 'device_type', 'network_segment', 'hour_sin', 
            'hour_cos', 'day_sin', 'day_cos', 'has_mqtt', 'has_ble'
        ]
        
        # Add biometric features if available
        biometric_features = [
            'heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp',
            'temperature', 'respiratory_rate', 'glucose'
        ]
        
        for feature in biometric_features:
            if feature in self.merged_data.columns:
                self.selected_features.append(feature)
        
        # Feature selection if enabled
        if self.feature_selection:
            # In a real implementation, this would use Random Forest importance
            # For this example, we'll just use a subset of features
            self.selected_features = self.selected_features[:30]  # Limit to top 30 features
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        self.merged_data[self.selected_features] = self.scaler.fit_transform(
            self.merged_data[self.selected_features]
        )
        
        # Create device IDs for GNN if needed
        if self.use_gnn:
            # Create a unique ID for each device based on IP and device type
            self.merged_data['device_id'] = (
                self.merged_data['src_ip'].astype(str) + '_' + 
                self.merged_data['device_type'].astype(str)
            ).astype('category').cat.codes
            
            # Create a mapping of connections between devices
            self.device_connections = self._create_device_connections()
        
    def _create_device_connections(self):
        """
        Create a graph of device connections for GNN
        
        Returns:
            dict: Dictionary mapping device IDs to connected devices
        """
        connections = {}
        
        # Group by source IP and find all destinations
        for src_device in self.merged_data['device_id'].unique():
            # Find all devices this source has communicated with
            dst_devices = self.merged_data[
                self.merged_data['device_id'] == src_device
            ]['dst_ip'].unique()
            
            # Map these destination IPs to device IDs
            connected_devices = self.merged_data[
                self.merged_data['src_ip'].isin(dst_devices)
            ]['device_id'].unique()
            
            connections[src_device] = connected_devices
        
        return connections
    
    def _create_data_splits(self):
        """
        Create train/validation/test splits
        """
        # Extract features and labels
        X = self.merged_data[self.selected_features].values
        y = self.merged_data['label_encoded'].values
        
        # Create device IDs array if using GNN
        if self.use_gnn:
            device_ids = self.merged_data['device_id'].values
        else:
            device_ids = None
        
        # Stratified split: 60% train, 20% validation, 20% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Handle class imbalance with SMOTE (only on training data)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Split device IDs if using GNN
        if self.use_gnn:
            device_train, device_temp = train_test_split(
                device_ids, test_size=0.4, random_state=42, stratify=y
            )
            
            device_val, device_test = train_test_split(
                device_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # For SMOTE-resampled data, we need to assign device IDs
            # This is a simplification; in practice, you'd need a more sophisticated approach
            device_train_resampled = np.zeros_like(y_train_resampled)
            for i, label in enumerate(y_train_resampled):
                # Assign a random device ID from the same class
                same_class_indices = np.where(y_train == label)[0]
                if len(same_class_indices) > 0:
                    random_idx = np.random.choice(same_class_indices)
                    device_train_resampled[i] = device_train[random_idx]
                else:
                    device_train_resampled[i] = np.random.choice(device_train)
        else:
            device_train_resampled = None
            device_val = None
            device_test = None
        
        # Create datasets
        self.train_dataset = MedFuseDataset(
            X_train_resampled, y_train_resampled, device_train_resampled, 
            self.sequence_length
        )
        
        self.val_dataset = MedFuseDataset(
            X_val, y_val, device_val, self.sequence_length
        )
        
        self.test_dataset = MedFuseDataset(
            X_test, y_test, device_test, self.sequence_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        # Create GNN data if needed
        if self.use_gnn:
            self.gnn_data = self._create_gnn_data()
    
    def _create_gnn_data(self):
        """
        Create PyTorch Geometric data objects for GNN
        
        Returns:
            dict: Dictionary of PyTorch Geometric data objects
        """
        # Create a graph for each network segment
        gnn_data = {}
        
        for segment in self.merged_data['network_segment'].unique():
            # Get devices in this segment
            segment_devices = self.merged_data[
                self.merged_data['network_segment'] == segment
            ]['device_id'].unique()
            
            # Create node features (average of all records for each device)
            node_features = []
            for device in segment_devices:
                device_records = self.merged_data[
                    self.merged_data['device_id'] == device
                ][self.selected_features].mean().values
                
                node_features.append(device_features)
            
            node_features = torch.FloatTensor(node_features)
            
            # Create edges (connections between devices)
            edge_index = []
            for i, src_device in enumerate(segment_devices):
                if src_device in self.device_connections:
                    for dst_device in self.device_connections[src_device]:
                        if dst_device in segment_devices:
                            j = np.where(segment_devices == dst_device)[0][0]
                            edge_index.append([i, j])
            
            if not edge_index:  # If no edges, create self-loops
                edge_index = [[i, i] for i in range(len(segment_devices))]
            
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            
            # Create PyTorch Geometric data object
            data = gdata.Data(
                x=node_features,
                edge_index=edge_index,
                device_ids=torch.LongTensor(segment_devices)
            )
            
            gnn_data[segment] = data
        
        return gnn_data
    
    def get_loaders(self):
        """
        Get data loaders
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_gnn_data(self):
        """
        Get GNN data
        
        Returns:
            dict: Dictionary of PyTorch Geometric data objects
        """
        if not self.use_gnn:
            raise ValueError("GNN data not available. Initialize with use_gnn=True")
        
        return self.gnn_data
    
    def get_num_classes(self):
        """
        Get number of classes
        
        Returns:
            int: Number of classes
        """
        return self.num_classes
    
    def get_feature_dim(self):
        """
        Get feature dimension
        
        Returns:
            int: Feature dimension
        """
        return len(self.selected_features)
    
    def get_class_names(self):
        """
        Get class names
        
        Returns:
            list: List of class names
        """
        return self.label_encoder.classes_.tolist()
