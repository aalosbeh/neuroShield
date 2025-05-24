import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class AttackerProfiler:
    """
    AI-Psychometric profiling module for NeuroShield++
    
    This module analyzes attack patterns to create a psychological profile of attackers
    based on a 5-dimensional model inspired by the OCEAN personality framework.
    """
    def __init__(self, momentum=0.7, profile_dim=5, initial_profile=None):
        """
        Initialize the attacker profiler
        
        Args:
            momentum (float): Momentum term for profile updates (0-1)
            profile_dim (int): Dimension of the profile vector (default: 5 for OCEAN-inspired model)
            initial_profile (np.ndarray, optional): Initial profile vector
        """
        self.momentum = momentum
        self.profile_dim = profile_dim
        
        # Initialize profile vector with neutral values if not provided
        if initial_profile is None:
            self.profile = np.ones(profile_dim) * 0.5
        else:
            self.profile = initial_profile
            
        # Initialize transformation matrix for mapping behavioral features to profile dimensions
        # This matrix is based on expert knowledge and can be refined through empirical testing
        self.W_p = np.array([
            # Openness: affected by attack diversity, technique sophistication
            [0.8, 0.6, 0.2, 0.1, -0.3, 0.7, 0.5],
            # Conscientiousness: affected by stealth, precision, consistency
            [0.1, -0.7, 0.2, 0.8, 0.6, 0.1, 0.7],
            # Extraversion: affected by targeting breadth, attack volume
            [0.2, 0.9, 0.8, -0.2, -0.1, 0.1, 0.0],
            # Agreeableness: affected by target selection, damage potential
            [-0.6, -0.3, -0.5, 0.2, 0.7, -0.8, -0.4],
            # Neuroticism: affected by pattern consistency, timing irregularity
            [0.3, 0.1, -0.2, -0.6, -0.4, 0.2, -0.5]
        ])
        
        # Bias term for profile update
        self.b_p = np.zeros(profile_dim)
        
        # Scaler for normalizing behavioral features
        self.feature_scaler = MinMaxScaler()
        
        # Initialize with some sample data to fit the scaler
        sample_features = np.random.rand(10, 7)
        self.feature_scaler.fit(sample_features)
        
        # History of profile updates for analysis
        self.profile_history = [self.profile.copy()]
        
        # Attack history for pattern analysis
        self.attack_history = []
        
    def extract_behavioral_features(self, attack_data):
        """
        Extract behavioral features from attack data
        
        Args:
            attack_data (dict): Dictionary containing attack information
                - attack_type (str): Type of attack
                - packet_count (int): Number of packets
                - target_count (int): Number of targets
                - critical_target (bool): Whether critical systems were targeted
                - evasion_techniques (list): List of evasion techniques used
                - payload_complexity (float): Complexity of attack payload
                - timing_pattern (str): Timing pattern of the attack
                
        Returns:
            np.ndarray: Behavioral feature vector
        """
        # Extract relevant features
        attack_type = attack_data.get('attack_type', 'unknown')
        packet_count = attack_data.get('packet_count', 0)
        target_count = attack_data.get('target_count', 1)
        critical_target = 1 if attack_data.get('critical_target', False) else 0
        evasion_count = len(attack_data.get('evasion_techniques', []))
        payload_complexity = attack_data.get('payload_complexity', 0.5)
        
        # Analyze timing pattern
        timing_pattern = attack_data.get('timing_pattern', 'regular')
        if timing_pattern == 'regular':
            timing_score = 0.3
        elif timing_pattern == 'irregular':
            timing_score = 0.7
        elif timing_pattern == 'random':
            timing_score = 0.9
        else:
            timing_score = 0.5
            
        # Calculate attack vector diversity based on history
        self.attack_history.append(attack_type)
        unique_attacks = len(set(self.attack_history[-10:]))  # Consider last 10 attacks
        attack_diversity = unique_attacks / min(10, len(self.attack_history))
        
        # Construct behavioral feature vector
        behavioral_features = np.array([
            attack_diversity,              # Diversity of attack vectors (Openness)
            packet_count,                  # Volume of attack (Extraversion)
            target_count,                  # Breadth of targeting (Extraversion)
            1.0 - critical_target,         # Avoidance of critical systems (Agreeableness)
            evasion_count,                 # Stealth techniques used (Conscientiousness)
            timing_score,                  # Timing irregularity (Neuroticism)
            payload_complexity             # Sophistication of payload (Openness, Conscientiousness)
        ])
        
        # Normalize features
        behavioral_features = self.feature_scaler.transform(behavioral_features.reshape(1, -1)).flatten()
        
        return behavioral_features
    
    def update_profile(self, attack_data):
        """
        Update the attacker profile based on observed attack data
        
        Args:
            attack_data (dict): Dictionary containing attack information
            
        Returns:
            np.ndarray: Updated profile vector
        """
        # Extract behavioral features
        behavioral_features = self.extract_behavioral_features(attack_data)
        
        # Update profile using the transformation matrix and momentum
        profile_update = self.sigmoid(np.dot(self.W_p, behavioral_features) + self.b_p)
        self.profile = self.momentum * self.profile + (1 - self.momentum) * profile_update
        
        # Ensure profile values are in [0, 1]
        self.profile = np.clip(self.profile, 0, 1)
        
        # Save to history
        self.profile_history.append(self.profile.copy())
        
        return self.profile
    
    def get_profile(self):
        """
        Get the current attacker profile
        
        Returns:
            dict: Dictionary containing profile dimensions and values
        """
        profile_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        return {name: value for name, value in zip(profile_names, self.profile)}
    
    def get_profile_history(self):
        """
        Get the history of profile updates
        
        Returns:
            pd.DataFrame: DataFrame containing profile history
        """
        profile_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        history_df = pd.DataFrame(self.profile_history, columns=profile_names)
        return history_df
    
    def adjust_detection_threshold(self, base_threshold, attack_class):
        """
        Adjust detection threshold based on attacker profile
        
        Args:
            base_threshold (float): Base detection threshold
            attack_class (str): Attack class
            
        Returns:
            float: Adjusted threshold
        """
        # Different attack classes are affected differently by profile dimensions
        # These weights represent how each profile dimension affects the threshold
        # for different attack classes
        class_weights = {
            'DDoS': [0.1, -0.2, 0.5, -0.3, 0.1],    # Extraversion strongly affects DDoS threshold
            'Reconnaissance': [0.3, 0.5, -0.1, 0.0, 0.2],  # Conscientiousness affects Recon
            'MITM': [0.2, 0.4, 0.1, -0.2, -0.1],    # Conscientiousness affects MITM
            'Injection': [0.4, 0.3, 0.0, -0.3, 0.1], # Openness affects Injection
            'Spoofing': [0.2, 0.5, 0.1, -0.1, 0.0],  # Conscientiousness affects Spoofing
            'Malware': [0.3, 0.2, 0.1, -0.2, 0.3]    # Openness and Neuroticism affect Malware
        }
        
        # Default weights if attack class not found
        weights = class_weights.get(attack_class, [0.2, 0.2, 0.2, -0.2, 0.2])
        
        # Calculate adjustment factor
        adjustment = sum(w * p for w, p in zip(weights, self.profile))
        
        # Scale adjustment to reasonable range (-30% to +30%)
        scaled_adjustment = 0.3 * (2 * adjustment - 1)
        
        # Apply adjustment to base threshold
        adjusted_threshold = base_threshold * (1 + scaled_adjustment)
        
        # Ensure threshold is in reasonable range
        adjusted_threshold = max(0.01, min(0.99, adjusted_threshold))
        
        return adjusted_threshold
    
    def predict_next_attack(self):
        """
        Predict the most likely next attack based on profile and history
        
        Returns:
            dict: Dictionary containing predicted attack types and probabilities
        """
        # This is a simplified heuristic approach
        # In a real implementation, this could use more sophisticated ML models
        
        # Define attack tendencies based on profile dimensions
        attack_tendencies = {
            'DDoS': 0.2 * self.profile[0] + 0.1 * self.profile[1] + 0.5 * self.profile[2] - 0.2 * self.profile[3] + 0.1 * self.profile[4],
            'Reconnaissance': 0.4 * self.profile[0] + 0.3 * self.profile[1] - 0.1 * self.profile[2] + 0.0 * self.profile[3] + 0.1 * self.profile[4],
            'MITM': 0.3 * self.profile[0] + 0.4 * self.profile[1] + 0.0 * self.profile[2] - 0.2 * self.profile[3] + 0.0 * self.profile[4],
            'Injection': 0.5 * self.profile[0] + 0.2 * self.profile[1] + 0.0 * self.profile[2] - 0.3 * self.profile[3] + 0.1 * self.profile[4],
            'Spoofing': 0.2 * self.profile[0] + 0.5 * self.profile[1] + 0.1 * self.profile[2] - 0.1 * self.profile[3] + 0.0 * self.profile[4],
            'Malware': 0.4 * self.profile[0] + 0.2 * self.profile[1] + 0.0 * self.profile[2] - 0.2 * self.profile[3] + 0.3 * self.profile[4]
        }
        
        # Consider attack history (attackers often repeat successful patterns)
        if len(self.attack_history) > 0:
            last_attack = self.attack_history[-1]
            for attack_type in attack_tendencies:
                if last_attack.startswith(attack_type):
                    attack_tendencies[attack_type] += 0.2
        
        # Normalize to get probabilities
        total = sum(attack_tendencies.values())
        attack_probabilities = {k: v/total for k, v in attack_tendencies.items()}
        
        # Sort by probability
        sorted_predictions = dict(sorted(attack_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_predictions
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output array after sigmoid activation
        """
        return 1 / (1 + np.exp(-x))
    
    def get_profile_description(self):
        """
        Get a human-readable description of the attacker profile
        
        Returns:
            str: Description of attacker profile
        """
        profile = self.get_profile()
        
        # Determine dominant traits (top 2)
        sorted_traits = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        dominant_traits = sorted_traits[:2]
        
        # Generate description based on dominant traits
        descriptions = {
            'Openness': [
                "shows high diversity in attack vectors",
                "demonstrates technical sophistication",
                "adapts tactics frequently"
            ],
            'Conscientiousness': [
                "exhibits methodical, careful attack patterns",
                "employs stealthy techniques to avoid detection",
                "shows precision in targeting"
            ],
            'Extraversion': [
                "launches broad, high-volume attacks",
                "targets multiple systems simultaneously",
                "prefers noisy, disruptive techniques"
            ],
            'Agreeableness': [
                "avoids targeting critical patient-care systems",
                "minimizes potential for physical harm",
                "focuses on data theft over service disruption"
            ],
            'Neuroticism': [
                "displays erratic, unpredictable patterns",
                "changes tactics frequently",
                "shows inconsistent timing in attacks"
            ]
        }
        
        # Generate description
        description = "Attacker profile analysis: This adversary "
        
        # Add descriptions for dominant traits
        for trait, value in dominant_traits:
            if value > 0.7:
                intensity = "strongly "
            elif value > 0.5:
                intensity = ""
            else:
                continue  # Skip traits with low values
                
            description += intensity + np.random.choice(descriptions[trait]) + " and "
        
        # Remove trailing "and " and add period
        if description.endswith(" and "):
            description = description[:-5] + "."
        
        # Add prediction
        next_attack = self.predict_next_attack()
        top_attack = list(next_attack.keys())[0]
        description += f" Based on this profile, the system anticipates potential {top_attack} attacks with heightened vigilance."
        
        return description
