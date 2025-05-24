import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import SAGEConv, global_mean_pool

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence modeling in NeuroShield++
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=8, dropout=0.1):
        """
        Initialize the Transformer encoder
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension size
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output dimension is hidden_dim * 2 (we concatenate mean and max pooling)
        self.output_dim = hidden_dim * 2
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling: concatenate mean and max pooling
        mean_pooled = torch.mean(x, dim=1)
        max_pooled = torch.max(x, dim=1)[0]
        
        # Concatenate pooled features
        output = torch.cat([mean_pooled, max_pooled], dim=1)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model (int): Model dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for spatial feature interactions in NeuroShield++
    """
    def __init__(self, input_dim, hidden_dims=[32, 64], kernel_size=3):
        """
        Initialize the CNN feature extractor
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dims (list): List of hidden dimensions for CNN layers
            kernel_size (int): Kernel size for convolutions
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Create a sequence of 1D convolution layers
        layers = []
        in_channels = 1  # Start with 1 channel (feature vector as 1D signal)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = hidden_dim
        
        self.cnn_layers = nn.Sequential(*layers)
        
        # Calculate output dimension
        self.output_dim = hidden_dims[-1] * (input_dim // (2 ** len(hidden_dims)))
        
        # If output dimension is too large, add a linear projection
        if self.output_dim > 128:
            self.projection = nn.Linear(self.output_dim, 64)
            self.output_dim = 64
        else:
            self.projection = None
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Reshape input to (batch_size, channels, input_dim)
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply projection if needed
        if self.projection is not None:
            x = self.projection(x)
        
        return x


class GNNModule(nn.Module):
    """
    Graph Neural Network module for device-relational context in NeuroShield++
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        """
        Initialize the GNN module
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension size
            num_layers (int): Number of GNN layers
        """
        super(GNNModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GraphSAGE layers
        self.conv_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.conv_layers.append(SAGEConv(input_dim, hidden_dim))
        
        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output dimension is hidden_dim
        self.output_dim = hidden_dim
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Node feature tensor of shape (num_nodes, input_dim)
            edge_index (torch.Tensor): Graph connectivity in COO format
            batch (torch.Tensor, optional): Batch vector for multiple graphs
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Apply GNN layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        # If batch is provided, perform graph-level pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class NeuroShieldPlusPlus(nn.Module):
    """
    Complete NeuroShield++ model with Transformer, CNN, and optional GNN
    """
    def __init__(self, input_dim, num_classes, seq_length=10, 
                 use_transformer=True, use_cnn=True, use_gnn=False,
                 transformer_hidden_dim=64, cnn_hidden_dims=[32, 64], 
                 gnn_hidden_dim=64, fusion_hidden_dims=[128, 64]):
        """
        Initialize the NeuroShield++ model
        
        Args:
            input_dim (int): Input feature dimension
            num_classes (int): Number of output classes
            seq_length (int): Sequence length for Transformer
            use_transformer (bool): Whether to use Transformer branch
            use_cnn (bool): Whether to use CNN branch
            use_gnn (bool): Whether to use GNN branch
            transformer_hidden_dim (int): Hidden dimension for Transformer
            cnn_hidden_dims (list): Hidden dimensions for CNN
            gnn_hidden_dim (int): Hidden dimension for GNN
            fusion_hidden_dims (list): Hidden dimensions for fusion layers
        """
        super(NeuroShieldPlusPlus, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.use_transformer = use_transformer
        self.use_cnn = use_cnn
        self.use_gnn = use_gnn
        
        # Initialize branches
        self.branches = []
        self.output_dims = []
        
        if use_transformer:
            self.transformer = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=transformer_hidden_dim
            )
            self.branches.append(self.transformer)
            self.output_dims.append(self.transformer.output_dim)
        
        if use_cnn:
            self.cnn = CNNFeatureExtractor(
                input_dim=input_dim,
                hidden_dims=cnn_hidden_dims
            )
            self.branches.append(self.cnn)
            self.output_dims.append(self.cnn.output_dim)
        
        if use_gnn:
            self.gnn = GNNModule(
                input_dim=input_dim,
                hidden_dim=gnn_hidden_dim
            )
            self.branches.append(self.gnn)
            self.output_dims.append(self.gnn.output_dim)
        
        # Total output dimension from all branches
        self.total_output_dim = sum(self.output_dims)
        
        # Fusion layers
        fusion_layers = []
        in_dim = self.total_output_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(in_dim, hidden_dim))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        
        self.fusion_layers = nn.Sequential(*fusion_layers)
        
        # Classification layer
        self.classifier = nn.Linear(fusion_hidden_dims[-1], num_classes)
        
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data (dict): Input data dictionary containing:
                - 'sequence': Tensor of shape (batch_size, seq_length, input_dim) for Transformer
                - 'features': Tensor of shape (batch_size, input_dim) for CNN
                - 'graph': PyTorch Geometric Data object for GNN (if use_gnn=True)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        branch_outputs = []
        
        # Transformer branch
        if self.use_transformer and 'sequence' in data:
            transformer_out = self.transformer(data['sequence'])
            branch_outputs.append(transformer_out)
        
        # CNN branch
        if self.use_cnn and 'features' in data:
            cnn_out = self.cnn(data['features'])
            branch_outputs.append(cnn_out)
        
        # GNN branch
        if self.use_gnn and 'graph' in data:
            x, edge_index, batch = data['graph'].x, data['graph'].edge_index, data['graph'].batch
            gnn_out = self.gnn(x, edge_index, batch)
            branch_outputs.append(gnn_out)
        
        # Concatenate branch outputs
        if len(branch_outputs) > 1:
            combined = torch.cat(branch_outputs, dim=1)
        else:
            combined = branch_outputs[0]
        
        # Apply fusion layers
        fused = self.fusion_layers(combined)
        
        # Apply classifier
        logits = self.classifier(fused)
        
        return logits
    
    def predict(self, data):
        """
        Make predictions
        
        Args:
            data (dict): Input data dictionary
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        logits = self.forward(data)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        return predicted_class, probabilities
