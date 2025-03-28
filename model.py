import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAttention(nn.Module):
    """GRU with attention mechanism for better focus on important patterns"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layer
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        attention_dim = hidden_dim * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.Tanh(),
            nn.Linear(attention_dim // 2, 1)
        )
        
        # Output linear layer to reduce dimensionality if bidirectional
        if bidirectional:
            self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x, return_attention=False):
        # x shape: [batch, seq_len, input_dim]
        
        # Run GRU
        gru_output, hidden = self.gru(x)  
        # gru_output shape: [batch, seq_len, hidden_dim*num_directions]
        
        # Calculate attention scores
        attention_scores = self.attention(gru_output)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
        
        # Apply attention to get context vector
        context = torch.sum(gru_output * attention_weights, dim=1)  # [batch, hidden_dim*num_directions]
        
        # Project back to hidden_dim if bidirectional
        if self.num_directions > 1:
            context = self.output_projection(context)
        
        if return_attention:
            return context, attention_weights
        return context

class HierarchicalGRUModel(nn.Module):
    """
    Hierarchical GRU model with attention for handling multivariate time-series data.
    Processes each feature group independently then integrates them.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract model parameters
        hidden_dim = config['model_params'].get('hidden_dim', 96)
        dropout = config['model_params'].get('dropout', 0.2)
        use_layernorm = config['model_params'].get('use_layernorm', True)
        bidirectional = config['model_params'].get('bidirectional', True)
        num_gru_layers = config['model_params'].get('num_gru_layers', 2)
        
        # Feature parameters
        self.feature_groups = config['feature_generation']['feature_groups']
        self.lookback = config['feature_generation']['feature_width']
        self._calculate_feature_indices()
        
        # Output ranges
        self.pred_range = config['model_params'].get('prediction_range', [0.35, 0.65])
        self.conf_range = config['model_params'].get('confidence_range', [0.25, 0.75])
        
        # Feature normalization
        self.feature_norms = nn.ModuleDict()
        for group_name, features in self.feature_groups.items():
            if len(features) > 0:
                self.feature_norms[group_name] = nn.LayerNorm(len(features))
        
        # Feature embeddings
        self.feature_embedders = nn.ModuleDict()
        for group_name, features in self.feature_groups.items():
            if len(features) > 0:
                self.feature_embedders[group_name] = nn.Sequential(
                    nn.Linear(len(features), hidden_dim),
                    nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout)
                )
        
        # Group-specific GRUs with attention
        self.group_grus = nn.ModuleDict()
        for group_name, features in self.feature_groups.items():
            if len(features) > 0:
                self.group_grus[group_name] = GRUAttention(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_gru_layers,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
        
        # Hierarchical GRU for integrating group outputs
        num_groups = len([g for g, f in self.feature_groups.items() if len(f) > 0])
        if num_groups > 1:
            self.hierarchical_gru = GRUAttention(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=num_gru_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        
        # Final integration with residual connections
        self.feature_integrator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2) if use_layernorm else nn.Identity(),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout/2)
        )
        
        # Output heads
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu', a=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def _calculate_feature_indices(self):
        """Calculate feature indices for each group"""
        self.feature_indices = {}
        current_idx = 0
        
        for group_name, features in self.feature_groups.items():
            n_features = len(features)
            if n_features > 0:
                self.feature_indices[group_name] = (current_idx, current_idx + n_features)
                current_idx += n_features
    
    def _scale_output(self, x, output_range):
        """Scale sigmoid output to the specified range"""
        low, high = output_range
        return low + (high - low) * x
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Process each feature group independently with GRUs
        group_outputs = []
        group_attentions = {}
        
        for group_name, (start_idx, end_idx) in self.feature_indices.items():
            if group_name in self.feature_embedders:
                # Extract features for this group
                group_features = x[:, :, start_idx:end_idx]  # [batch, seq, features]
                
                # Normalize features
                normalized_features = []
                for t in range(seq_len):
                    normalized_features.append(self.feature_norms[group_name](group_features[:, t]))
                normalized_sequence = torch.stack(normalized_features, dim=1)  # [batch, seq, features]
                
                # Embed features
                embedded_sequence = torch.stack([
                    self.feature_embedders[group_name](normalized_sequence[:, t])
                    for t in range(seq_len)
                ], dim=1)
                
                # Process with GRU and attention
                context, attention = self.group_grus[group_name](embedded_sequence, return_attention=True)
                
                # Store outputs
                group_outputs.append(context)
                group_attentions[group_name] = attention
        
        # Integrate group outputs with hierarchical attention
        if len(group_outputs) > 1 and hasattr(self, 'hierarchical_gru'):
            # Stack group outputs for hierarchical processing
            stacked_outputs = torch.stack(group_outputs, dim=1)  # [batch, num_groups, hidden]
            integrated_context = self.hierarchical_gru(stacked_outputs)
        elif len(group_outputs) == 1:
            integrated_context = group_outputs[0]
        else:
            # Handle edge case with no features
            integrated_context = torch.zeros(batch_size, self.config['model_params'].get('hidden_dim', 96), device=x.device)
        
        # Final feature integration
        final_features = self.feature_integrator(integrated_context)
        
        # Generate prediction probability
        pred_logits = self.pred_head(final_features).squeeze(-1)
        probs = torch.sigmoid(pred_logits)
        scaled_probs = self._scale_output(probs, self.pred_range)
        
        # Generate confidence score
        conf_logits = self.conf_head(final_features).squeeze(-1)
        confidence = torch.sigmoid(conf_logits)
        scaled_confidence = self._scale_output(confidence, self.conf_range)
        
        # Binary predictions based on threshold
        predictions = (scaled_probs > 0.5).float()
        
        return scaled_probs, scaled_confidence, predictions
    
    
class StandardBinaryLoss(nn.Module):
	"""
	Standard binary classification loss function using binary cross-entropy.
	
	This is a simple baseline replacement for the EnhancedVWAPLoss.
	"""
	def __init__(self, config=None):
		super().__init__()
		if config and 'loss_params' in config:
			params = config['loss_params']
			self.pos_weight = params.get('pos_weight', 1.0)  # Weight for positive class
			self.use_weighted_loss = params.get('use_weighted_loss', False)
		else:
			self.pos_weight = 1.0
			self.use_weighted_loss = False
		
	def forward(self, probs, confidence, targets, return_components=False):
		# Basic binary cross-entropy loss
		if self.use_weighted_loss:
			# Create weights tensor based on targets
			weights = torch.ones_like(targets)
			weights[targets > 0.5] = self.pos_weight
			
			# Apply weighted BCE
			bce_loss = F.binary_cross_entropy(probs, targets, weight=weights)
		else:
			# Standard BCE
			bce_loss = F.binary_cross_entropy(probs, targets)
		
		# Simplified confidence loss (optional, can be ignored but keeping
		# the same interface as the original loss function)
		confidence_loss = torch.tensor(0.0, device=probs.device)
			
		if return_components:
			return {
				'total_loss': bce_loss,
				'bce_loss': bce_loss,
				'confidence_loss': confidence_loss
			}
		
		return bce_loss


def create_model(config, device):
	"""Create model and standard loss function"""
	model = HierarchicalGRUModel(config)
	model = model.to(device)
	
	criterion = StandardBinaryLoss(config)
	
	return model, criterion