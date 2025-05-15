import numpy as np
import torch
import torch.nn as nn
from decision_transformer.models.model import TrajectoryModel

class MLPBCModel(TrajectoryModel):
    """
    Enhanced MLP that predicts next action a from past states s with better temporal context handling.
    """
    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Add positional encoding to preserve temporal information
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, state_dim))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.1)
        
        # First process each state individually
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Then process the sequence
        layers = [nn.Linear(max_length * hidden_size, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])
        self.model = nn.Sequential(*layers)
        
    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):
        # Use only the most recent max_length states
        states = states[:, -self.max_length:]
        batch_size, seq_len, _ = states.shape
        
        # Add positional encodings to preserve temporal information
        if seq_len < self.max_length:
            # If shorter than max_length, use the corresponding subset of positional encodings
            pos_enc = self.positional_encoding[:, -seq_len:]
        else:
            pos_enc = self.positional_encoding
            
        states = states + pos_enc[:, :seq_len]
        
        # Encode each state
        encoded_states = self.state_encoder(states)
        
        # Flatten and process through main model
        states_flat = encoded_states.reshape(batch_size, -1)  # Keep variable name similar but encode better
        actions = self.model(states_flat).reshape(batch_size, 1, self.act_dim)
        
        return None, actions, None
        
    def get_action(self, states, actions, rewards, target_return=None, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        
        # Handle sequence length - use consistent padding strategy
        if states.shape[1] > self.max_length:
            # If too long, use only the most recent states
            states = states[:, -self.max_length:]
        elif states.shape[1] < self.max_length:
            # If too short, pad with zeros at the beginning
            # This is kept the same as your original code for consistency
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                           dtype=torch.float32, device=states.device), states], dim=1)
        
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]