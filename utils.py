import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the token embeddings.
        """
        super(TokenEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Optional: Add a dropout layer if needed
        # self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the embedding layer.

        Args:
            token_ids: Tensor of token ids with shape [batch_size, seq_length].
        
        Returns:
            out: Tensor of embeddings with shape [batch_size, seq_length, embed_dim].
        """
        return self.embed(token_ids)
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, embed_model_dim: int):
        """
        Args:
            max_seq_len: The maximum length of the input sequences.
            embed_model_dim: The dimension of the embedding.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim

        # Create a position tensor
        position = torch.arange(max_seq_len).unsqueeze(1)
        # Compute the div term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, embed_model_dim, 2) * \
                             -(math.log(10000.0) / embed_model_dim))
        
        # Initialize a zeros tensor for positional encoding
        pe = torch.zeros(max_seq_len, embed_model_dim)

        # Apply the sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension to the positional encoding tensor
        pe = pe.unsqueeze(0)

        # Register pe as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape [batch_size, seq_length, embed_dim]
        Returns:
            Tensor with positional embeddings added to the original embedding.
        """
        # Scale embeddings by the square root of the embedding dimension
        x = x * math.sqrt(self.embed_dim)

        # Add positional encoding to the input tensor
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x
    



