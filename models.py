import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNormalization(nn.Module):
    def __init__(self, n_feats: int, eps: float = 1e-6):
        """
        Args:
            n_feats: Number of features. This is the size of the last dimension of the input.
            eps: Epsilon value to avoid division by zero.
        """
        super(LayerNormalization, self).__init__()
        
        # Define alpha and beta tensors as learnable parameters for each feature
        self.alpha = nn.Parameter(torch.ones(n_feats))
        self.beta = nn.Parameter(torch.zeros(n_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward path of the layer normalization.

        Args:
            x: The input feature tensor. Shape: [batch_size, seq_len, n_feat]

        Returs:
            Normalized tensor of the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True) # Shape: [batch_size, seq_len, 1]
        std = x.std(dim=-1, keepdim=True) # Shape: [batch_size, seq_len, 1]

        normalized_x = (x - mean) / (std + self.eps)

        return self.alpha * normalized_x + self.beta
    

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, d_ff: int = 2048, dropout: float = 0.2):
        """
        Args:
            embed_dim: The embedding dimension.
            d_ff: The intermediate feature size in the feedforward block.
            dropout: The dropout rate.
        """
        super(FeedForwardBlock, self).__init__()
        
        self.ffb = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardBlock.
        This block consists of two linear transformations with a ReLU activation in between.

        Args:
            x: Input tensor. Shape: [batch_size, seq_len, embed_dim].

        Returns:
            Output tensor. Shape: [batch_size, seq_len, embed_dim].
        """
        return self.ffb(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the token embeddings.
        """
        super(TokenEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the embedding layer.

        Args:
            token_ids: Tensor of token ids. Shape: [batch_size, seq_length].
        
        Returns:
            out: Tensor of embeddings. Shape [batch_size, seq_length, embed_dim].
        """
        return self.embed(token_ids)
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, embed_model_dim: int, dropout: float = 0.2):
        """
        Args:
            max_seq_len: The maximum length of the input sequences.
            embed_model_dim: The dimension of the embedding.
            droptout: The dropout rate.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_model_dim
        self.dropout = nn.Dropout(dropout)

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

        return self.dropout(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, embed_dim: int = 512, dropout: float = 0.2):
        """
        Args:
            embed_dim: The dimensionality of the input embeddings.
            dropout: The dropout rate.
        """
        super(ResidualConnection, self).__init__()
        
        self.norm = LayerNormalization(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Forward pass of the residual connection with layer normalization and dropout.

        Args:
            x: The input tensor.
            sublayer: A neural network layer (e.g., multi-head attention, feed-forward layer) 
                                  that processes the input tensor after normalization.

        Returns:
            The output tensor after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, n_heads: int = 8, dropout: float = 0.2):
        """
        Multi-Head Attention block as described in "Attention Is All You Need".

        Args:
            embed_dim: Total dimension of the embedding.
            n_heads: Number of parallel attention heads.
            dropout: Dropout rate applied to attention scores.

        The embed_dim should be divisible by n_heads.
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads."

        self.embed_dim_h = embed_dim // n_heads
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)    
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)    
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)    
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)    
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout):
        """
        Scaled dot-product attention mechanism.

        Args:
            query, key, value: Query, key, and value tensors.
            mask: Optional mask tensor.
            dropout: Dropout layer to apply on attention scores.

        Returns:
            Output tensor and attention scores.
        """
        embed_dim_h = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embed_dim_h)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = torch.matmul(attention_scores, value)
        return output, attention_scores
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttentionBlock.

        Args:
            query, key, value: Input tensors of shape [batch_size, seq_len, embed_dim].
            mask: Optional mask of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            Output of the multi-head attention mechanism.
        """
        batch_size, seq_len, _ = query.size()

        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # Reshape and transpose for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.embed_dim_h).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.embed_dim_h).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.embed_dim_h).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate and linear transformation
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.embed_dim_h)
        out = self.w_o(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, d_ff: int, dropout: float):
        """
        Args:
            embed_dim: The number of features in the input embeddings.
            n_heads: The number of heads in the multi-head attention mechanism.
            d_ff: The intermediate dimension of the feed-forward network.
            dropout: The dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.self_attention_block = MultiHeadAttentionBlock(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
        self.feed_forward_block = FeedForwardBlock(embed_dim=embed_dim, d_ff=d_ff, dropout=dropout)
        self.residual_connections = nn.ModuleList([
            ResidualConnection(embed_dim=embed_dim, dropout=dropout) for _ in range(2)
        ])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer encoder block.

        Args:
            x: The input tensor with shape [batch_size, seq_len, embed_dim].
            src_mask: The source mask tensor with shape [batch_size, 1, seq_len, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim].
        """
        # Apply self-attention
        self_attention = lambda x: self.self_attention_block(x, x, x, src_mask)
        x = self.residual_connections[0](x, self_attention)

        # Apply feed-forward network
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x