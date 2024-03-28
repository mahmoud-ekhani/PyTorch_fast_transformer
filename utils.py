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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: Dimension of embedding vector output.
            n_heads: Number of self-attention heads.

        The embedding dimension should be divisible by the number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = embed_dim // n_heads # Dimension per head (e.g., 512 / 8 = 64)

        # Linear transformation for queries, keys, and values
        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)

        # Output linear layer
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttention layer.
        
        Args:
            key, query, value: Tensors of shape [batch_size, seq_length, embed_dim]
            mask: Optional mask for the decoder. Shape: [batch_size, 1, seq_length_query, seq_length]

        Returns:
            Output of the multi-head attention. Shape: [batch_size, seq_length_query, embed_dim]
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)

        # Reshape and linearly transform queries, keys, and values
        key = self.key_matrix(key.view(batch_size, seq_length, self.n_heads, self.single_head_dim))
        query = self.query_matrix(query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim))
        value = self.value_matrix(value.view(batch_size, seq_length, self.n_heads, self.single_head_dim))

        # Transpose to get dimensions [batch_size, n_heads, seq_length, single_head_dim]
        key, query, value = [x.transpose(1, 2) for x in (key, query, value)]

        # Scaled Dot-Product Attention
        k_adjusted = key.transpose(-1, -2) # Shape: [batch_size, n_heads, single_head_dim, seq_length]
        product = torch.matmul(query, k_adjusted) / math.sqrt(self.single_head_dim) # Shape: [batch_size, n_heads, seq_length_query, seq_length]

        # Masking for decoder
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        scores = F.softmax(product, dim=-1)
        attention = torch.matmul(scores, value) # Shape: [batch_size, n_heads, seq_length_query, signle_head_dim]

        # Concatenate the heads and pass through the final lineaer layer
        concat = attention.transpose(1, 2).contiguous().view(batch_size, seq_length_query, -1)
        output = self.out(concat) # Shape: [batch_size, seq_length_query, embed_dim]
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 8):
        """
        Args:
            embed_dim: Dimension of the embedding vectors.
            expansion_factor: Factor determining the output dimension of the first linear layer in the feed-forward network.
            n_heads: Number of attention heads in the multi-head self-attention module.
        """
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Forward pass of the Transformer Block with multi-head self-attention and a feed-forwad network.

        Args:
            key: Key tensor. Shape: [batch_size, seq_length, embed_dim]
            query: Query tensor. Shape: [batch_size, seq_length_query, embed_dim]
            value: Value tensor. Shape: [batch_size, seq_length, embed_dim]
            mask: Optional mask for the decoder. Shape: [batch_size, 1, seq_length_query, seq_length]

        Returns:
            Output after processing through the Transformer block. Shape: [batch_size, seq_length_query, embed_dim]
        """
        attention_out = self.attention(key, query, value, mask) # Shape: [batch_size, seq_length_query, embed_dim]
        attention_residual_out = attention_out + query
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out


    



