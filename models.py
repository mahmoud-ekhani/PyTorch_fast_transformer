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
            key, query, value: Tensors of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask for the decoder. Shape: [batch_size, 1, seq_len_query, seq_len]

        Returns:
            Output of the multi-head attention. Shape: [batch_size, seq_len_query, embed_dim]
        """
        batch_size = key.size(0)
        seq_len = key.size(1)
        seq_len_query = query.size(1)

        # Reshape and linearly transform keys, queries, and values
        key = self.key_matrix(key.view(batch_size, seq_len, self.n_heads, self.single_head_dim))
        query = self.query_matrix(query.view(batch_size, seq_len_query, self.n_heads, self.single_head_dim))
        value = self.value_matrix(value.view(batch_size, seq_len, self.n_heads, self.single_head_dim))

        # Transpose to get dimensions [batch_size, n_heads, seq_len, single_head_dim]
        key, query, value = [x.transpose(1, 2) for x in (key, query, value)]

        # Scaled Dot-Product Attention
        k_adjusted = key.transpose(-1, -2) # Shape: [batch_size, n_heads, single_head_dim, seq_len]
        product = torch.matmul(query, k_adjusted) / math.sqrt(self.single_head_dim) # Shape: [batch_size, n_heads, seq_len_query, seq_len]

        # Masking for decoder
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        scores = F.softmax(product, dim=-1)
        attention = torch.matmul(scores, value) # Shape: [batch_size, n_heads, seq_len_query, signle_head_dim]

        # Concatenate the heads and pass through the final lineaer layer
        concat = attention.transpose(1, 2).contiguous().view(batch_size, seq_len_query, -1) # Shape: [batch_size, seq_len_query, n_heads * single_head_dim]
        output = self.out(concat) # Shape: [batch_size, seq_len_query, embed_dim]
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 8):
        """
        Args:
            embed_dim: Dimension of the embedding vectors.
            expansion_factor: Factor determining the intermediate dimension in the feed-forward network.
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
            key: Key tensor. Shape: [batch_size, seq_len, embed_dim]
            query: Query tensor. Shape: [batch_size, seq_len_query, embed_dim]
            value: Value tensor. Shape: [batch_size, seq_len, embed_dim]
            mask: Optional mask. Shape: [batch_size, 1, seq_len_query, seq_len]

        Returns:
            Output after processing through the Transformer block. Shape: [batch_size, seq_len_query, embed_dim]
        """
        attention_out = self.attention(key, query, value, mask) # Shape: [batch_size, seq_len_query, embed_dim]
        attention_residual_out = attention_out + query
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out


class TransformerEncoder(nn.Module):
    def __init__(self, src_seq_len: int, src_vocab_size: int, embed_dim: int, num_layers: int = 6, expansion_factor: int = 4, n_heads: int = 8):
        """
        Args:
            src_seq_len: Length of the source input sequence.
            src_vocab_size: Size of the source vocabulary.
            embed_dim: Dimension of the embedding.
            num_layers: Number of Transformer encoder layers.
            expansion_factor: Factor determining the intermediate size of the feed-forward layer in the Transformer block.
            n_heads: Number of attention heads in multi-head attention.
        """
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = TokenEmbedding(src_vocab_size, embed_dim)
        self.positional_encoder = PositionalEncoding(src_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, expansion_factor, n_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Encoder consisting multiple Transformer blocks.

        Args:
            x: Input sequence. Shape: [batch_size, src_seq_len]

        Returns:
            Encoded output. Shape: [batch_size, src_seq_len, embed_dim]
        """
        embed_out = self.embedding_layer(x) # Shape: [batch_size, src_seq_len, embed_dim]
        out = self.positional_encoder(embed_out) # Shape: [batch_size, src_seq_len, embed_dim]

        for layer in self.layers:
            out = layer(out, out, out) # Shape: [batch_size, src_seq_len, embed_dim]

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 8):
        """
        Args:
            embed_dim: Dimension of the embedding.
            expansion_factor: Determines the intermediate dimension in the feed-forwad network.
            n_heads: Number of attention heads in the multi-head self-attention.
        """
        super(DecoderBlock, self).__init__()

        # Masked multi-head self-attention mechanism
        self.self_attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.2)

        # Encoder-decoder attention + feed-forward network
        self.enc_dec_attention = TransformerBlock(embed_dim, expansion_factor, n_heads)
       

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder Block.

        Args:
            x: Input tensor from previous decoder layer. Shape: [batch_size, tgt_seq_len, embed_dim]
            enc_out: Output tensor from the encoder. Shape: [batch_size, src_seq_len, embed_dim]
            src_mask: Source mask tensor. Shape: [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: Target mask tensor. Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]

        Returns:
            Output tensor. Shape: [batch_size, tgt_seq_len, embed_dim]
        """

        # Masked self-attention
        self_attn_out = self.self_attention(x, x, x, mask=tgt_mask)
        self_attn_out = self.norm1(self.dropout1(self_attn_out + x)) # Shape: [batch_size, tgt_seq_length, embed_dim]
    
        # Encoder-decoder attention + feed-forward network
        out = self.enc_dec_attention(enc_out, self_attn_out, enc_out, src_mask)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, tgt_seq_len: int, tgt_vocab_size: int, embed_dim: int,  num_layers: int = 6, 
                 expansion_factor: int = 4, n_heads: int = 8):
        """
        Args:
            tgt_seq_len: Length of the target sequence.
            tgt_vocab_size: Target vocabulary size.
            embed_dim: Dimension of embedding vector.
            num_layers: Number of decoder layers.
            expansion_factor: Determines the intermediate dimension of the feed-forward network.
            n_heads: Number of heads in the multi-head self-attention mechanism.
        """
        super(TransformerDecoder, self).__init__()
        self.embedding_layer = TokenEmbedding(tgt_vocab_size, embed_dim)
        self.positional_encoder = PositionalEncoding(tgt_seq_len, embed_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, expansion_factor, n_heads)
            for _ in range(num_layers)
        ])

        # Decoder output linear layer
        self.out_lin = nn.Linear(embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer Decoder consisting of multiple Decoder blocks.

        Args:
            x: Target sequence. Shape: [batch_size, tgt_seq_len]
            enc_out: Encoder output. Shape: [batch_size, src_seq_len, embed_dim]
            src_mask: Encoder mask. Shape: [batch_size, 1, 1, src_seq_len]
            tgt_mask: Decoder mask. Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]

        Returns:
            Decoder output. Shape: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        embed_out = self.embedding_layer(x) # Shape: [batch_size, tgt_seq_len, embed_dim]
        out = self.positional_encoder(embed_out) # Shape: [batch_size, tgt_seq_len, embed_dim]
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, enc_out, src_mask, tgt_mask) # Shape: [batch_size, tgt_seq_len, embed_dim]

        # Fully-connected layer + Softmax
        out = F.softmax(self.out_lin(out), dim=-1) # Shape: [batch_size, tgt_seq_len, tgt_vocab_size]

        return out


class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int,
                 src_seq_len: int,
                 tgt_seq_len: int,
                 embed_dim: int,
                 num_layers: int = 6,
                 expansion_factor: int = 4,
                 n_heads: int = 8):
        """
        Args:
            src_vocab_size: Size of the source vocabulary.
            tgt_vocab_size: Size of the target vocabulary.
            src_seq_len: Length of the input source sequences.
            tgt_seq_len: Length of the input target sequences.
            embed_dim: Size of the embedding vectors.
            num_layers: Number of the encoder and decoder layers.
            expansion_factor: Determines the intermediate dimension in the feed-forward networks.
            n_heads: Number of the heads in the multi-head attention (both self-attention and encoder-decoder attention) mechnism.
        """
        super(Transformer, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size

        # Encoder
        self.encoder = TransformerEncoder(src_seq_len, 
                                          src_vocab_size, 
                                          embed_dim, 
                                          num_layers, 
                                          expansion_factor, 
                                          n_heads)
        
        # Decoder
        self.decoder = TransformerDecoder(tgt_seq_len,
                                          tgt_vocab_size,
                                          embed_dim,
                                          num_layers,
                                          expansion_factor,
                                          n_heads)
        
    def make_mask(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Creates masks for the source and target sequences.

        Args:
            src: Source sequence. Shape: [batch_size, src_seq_len]
            tgt: Target sequence. Shape: [batch_size, tgt_seq_len]

        Returns:
            src_mask: Mask for source sequence. Shape: [batch_size, 1, 1, src_seq_len]
            tgt_mask: Mask for target sequence. Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        """
        batch_size, tgt_seq_len = tgt.size()

        # Mask for source sequence
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # Shape: [batch_size, 1, 1, src_seq_len]

        # Mask for target sequence (future information prevention)
        nopeak_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool() # Shape: [tgt_seq_len, tgt_seq_len]
        nopeak_mask = nopeak_mask.unsqueeze(0).expand(batch_size, 1, tgt_seq_len, tgt_seq_len)

        # Mask for target sequence (padding)
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # Shape: [batch_size, 1, 1, tgt_seq_len]
        tgt_mask = pad_mask & nopeak_mask  # Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]

        return src_mask, tgt_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args: 
            src: Input tensor of the encoder. Shape: [batch_size, src_seq_len]
            tgt: Input tensor of the decoder. Shape: [batch_size, tgt_seq_len]

        Returns:
            Output probabilities for each target word. Shape: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        src_mask, tgt_mask = self.make_mask(src, tgt)
        enc_out = self.encoder(src)
        outputs = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        return outputs