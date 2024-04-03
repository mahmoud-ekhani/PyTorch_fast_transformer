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
        self.eps = eps
        
        # Define alpha and beta tensors as learnable parameters for each feature
        self.alpha = nn.Parameter(torch.ones(n_feats))
        self.beta = nn.Parameter(torch.zeros(n_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer normalization.

        Args:
            x: The input feature tensor of shape: [batch_size, seq_len, n_feat]

        Returs:
            Normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True) # Shape: [batch_size, seq_len, 1]
        std = x.std(dim=-1, keepdim=True) # Shape: [batch_size, seq_len, 1]

        normalized_x = (x - mean) / (std + self.eps)

        return self.alpha * normalized_x + self.beta
    

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, d_ff: int, dropout: float):
        """
        Args:
            embed_dim: The embedding dimension.
            d_ff: The intermediate dimension of the feed-forward network.
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
        The forward pass of the FeedForwardBlock, which consists of two linear
          transformations with a ReLU activation in between.

        Args:
            x: Input tensor of ahape: [batch_size, seq_len, embed_dim].

        Returns:
            Output tensor of shape: [batch_size, seq_len, embed_dim].
        """
        return self.ffb(x)


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: The vocabulary size.
            embed_dim: The embedding dimension.
        """
        super(InputEmbeddings, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the embedding layer.

        Args:
            token_ids: Tensor of token ids with shape: [batch_size, seq_length].
        
        Returns:
            out: Tensor of embeddings with shape [batch_size, seq_length, embed_dim].
        """
        return self.embed(token_ids) * math.sqrt(self.embed_dim)
    

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, dropout: float):
        """
        Args:
            seq_len: The maximum length of the input sequences.
            embed_dim: The embedding dimension.
            droptout: The dropout rate.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        # Create a position tensor
        position = torch.arange(seq_len).unsqueeze(1)
        # Compute the div term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * \
                             -(math.log(10000.0) / embed_dim))
        
        # Initialize a zeros tensor for positional encoding
        pe = torch.zeros(seq_len, embed_dim)

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
            x: Input tensor of input embeddings with shape [batch_size, seq_length, embed_dim]
        Returns:
            Tensor with positional embeddings added to the original embedding.
        """
        # Add positional encoding to the input tensor
        seq_len = x.size(1)
        x = x + (self.pe[:, :seq_len, :]).requires_grad_(False) # Shape: [batch_size, seq_len, embed_dim]

        return self.dropout(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        """
        Args:
            embed_dim: The embedding dimension.
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
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        """
        Multi-Head Attention block as described in "Attention Is All You Need".

        Args:
            embed_dim: The embeddings dimension.
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
        Scaled dot-product attention.

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
            attention_scores = attention_scores.masked_fill_(mask == 0, float('-inf'))
        
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
            mask: Optional mask.

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
    def __init__(self, 
                 embed_dim: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        """
        Args:
            embed_dim: The embedding dimension.
            self_attention_block: The multi-head self-attention block.
            feed_forward_block: The feed-forward network.
            dropout: The dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(embed_dim, dropout) for _ in range(2)
        ])
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        The encoder block consists of a self-attention layer and a feed-forward network.

        Args:
            x: The input to self-attention. Shape: [batch_size, seq_len, embed_dim]
            src_mask: Optional input sequence mask.
        
        Returns:
            The output tensor. Shape: [batch_size, seq_len, embed_dim]
        """
        # Apply self-attention
        self_attention = lambda x: self.self_attention_block(x, x, x, src_mask)
        x = self.residual_connections[0](x, self_attention)

        # Apply feed-forward network
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim: int, layers: nn.ModuleList):
        """
        Args:
            embed_dim: The number of features in the input embeddings.
            layers: A module list of EncoderBlock layers.
        """
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(embed_dim)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Encoder through its layers.

        Args:
            x: The input embedding tensor of shape [batch_size, seq_len, embed_dim].
            src_mask: The source mask.

        Returns:
            An encoder output tensor of shape [batch_size, seq_len, embed_dim].
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        """
        Args:
            embed_dim: The embeddings dimension.
            self_attention_block: The decoder self-attention block.
            cross_attention_block: The encoder-decoder cross-attention block.
            feed_forward_block: The feed-forward block of the decoder.
            dropout: The dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(embed_dim, dropout) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer decoder block composed of self-attention, 
            encoder-decoder cross-attention, and a feed-forward network.

        Args:
            x: The decoder input tensor of shape [batch_size, seq_len, embed_dim].
            enc_out: The encoder output tensor of shape [batch_size, seq_len, embed_dim].
            src_mask, tgt_mask: The mask tensors of the source and target sequences.

        Returns:
            The decoder block output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Apply self-attention
        self_attention = lambda x: self.self_attention_block(x, x, x, tgt_mask)
        x = self.residual_connections[0](x, self_attention)

        # Apply encoder-decoder cross-attention
        cross_attention = lambda x: self.cross_attention_block(x, enc_out, enc_out, src_mask)
        x = self.residual_connections[1](x, cross_attention)

        # Apply feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x
        
    
class Decoder(nn.Module):
    def __init__(self, embed_dim: int, layers: nn.ModuleList):
        """
        Args:
            embed_dim: The number of features in the input embeddings.
            layers: A module list of DecoderBlock layers.
        """
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(embed_dim)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the decoder through its layers.

        Args:
            x: The target input tensor of shape [batch_size, seq_len, embed_dim].
            enc_out: The encoder output tensor of shape [batch_size, seq_len, embed_dim].
            src_mask, tgt_mask: The source and target masks, respectively.

        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim].
        """
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        """
        Args:
            embed_dim: The number of features in token embeddings.
            vocab_size: The target vocabulary size.
        """
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the projection layer.
        
        Args:
            x: The decoder output of shape [batch_size, seq_len, embed_dim].

        Returns:
            The decoder output projected to the shape [batch_size, seq_len, vocab_size].
        """
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        """
        Args:
            encoder: The Transformer's encoder.
            decoder: The Transformer's decoder.
            src_embed: Embedding layer for source vocabulary.
            tgt_embed: Embedding layer for target vocabulary.
            src_pos: Positional encoding layer for source sequence.
            tgt_pos: Positional encoding layer for target sequence.
            projection_layer: Projection layer to output vocabulary size.
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
            src: The source sequence tensor.
            src_mask: The source mask tensor.

        Returns:
            The encoded output.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the target sequence.

        Args:
            encoder_output: The output from the encoder.
            src_mask: The source mask tensor.
            tgt: The target sequence tensor.
            tgt_mask: The target mask tensor.

        Returns:
            The decoded output.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the decoded sequence to the vocabulary space.

        Args:
            x: The decoded sequence tensor.

        Returns:
            The output tensor in the vocabulary space.
        """
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      embed_dim: int = 512, 
                      n_layers: int = 6,
                      n_heads: int = 8, 
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    """
    Builds a Transformer model with specified parameters.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        src_seq_len: Maximum length of source sequences.
        tgt_seq_len: Maximum length of target sequences.
        embed_dim: Dimensionality of the model's embeddings.
        n_layers: Number of layers in both the encoder and decoder.
        n_heads: Number of attention heads.
        dropout (float): Dropout rate.
        d_ff (int): Dimensionality of the feed-forward network's intermediate layer.

    Returns:
        An instance of the Transformer model.
    """
    # Create embedding and positional encoding layers
    src_embed = InputEmbeddings(embed_dim, src_vocab_size)
    tgt_embed = InputEmbeddings(embed_dim, tgt_vocab_size)
    src_pos = PositionalEncoding(embed_dim, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(embed_dim, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)
        encoder_block = EncoderBlock(embed_dim, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(n_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, n_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(embed_dim, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)
        decoder_block = DecoderBlock(embed_dim, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(embed_dim, nn.ModuleList(encoder_blocks))
    decoder = Decoder(embed_dim, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(embed_dim, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

     # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
