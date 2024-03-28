import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            max_seq_len: maximum lenght of the input sequence
            embed_model_dim: dimension of embedding
        """
        super().__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos/ 10000**((2*i)/self.embed_dim))
                pe[pos, i+1] = math.cos(pos/ 10000**((2*(i+1))/self.embed_dim))