import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    norm_x = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, d_model, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) # scale 
        self.beta = nn.Parameter(torch.zeros(d_model)) # bias
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class PositionalEncoding(nn.Module):
    """
    x = x + PE
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class FeedForwardBlock(nn.Module):
    """
    FFN(x) = ReLU(xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class InputEmbeddings(nn.Module):
    """
    x = Embedding(x) * sqrt(d_model)
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class ResidualConnection(nn.Module):
    """
    x = x + Dropout(sublayer(LayerNorm(x)))

    sublayer: nn.Module, e.g. MultiHeadAttention, FeedForwardBlock

    Example:
    sublayer = FeedForwardBlock(d_model, d_ff, dropout)
    residual = ResidualConnection(d_model, dropout)
    --> LayerNorm(x) = x' = gamma * (x - mean) / sqrt(var + eps) + beta
    --> sublayer(LayerNorm(x)) = FFN(LayerNorm(x)) = FFN(x') = ReLU(x'W1 + b1)W2 + b2
    --> x = x + ReLU(x'W1 + b1)W2 + b2
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
        pass

    @staticmethod
    def attention(Q, K, V, mask, dropout: nn.Dropout):
        """
        Q: (batch x seq_len x d_model) or (batch x n_heads x seq_len x d_k)
        K: (batch x seq_len x d_model) or (batch x n_heads x seq_len x d_k)
        V: (batch x seq_len x d_model) or (batch x n_heads x seq_len x d_k)

        However, in current implementation, we will use the second form
        """
        d_k = Q.shape[-1]

        # (batch x n_heads x seq_len x d_k) --> (batch x n_heads x seq_len x seq_len)
        attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        # (batch x n_heads x seq_len x seq_len)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ V, attention_scores)

    
