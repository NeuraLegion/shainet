import sys
import math
import warnings
try:
    import torch
except Exception as e:
    sys.stderr.write("PyTorch not installed: %s\n" % e)
    sys.exit(1)

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    sys.stderr.write("usage: build_transformer_model.py <output.pt>\n")
    sys.exit(1)

class SHAIMultiheadAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w_q = torch.nn.Linear(dim, dim, bias=False)
        self.w_k = torch.nn.Linear(dim, dim, bias=False)
        self.w_v = torch.nn.Linear(dim, dim, bias=False)
        self.w_o = torch.nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x.size(-1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.w_o(out)

class SHAIPositionWiseFF(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden)
        self.w2 = torch.nn.Linear(hidden, dim)
    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))

class SHAISimpleLayer(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.mha = SHAIMultiheadAttention(dim)
        self.ffn = SHAIPositionWiseFF(dim, hidden)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
    def forward(self, x):
        attn = self.mha(x)
        n1 = self.norm1(attn)
        ff = self.ffn(n1)
        return self.norm2(ff)

class TinyTransformer(torch.nn.Module):
    def __init__(self, vocab=4, dim=2, hidden=8, out_dim=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, dim)
        self.layer = SHAISimpleLayer(dim, hidden)
        self.out = torch.nn.Linear(dim, out_dim)
    def forward(self, x):
        emb = self.embedding(x)
        z = self.layer(emb)
        return self.out(z)

torch.manual_seed(0)
model = TinyTransformer()
example = torch.randint(0, 4, (1, 1))
traced = torch.jit.trace(model, example)
traced.save(sys.argv[1])
