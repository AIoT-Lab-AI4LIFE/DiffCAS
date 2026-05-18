import torch
from torch import nn
import einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


### Không trả về attention weight
# class MutliHeadCrossAttention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.normx = nn.LayerNorm(dim)
#         self.normy = nn.LayerNorm(dim)
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
#         self.to_q = nn.Linear(dim, inner_dim, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x, y):
#         x = self.normx(x)
#         y = self.normy(y)
#         q = self.to_q(x)
#         kv = self.to_kv(y).chunk(2, dim = -1)
#         q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
#         k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

#         # print("q shape:", q.shape)
#         # print("k shape:", k.shape)
#         # print("v shape:", v.shape)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# #NC qua vq gan cross CT
# class Transformer(nn.Module):
#     def __init__(self, dim, depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout = 0.1, h_patch = 16, w_patch = 16, d_patch = 16, embed_size=512):
#         super().__init__()
#         self.h_patch = h_patch
#         self.w_patch = w_patch
#         self.d_patch = d_patch  
#         self.embed_size = embed_size
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b (d_patch p1) (h_patch p2) (w_patch p3) -> b (d_patch h_patch w_patch) (p1 p2 p3)', p1 = h_patch, p3 = w_patch, p2 = h_patch),
#             nn.Linear(h_patch*d_patch*w_patch, embed_size)

#         )
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 MutliHeadCrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
#                 FeedForward(dim, mlp_dim, dropout = dropout)
#             ]))
#     def forward(self, x, y):
#         y = self.to_patch_embedding(y)
#         for attn, ff in self.layers:
#             x = attn(x, y) + x
#             x = ff(x) + x
#         return x
    



###Trả về attention weight
class MutliHeadCrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.normx = nn.LayerNorm(dim)
        self.normy = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        x = self.normx(x)
        y = self.normy(y)
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        # print("q shape:", q.shape)
        # print("k shape:", k.shape)
        # print("v shape:", v.shape)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

#NC qua vq gan cross CT
class Transformer(nn.Module):
    def __init__(self, dim, depth=1, heads=8, dim_head=64, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MutliHeadCrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, dim*4, dropout = dropout)
            ]))
    def forward(self, x, y):
        loss_to_learn_from_attmap = 0.0
        attentions = []

        for attn, ff in self.layers:
            x_out, attn_weights = attn(x, y)


            attentions.append(attn_weights)

            # Residual + FFN
            x = x + x_out
            x = x + ff(x)

        return x, attentions, loss_to_learn_from_attmap

    

