import pdb

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat


from model.BrownianBridge.base.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
    


from model.BrownianBridge.transformer.transformer import Transformer, MutliHeadCrossAttention, FeedForward

# class CT_transformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ct_down = nn.Conv3d(
#                             in_channels=1,
#                             out_channels=128,
#                             kernel_size=8,
#                             stride=8,
#                             padding=0  # bạn có thể điều chỉnh nếu muốn giữ biên tốt hơn
#                         )
#         self.nc_down = nn.Conv2d(
#                             in_channels=3,
#                             out_channels=128,
#                             kernel_size=8,
#                             stride=8,
#                             padding=0
#                         )
#         self.deconv = nn.ConvTranspose2d(
#             in_channels=128,
#             out_channels=3,
#             kernel_size=8,
#             stride=8,
#             padding=0
#         )
#         self.cross_attn = MutliHeadCrossAttention(dim=128, heads=8, dim_head=128, dropout=0.1)
#     def forward(self, x_cond_latent, attmap):
#         y = self.ct_down(attmap)  # Thêm chiều kênh
#         x = self.nc_down(x_cond_latent)



#         y = y.view(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
#         x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

#         x, attn_weights = self.cross_attn(x, y)
#         x = x.permute(0, 2, 1)
#         x = x.view(x.shape[0], x.shape[1], 8, 8)
#         x = self.deconv(x)
#         # attn_weights = attn_weights.view(attn_weights.shape[0], attn_weights.shape[2], attn_weights.shape[3]).permute(0, 2, 1)

#         return x, attn_weights



# class CT_transformer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.ct_down = nn.Conv3d(
#         #                     in_channels=1,
#         #                     out_channels=128,
#         #                     kernel_size=8,
#         #                     stride=8,
#         #                     padding=0  # bạn có thể điều chỉnh nếu muốn giữ biên tốt hơn
#         #                 )
        
#         self.ct_down = nn.Sequential(
#                             nn.Conv3d(1, 128, kernel_size=3, stride=1, padding=1), # giữ nguyên kích thước
#                             nn.ReLU(inplace=True),
#                             nn.AdaptiveAvgPool3d((8, 8, 8))                       # ép về 8x8x8
#                         )
#         # self.nc_down = nn.Conv2d(
#         #                     in_channels=1,
#         #                     out_channels=128,
#         #                     kernel_size=8,
#         #                     stride=8,
#         #                     padding=0
#         #                 )
#         self.nc_down = nn.Sequential(
#                             nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1), # giữ nguyên 64x64
#                             nn.ReLU(inplace=True),
#                             nn.AdaptiveAvgPool2d((4, 4))                           # ép về 4x4
#                         )
#         self.deconv = nn.Sequential(
#                         nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 4 -> 8
#                         nn.BatchNorm2d(128),
#                         nn.ReLU(),
#                         nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
#                         nn.BatchNorm2d(64),
#                         nn.ReLU(),
#                         nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 -> 32
#                         nn.BatchNorm2d(32),
#                         nn.ReLU(),
#                         nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 32 -> 64
#                     )
#         self.cross_attn = MutliHeadCrossAttention(dim=128, heads=8, dim_head=128, dropout=0.1)
#     def forward(self, x_cond_latent, attmap):
#         y = self.ct_down(attmap)  # Thêm chiều kênh
#         x = self.nc_down(x_cond_latent)



#         y = y.view(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
#         x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

#         x, attn_weights = self.cross_attn(x, y)
#         x = x.permute(0, 2, 1)
#         x = x.view(x.shape[0], x.shape[1], 4, 4)
#         x = self.deconv(x)
#         # attn_weights = attn_weights.view(attn_weights.shape[0], attn_weights.shape[2], attn_weights.shape[3]).permute(0, 2, 1)
#         return x, attn_weights




class CT_transformer(nn.Module):
    def __init__(self, neighbor_slices, depth = 1, dim=512, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.nc_down = nn.Sequential(
            nn.Conv2d(2*neighbor_slices+1, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.nc_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(64, 2*neighbor_slices+1, kernel_size=3, padding=1),
        )
    #    self.conv = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.cross_attn = Transformer(dim, depth, heads, dim_head, dropout)


        self.down1 = nn.Sequential(
    nn.Conv2d(32, 32, 3, 2, 1),
    nn.LayerNorm([32, 4, 4]),
    nn.GELU()
        )

        self.down2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),

        )
    def forward(self, x_cond_latent, attmap):
        # x_cond_latent: b, c, h, w
        y = self.down2(attmap)
        y = rearrange(y, 'b t h w d -> (b t) d h w')
     #   y = self.conv(y)
        y = self.down1(y)
        y = rearrange(y, '(b t) d h w -> b t (d h w)', t = 101)



        x = self.nc_down(x_cond_latent)
        x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
        y = y.view(y.shape[0], -1, y.shape[-1])
        x_attn, attn_weights = self.cross_attn(x, y)

        x_attn = x_attn.permute(0, 2, 1)
        x_attn = x_attn.view(x_attn.shape[0], x_attn.shape[1], 16, 16)
        x_attn = self.nc_up(x_attn) + x_cond_latent
        # attn_weights = attn_weights.view(attn_weights.shape[0], attn_weights.shape[2], attn_weights.shape[3]).permute(0, 2, 1)
        return x_attn, attn_weights



class CT_transformer_3D(nn.Module):
    def __init__(self, neighbor_slices, depth=1, dim=512, heads=8, dim_head=64, dropout=0.1):
        super().__init__()

        self.neighbor_slices = neighbor_slices
        self.D = 2 * neighbor_slices + 1

        self.nc_down = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),

            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GELU(),

            nn.Conv3d(256, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
        )

        self.nc_up = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv3d(64, 1, kernel_size=3, padding=1),
        )

        self.cross_attn = Transformer(dim, depth, heads, dim_head, dropout)

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LayerNorm([32, 4, 4]),
            nn.GELU()
        )

        self.down2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

    def forward(self, x_cond_latent, attmap):
        """
        x_cond_latent: [B, D, H, W]
        attmap:        [B, T, H_att, W_att, 512]
        """

        B, D, H, W = x_cond_latent.shape
        T = attmap.shape[1]

        # ===== attmap branch giữ như cũ =====
        y = self.down2(attmap)                       # [B, T, h, w, 32]
        y = rearrange(y, 'b t h w d -> (b t) d h w')
        y = self.down1(y)                            # [(B*T), 32, 4, 4]
        y = rearrange(y, '(b t) d h w -> b t (d h w)', t=T)
        y = y.view(B, -1, y.shape[-1])               # [B, T, 512]

        # ===== NAC branch 3D =====
        x_in = x_cond_latent.unsqueeze(1)            # [B, 1, D, H, W]

        x = self.nc_down(x_in)                       # [B, 512, D, H/4, W/4]

        _, C, D2, H2, W2 = x.shape

        x = x.view(B, C, -1).transpose(1, 2)         # [B, D*H2*W2, 512]

        # ===== cross attention =====
        x_attn, attn_weights = self.cross_attn(x, y)

        # ===== reshape back to 3D =====
        x_attn = x_attn.transpose(1, 2)              # [B, 512, D*H2*W2]
        x_attn = x_attn.view(B, C, D2, H2, W2)       # [B, 512, D, H/4, W/4]

        x_attn = self.nc_up(x_attn)                  # [B, 1, D, H, W]
        x_attn = x_attn.squeeze(1)                   # [B, D, H, W]

        # residual
        x_attn = x_attn + x_cond_latent              # [B, D, H, W]

        return x_attn, attn_weights