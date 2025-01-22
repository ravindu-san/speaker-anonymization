from einops import rearrange
from einops.layers.torch import Rearrange
from einops import reduce
import torch
from torch import nn, einsum
from functools import partial
import torch.nn.functional as F

from position_embeddings import SinusoidalPositionEmbeddings

#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x
#


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# class Block(nn.Module):
#     def __init__(self, dim_in, dim_out, groups=8):
#         super().__init__()
#         self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)  # TODO weight standardized conv
#         self.norm = nn.GroupNorm(groups, dim_out)
#         self.activation = nn.SiLU()
#
#     def forward(self, x, scale_shift=None):
#         x = self.proj(x)
#         x = self.norm(x)
#
#         # time embedding
#         if scale_shift is not None:
#             scale, shift = scale_shift
#             x = x * (scale + 1) + shift
#
#         x = self.activation(x)
#         return x


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, cond_emb_dim, *, groups=8):
        super().__init__()
        full_emb_dim = time_emb_dim + cond_emb_dim
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(full_emb_dim, dim_out * 2))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, cond_emb):
        # scale_shift = None
        full_emb_tupple = (time_emb, cond_emb)
        full_emb = torch.cat(full_emb_tupple, dim=-1)
        full_emb = self.mlp(full_emb)
        full_emb = rearrange(full_emb, "b c -> b c 1 1")
        scale_shift = full_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


# class ResnetBlock(nn.Module):
#     def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
#         super().__init__()
#         self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if (time_emb_dim is not None) else None)
#         self.block1 = Block(dim_in, dim_out, groups=groups)
#         self.block2 = Block(dim_out, dim_out, groups=groups)
#         self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
#
#     def forward(self, x, time_emb=None):
#         scale_shift = None
#         if (self.mlp is not None) and (time_emb is not None):
#             time_emb = self.mlp(time_emb)
#             time_emb = rearrange(time_emb, "b c -> b c 1 1")
#             scale_shift = time_emb.chunk(2, dim=1)
#
#         h = self.block1(x, scale_shift=scale_shift)
#         h = self.block2(h)
#         return h + self.res_conv(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         # self.dim = dim
#         self.heads = heads
#         # self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
#         self.to_q = nn.Conv2d(dim, dim_head * heads, 1, bias=False)
#         self.to_k = nn.Conv2d(dim, dim_head * heads, 1, bias=False)
#         self.to_v = nn.Conv2d(dim, dim_head * heads, 1, bias=False)
#         self.to_out = nn.Conv2d(dim_head * heads, dim, 1)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         q = self.to_q(x)
#         k = self.to_k(x)
#         v = self.to_v(x)
#
#         q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads) * self.scale
#         k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
#         v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)
#
#         qk = einsum('b h d i, b h d j -> b h i j', q, k)
#         qk = qk - qk.amax(dim=-1, keepdim=True).detach()
#         attention = qk.softmax(dim=-1)
#
#         att_val = einsum('b h i j, b h d j -> b h i d', attention, v)
#         out = rearrange(att_val, 'b h (x y) d -> b (h d) x y', x=h, y=w)
#         return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


def Upsample(dim, dim_out=None):
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.normalize = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.normalize(x)
        return self.fn(x)


# class WaveEncoder(nn.Module):
#     def __init__(self, channels=None, groups=8):
#         super().__init__()
#
#         if channels is None:
#             channels = [32, 64, 128, 256]
#         layers = []
#         in_channels = 1  # Raw audio is single channel
#
#         for out_channels in channels:
#             layers.extend([
#                 nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#                 nn.GroupNorm(groups, out_channels),
#                 nn.ReLU(),
#             ])
#             in_channels = out_channels
#
#         self.encoder = nn.Sequential(*layers)
#
#         #
#         # self.pre_net = nn.Sequential(
#         #     nn.Conv1d(1, channels, 7, padding=3),
#         #     nn.GroupNorm(8, channels),
#         #     nn.GELU(),
#         #     nn.Dropout(0.1)
#         # )
#         #
#         # self.dilated_convs = nn.ModuleList([
#         #     nn.Sequential(
#         #         nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
#         #         nn.GroupNorm(8, channels),
#         #         nn.GELU(),
#         #         nn.Dropout(0.1)
#         #     ) for dilation in [1, 2, 4, 8, 16, 32]
#         # ])
#
#
#
#     def forward(self, x):
#         # wave_encoding = self.encoder(x)
#
#         wave_encoding = self.pre_net(x)
#         wave_encoding = torch.mean(wave_encoding, dim=2)
#         return wave_encoding


# class DilatedLinguisticEncoder(nn.Module):
#     def __init__(self, input_dim=1, channels=256, embedding_size=128):
#         super().__init__()
#
#         self.embedding_size = embedding_size
#         self.channels = channels
#
#         self.pre_net = nn.Sequential(
#             nn.Conv1d(input_dim, channels, 7, padding=3),
#             nn.GroupNorm(8, channels),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )
#
#         self.dilated_convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
#                 nn.GroupNorm(8, channels),
#                 nn.GELU(),
#                 nn.Dropout(0.1)
#             ) for dilation in [1, 2, 8, 16]
#         ])
#
#         self.residual_convs = nn.ModuleList([
#             nn.Conv1d(channels, channels, 1)
#             for _ in range(4)  # match number of dilated convs
#         ])
#
#         self.post_net = nn.Sequential(
#             nn.Conv1d(channels, channels, 1),
#             nn.GroupNorm(8, channels),
#             nn.GELU()
#         )
#
#         # self.pool_and_embed = nn.Sequential(
#         #     # Global attention pooling
#         #     nn.Conv1d(channels, 1, 1),  # Attention weights
#         #     nn.Softmax(dim=2)  # Softmax over sequence dimension
#         # )
#
#         self.final_projection = nn.Sequential(
#             nn.Linear(channels, embedding_size),
#             nn.LayerNorm(embedding_size)  # Normalize final embeddings
#         )
#
#     def forward(self, x):
#         x = self.pre_net(x)
#
#         for dilated_conv, res_conv in zip(self.dilated_convs, self.residual_convs):
#             residual = res_conv(x)
#             x = dilated_conv(x) + residual
#
#         x = self.post_net(x)
#
#         # attention_weights = self.pool_and_embed(x)
#         # pooled = torch.sum(x * attention_weights, dim=2)
#
#         pooled = F.adaptive_avg_pool2d(x, (self.channels, 1))
#
#         embedding = self.final_projection(pooled.squeeze(-1))
#
#         return embedding


class DilatedLinguisticEncoder(nn.Module):
    def __init__(self, input_dim=1, channels=128, embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size
        self.channels = channels

        # Define channel progression ensuring divisibility by 8 for GroupNorm
        self.channel_sizes = [128, 64, 32, 16, 8]  # All divisible by 8

        self.pre_net = nn.Sequential(
            nn.Conv1d(input_dim, self.channel_sizes[0], 5, padding=2),
            nn.GroupNorm(8, self.channel_sizes[0]),  # 128/8 = 16 channels per group
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.dilated_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.channel_sizes[i],
                        self.channel_sizes[i + 1],
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(
                        8 if self.channel_sizes[i + 1] >= 8 else 1,
                        self.channel_sizes[i + 1],
                    ),  # Adjust groups
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for i, dilation in enumerate([1, 2, 4, 8])
            ]
        )

        self.residual_convs = nn.ModuleList(
            [
                nn.Conv1d(self.channel_sizes[i], self.channel_sizes[i + 1], 1)
                for i in range(4)
            ]
        )

        self.post_net = nn.Sequential(
            nn.Conv1d(self.channel_sizes[-1], self.channel_sizes[-1], 1),
            nn.GroupNorm(
                1, self.channel_sizes[-1]
            ),  # Use 1 group for small channel counts
            nn.GELU(),
        )

        self.final_projection = nn.Sequential(
            nn.Linear(self.channel_sizes[-1], embedding_size),
            nn.LayerNorm(embedding_size),
        )

    def forward(self, x):
        x = self.pre_net(x)

        # Optional: Gradient checkpointing
        for dilated_conv, res_conv in zip(self.dilated_convs, self.residual_convs):
            residual = res_conv(x)
            x = dilated_conv(x) + residual

        x = self.post_net(x)
        pooled = F.adaptive_avg_pool2d(x, (x.size(1), 1))
        embedding = self.final_projection(pooled.squeeze(-1))

        return embedding


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 3, 4),
        channels=1,
        resnet_block_groups=4,
        *args,
        **kwargs
    ):
        super().__init__()
        if init_dim is None:
            init_dim = dim
        self.init_conv = nn.Conv2d(channels, init_dim, 1)

        time_dim = 4 * dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        cond_dim = 4 * dim
        # self.cond_mlp = nn.Sequential(WaveEncoder(channels=[32,64,128,256]), nn.Linear(256, cond_dim), nn.GELU(),
        #                               nn.Linear(cond_dim, cond_dim))
        # self.cond_mlp = WaveEncoder(channels=cond_dim, groups=resnet_block_groups)
        self.cond_mlp = DilatedLinguisticEncoder(
            input_dim=1, channels=128, embedding_size=cond_dim
        )

        dims = [init_dim, *[i * dim for i in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.downs = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            cond_emb_dim=cond_dim,
                            groups=resnet_block_groups,
                        ),
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            cond_emb_dim=cond_dim,
                            groups=resnet_block_groups,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_resnet_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim,
            cond_emb_dim=cond_dim,
            groups=resnet_block_groups,
        )
        self.mid_attention = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_resnet_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            time_emb_dim=time_dim,
            cond_emb_dim=cond_dim,
            groups=resnet_block_groups,
        )

        self.ups = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in + dim_out,
                            dim_out,
                            time_emb_dim=time_dim,
                            cond_emb_dim=cond_dim,
                            groups=resnet_block_groups,
                        ),
                        ResnetBlock(
                            dim_in + dim_out,
                            dim_out,
                            time_emb_dim=time_dim,
                            cond_emb_dim=cond_dim,
                            groups=resnet_block_groups,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_resnet_block = ResnetBlock(
            2 * dim,
            dim,
            time_emb_dim=time_dim,
            cond_emb_dim=cond_dim,
            groups=resnet_block_groups,
        )
        if out_dim is None:
            out_dim = channels
        self.final_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time, conditioner):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        cond = self.cond_mlp(conditioner)

        h = []

        for resnet1, resnet2, attention, downsample in self.downs:
            x = resnet1(x, t, cond)
            h.append(x)

            x = resnet2(x, t, cond)
            x = attention(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_resnet_block1(x, t, cond)
        x = self.mid_attention(x)
        x = self.mid_resnet_block2(x, t, cond)

        for resnet1, resnet2, attention, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t, cond)

            x = torch.cat((x, h.pop()), dim=1)
            x = resnet2(x, t, cond)

            x = attention(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_resnet_block(x, t, cond)
        return self.final_conv(x)


if __name__ == "__main__":
    batch_size = 8
    channels = 1
    time_steps = 100

    x = torch.rand((batch_size, channels, 32, 32))
    cond = torch.rand((batch_size, channels, 32 * 256))
    t = torch.randint(0, time_steps, (batch_size,)).long()
    model = Unet(dim=32, dim_mults=(1, 2, 4), channels=channels)

    out = model(x, t, cond)
    print("input shape: ", x.shape, "\t output shape: ", out.shape)
    assert (
        x.shape == out.shape
    ), "output shape should be same as input shape as out_dim is not specified"

    out_dim = 8
    model2 = Unet(dim=32, dim_mults=(1, 2, 4), out_dim=out_dim, channels=channels)
    out2 = model2(x, t, cond)
    print("input shape: ", x.shape, "\t output shape: ", out2.shape)
    assert (
        out2.shape[1] == out_dim
    ), "number of channels in output should be same out_dim of the model"
