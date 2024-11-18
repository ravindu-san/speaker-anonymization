from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum

from position_embeddings import SinusoidalPositionEmbeddings


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)  # TODO weight standardized conv
        self.norm = nn.GroupNorm(groups, dim_out)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # time embedding
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if (time_emb_dim is not None) else None)
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if (self.mlp is not None) and (time_emb is not None):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        # self.dim = dim
        self.heads = heads
        # self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Conv2d(dim, dim_head * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_head * heads, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_head * heads, 1, bias=False)
        self.to_out = nn.Conv2d(dim_head * heads, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads) * self.scale
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)

        qk = einsum('b h d i, b h d j -> b h i j', q, k)
        qk = qk - qk.amax(dim=-1, keepdim=True).detach()
        attention = qk.softmax(dim=-1)

        att_val = einsum('b h i j, b h d j -> b h i d', attention, v)
        out = rearrange(att_val, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Upsample(dim, dim_out=None):
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, dim_out, 3, padding=1))


def Downsample(dim, dim_out=None):
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2), nn.Conv2d(dim * 4, dim_out, 1))


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.normalize = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.normalize(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 3, 4), channels=1, resnet_block_groups=4,
                 *args, **kwargs):
        super().__init__()
        if init_dim is None:
            init_dim = dim
        self.init_conv = nn.Conv2d(channels, init_dim, 1)

        time_dim = 4 * dim
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        dims = [init_dim, *[i * dim for i in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.downs = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList(
                [ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                 ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                 Residual(PreNorm(dim_in, Attention(dim_in))),
                 Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)]))

        mid_dim = dims[-1]
        self.mid_resnet_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attention = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_resnet_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        self.ups = nn.ModuleList([])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_in + dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in + dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_out, Attention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)

            ]))

        self.final_resnet_block = ResnetBlock(2 * dim, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        if out_dim is None:
            out_dim = channels
        self.final_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for resnet1, resnet2, attention, downsample in self.downs:
            x = resnet1(x, t)
            h.append(x)

            x = resnet2(x, t)
            x = attention(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_resnet_block1(x)
        x = self.mid_attention(x)
        x = self.mid_resnet_block2(x)

        for resnet1, resnet2, attention, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = resnet2(x, t)

            x = attention(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_resnet_block(x, t)
        return self.final_conv(x)


if __name__ == "__main__":
    batch_size = 8
    channels = 1
    time_steps = 100

    x = torch.rand((batch_size, channels, 32, 32))
    t = torch.randint(0, time_steps, (batch_size,)).long()
    model = Unet(dim=32, dim_mults=(1, 2, 4), channels=channels)

    out = model(x,t)
    print("input shape: ", x.shape, "\t output shape: ", out.shape)
    assert x.shape == out.shape, "output shape should be same as input shape as out_dim is not specified"

    out_dim = 8
    model2 = Unet(dim=32, dim_mults=(1, 2, 4), out_dim=out_dim, channels=channels)
    out2 = model2(x,t)
    print("input shape: ", x.shape, "\t output shape: ", out2.shape)
    assert out2.shape[1] == out_dim, "number of channels in output should be same out_dim of the model"

