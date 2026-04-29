from typing import cast,TypeAlias
from collections import namedtuple, deque
from itertools import pairwise

import torch
from torch import nn
from torchvision.ops import MLP

from tqdm import tqdm

#--------------------------------------------------------------------
Tensor: TypeAlias = torch.Tensor
Res_Ch = namedtuple('Res_Ch', ['i', 'm', 'o'])
Layer_Ch = namedtuple('Layer_Ch', ['i', 'm', 'e', 'o'])
"input, middle, enhance, output"

Arch = tuple[Layer_Ch, ...]

#--------------------------------------------------------------------
ARCH: Arch = (
    Layer_Ch(64,  64,  64, 64),
    Layer_Ch(64,  128, 0, 128),
    Layer_Ch(128, 256, 0, 256),
    Layer_Ch(256, 256, 0, 512)
) # Diffusion/unet_diffusion.png

TIME_DIM = 512
TIMESTEP = 1000
TAU = list(range(0, TIMESTEP+1, 40))

#--------------------------------------------------------------------
def sinPosEmbed( time_step = TIMESTEP, dim = TIME_DIM):
    half_dim = dim // 2                       # i in range(half_dim)
    pos_arr = torch.arange(time_step)
    lg_cycle = -torch.arange(half_dim) / half_dim
    cycle = 10000 ** lg_cycle                    # <half_dim>
    raw_embeddings = torch.outer(pos_arr, cycle) # <timestep, half_dim>

    embeddings = torch.zeros((time_step,dim)) # <timestep, 2*half_dim>
    embeddings[:,0::2] = raw_embeddings.sin() # slice assignment
    embeddings[:,1::2] = raw_embeddings.cos()
    
    return embeddings


class ResBlock(nn.Module):
    def __init__(self, ch: Res_Ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.ch = ch
        self.norm1 = nn.GroupNorm(num_groups, ch.i)
        self.conv1 = nn.Conv2d(ch.i, ch.m, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, ch.m)
        self.conv2 = nn.Conv2d(ch.m, ch.o, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, ch.m)
        self.act = nn.SiLU()  # or nn.SiLU(inplace=True)
        self.shortcut = nn.Conv2d(ch.i, ch.o, 1)

    def forward(self, x: Tensor, t_emb):
        B, C, H, W = x.shape
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        t_emb_local: Tensor = self.time_proj(t_emb) # <B, ch.m>
        h = h + t_emb_local.view(B, self.ch.m, 1, 1) # <B, ch.m, H_, W_>

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.proj = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        h: Tensor = self.norm(x)
        h = h.flatten(2).transpose(1,2) # <B, HW, C>
        h, attn_weight = self.attn(h, h, h)
        h = h.transpose(1,2).reshape((B,C,H,W)) # <B, C, H, W>

        return x + self.proj(h)


class DownLayer(nn.Module):
    def __init__(self, ch: Layer_Ch, time_emb_dim, is_tail: bool = False):
        super().__init__()
        self.ch =  ch
        self.is_tail = is_tail

        ch_res1 = Res_Ch(ch.i, ch.m, ch.m)

        if ch.e != 0:
            ch_res2 = Res_Ch(ch.m, ch.e, ch.e)
            ch_res3 = Res_Ch(ch.e, ch.o, ch.o)
            self.res3 = ResBlock(ch_res3, time_emb_dim)
        else:
            ch_res2 = Res_Ch(ch.m, ch.o, ch.o)

        self.res1 = ResBlock(ch_res1, time_emb_dim)
        self.res2 = ResBlock(ch_res2, time_emb_dim)

        self.down = nn.Conv2d(ch.o, ch.o, 3, 2, 1)
    
    def forward(self, x, t_emb):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        if self.ch.e != 0: h = self.res3(h, t_emb)

        out = h if self.is_tail else self.down(h)
        return out, h


class UpLayer(nn.Module):
    def __init__(self, ch: Layer_Ch, time_emb_dim, is_head: bool = False):
        super().__init__()
        self.ch = ch
        self.is_head = is_head
        self.ch_cat = 2*ch.i # with the concated tensor

        ch_res1 = Res_Ch(self.ch_cat, self.ch_cat, ch.m)

        if ch.e != 0:
            ch_res2 = Res_Ch(ch.m, ch.m, ch.e)
            ch_res3 = Res_Ch(ch.e, ch.e, ch.o)
            self.res3 = ResBlock(ch_res3, time_emb_dim)
        else:
            ch_res2 = Res_Ch(ch.m, ch.m, ch.o)
        
        self.up = nn.ConvTranspose2d(ch.i, ch.i, 4, 2, 1)
        self.res1 = ResBlock(ch_res1, time_emb_dim)
        self.res2 = ResBlock(ch_res2, time_emb_dim)
    
    def forward(self, x, concat, t_emb):
        h = x if self.is_head else self.up(x)
        h = torch.cat((h,concat), dim=1) # concat in C dim
        h = self.res1(h, t_emb)
        h = self.res2(h, t_emb)
        if self.ch.e != 0: h = self.res3(h, t_emb)
        return h


class Bridge(nn.Module):
    def __init__(self, ch: Layer_Ch, time_emb_dim):
        super().__init__()
        self.time_dim = ted = time_emb_dim
        ch_res1 = Res_Ch(ch.i, ch.m, ch.m)
        ch_res2 = Res_Ch(ch.m, ch.m, ch.o)

        self.res1 = ResBlock(ch_res1, ted)
        self.attn = AttnBlock(ch.m)
        self.res2 = ResBlock(ch_res2, ted)
    
    def forward(self, x, t_emb):
        h = self.res1(x, t_emb)
        h = self.attn(h)
        h = self.res2(h, t_emb)
        return h

class Unet(nn.Module):
    def __init__(self, arch: Arch, time_emb_dim):
        super().__init__()
        self.num_layers = len(arch) # default 4 (ch:64->512)
        self.encoder_arch = arch
        self.decoder_arch = tuple(Layer_Ch(c.o, c.m, c.e, c.i) for c in reversed(arch))

        self.time_dim = ted = time_emb_dim
        self.store = deque([], self.num_layers)
        self.input_conv = nn.Conv2d(3, arch[0].i, 3, padding=1)
        self.out_conv = nn.Conv2d(arch[0].i, 3, 3, padding=1)

        self.encoder = nn.ModuleList( DownLayer(io_ch, ted) for io_ch in self.encoder_arch)
        tail_layer = cast(DownLayer, self.encoder[-1])
        tail_layer.is_tail = True

        self.decoder = nn.ModuleList( UpLayer(io_ch, ted) for io_ch in self.decoder_arch)
        head_layer: UpLayer = cast(UpLayer, self.decoder[0])
        head_layer.is_head = True
        # The decoder has an architecture symmetric to that of the encoder.

        ch_bridge = Layer_Ch(arch[-1].o, arch[-1].o, arch[-1].o, arch[-1].o) # the input dim for attn block
        self.bridge = Bridge(ch_bridge, time_emb_dim)
    
    def forward(self, x, t_emb):
        x = self.input_conv(x)
        for down_block in self.encoder:
            x,skip = down_block(x, t_emb)
            self.store.append(skip)
        h = self.bridge(x, t_emb)
        for up_block in self.decoder:
            skip = self.store.pop()
            h = up_block(h, skip, t_emb)
        y = self.out_conv(h)
        return y


class DDIM(nn.Module):
    def __init__(self, arch, dim, time_emb_dim):
        super().__init__()
        self.arch = arch

        self.T = TIMESTEP
        self.time_emb_dim = ted = time_emb_dim
        self.time_emb = sinPosEmbed(self.T, ted)

        s = 0.008
        timesteps = torch.arange(self.T + 1, dtype=torch.float32) / self.T # 0...T
        alpha_bar = torch.cos((timesteps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # 0...T, alpha_bar[0] = 1

        betas = torch.zeros_like(alpha_bar)
        betas[1:] = 1 - (alpha_bar[1:] / alpha_bar[:-1])  # 0...T
        self.alpha_bar = alpha_bar
        self.betas = betas

        # self.beta = torch.linspace(0.0001, 0.02, self.T)
        # self.alpha = 1-self.beta
        # self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sigma = torch.sqrt(self.betas)  # DDPM 使用 σ_t = sqrt(β_t)

        self.denoiser = Unet(arch, ted)
        self.time_mlp = MLP(ted, [ted,ted], activation_layer=nn.SiLU)

    def noise_predicter(self, xt, t: Tensor):
        t_emb = self.time_emb[t-1]  # t>0
        t_emb = self.time_mlp(t_emb)
        eps_pred = self.denoiser(xt, t_emb)
        return eps_pred
    

    @torch.no_grad()
    def sample(self, shape, eta=0.4, tau: list[int]=TAU):
        """
        DDIM Sample:\\
        sigma_t = eta * sqrt[ (1-alpha_{t-1}) / (1-alpha_t) ] * beta_t; 0<= eta <= 1\\
        tau = [tau_1, tau_2, ..., tau_k] where tau_i in [0, T] and tau_1 = 0 < tau_2 < ... < tau_k = T
        """

        B, C, H, W = shape
        tau = sorted(set(tau))

        assert tau[0] == 0, "tau_1 should be 0"
        assert tau[-1] == self.T, "tau_k should be T"
        assert 0.0 <= eta <= 1.0, "eta should be in [0.0, 1.0]."

        x = torch.randn(shape)
        steps = pairwise(reversed(tau)) # (tau_k, tau_{k-1}), (tau_{k-1}, tau_{k-2}), ..., (tau_2, tau_1)

        for t_cur, t_pre in tqdm(steps, "DDIM Sample"):
            t_batch = torch.full((B,), t_cur, dtype=torch.long)

            eps_pred = self.noise_predicter(x, t_batch)
            ac = self.alpha_bar[t_cur]  # alpha_bar_cur
            ap = self.alpha_bar[t_pre]  # alpha_bar_pre
            st = eta * self.sigma[t_cur]  # eta * orign_sigma_t

            z = torch.randn(shape)

            # DDIM Sample
            x0_pred = ( x - torch.sqrt(1-ac)*eps_pred ) / torch.sqrt(ac)
            mean_x = torch.sqrt(ap)*x0_pred + torch.sqrt(1-ap-st**2)*eps_pred
            x = mean_x + st*z

        return x
