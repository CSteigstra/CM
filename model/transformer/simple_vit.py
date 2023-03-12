import torch
from torch import nn
import numpy as np

from torch import Tensor
from typing import Optional, Union
from einops import repeat

from einops import rearrange
from einops.layers.torch import Rearrange

# def rot_vec(vec, rad):
#     cs, sn = torch.cos(rad), torch.sin(rad)
#     return 

class RandeAPEv0(nn.Module):
    def __init__(self, dim: int, max_global_shift: float = 0.0, max_local_shift: float = 0.0,
                 max_global_scaling: float = 1.0, batch_first: bool = True):
        super().__init__()

        assert max_global_shift >= 0, f"""Max global shift is {max_global_shift},
        but should be >= 0."""
        assert max_local_shift >= 0, f"""Max local shift is {max_local_shift},
        but should be >= 0."""
        assert max_global_scaling >= 1, f"""Global scaling is {max_global_scaling},
        but should be >= 1."""
        assert dim % 2 == 0, f"""The number of channels should be even,
                                     but it is odd! # channels = {dim}."""

        self.max_global_shift = max_global_shift
        self.max_local_shift = max_local_shift
        self.max_global_scaling = max_global_scaling
        self.batch_first = batch_first

        half_channels = dim // 2
        rho = 10 ** torch.linspace(0, 1, half_channels)
        w_x = rho * torch.cos(torch.arange(half_channels))
        w_y = rho * torch.sin(torch.arange(half_channels))
        self.register_buffer('w_x', w_x)
        self.register_buffer('w_y', w_y)

        self.register_buffer('content_scale', Tensor([np.sqrt(dim)]))

    def forward(self, patches: Tensor) -> Tensor:
        _, h, w, dim = patches.shape
        return ((patches * self.content_scale) + self.compute_pos_emb(patches)).reshape(-1, h * w, dim)

    def compute_pos_emb(self, patches: Tensor) -> Tensor:
        if self.batch_first:
            batch_size, patches_x, patches_y, _ = patches.shape # b, x, y, c
        else:
            patches_x, patches_y, batch_size, _ = patches.shape # x, y, b, c
        device = patches.device

        x = torch.zeros([batch_size, patches_x, patches_y], device=device)
        y = torch.zeros([batch_size, patches_x, patches_y], device=device)
        x += torch.linspace(-1, 1, patches_x, device=device)[None, :, None]
        y += torch.linspace(-1, 1, patches_y, device=device)[None, None, :]

        x, y = self.augment_positions(x, y)

        phase = torch.pi * (self.w_x * x[:, :, :, None]
                            + self.w_y * y[:, :, :, None])
        pos_emb = torch.cat([torch.cos(phase), torch.sin(phase)], axis=-1)

        if not self.batch_first:
            pos_emb = rearrange(pos_emb, 'b x y c -> x y b c')

        return pos_emb

    def augment_positions(self, x: Tensor, y: Tensor):
        if self.training:
            batch_size, _, _ = x.shape

            if self.max_global_shift:
                x += (torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift,
                                                                   self.max_global_shift)
                     ).to(x.device)
                y += (torch.FloatTensor(batch_size, 1, 1).uniform_(-self.max_global_shift,
                                                                   self.max_global_shift)
                     ).to(y.device)

            if self.max_local_shift:
                diff_x = x[0, -1, 0] - x[0, -2, 0]
                diff_y = y[0, 0, -1] - y[0, 0, -2]
                epsilon_x = diff_x*self.max_local_shift
                epsilon_y = diff_y*self.max_local_shift
                x += torch.FloatTensor(x.shape).uniform_(-epsilon_x,
                                                         epsilon_x).to(x.device)
                y += torch.FloatTensor(y.shape).uniform_(-epsilon_y,
                                                         epsilon_y).to(y.device)

            if self.max_global_scaling > 1.0:
                log_l = np.log(self.max_global_scaling)
                lambdas = (torch.exp(torch.FloatTensor(batch_size, 1, 1).uniform_(-log_l,
                                                                                  log_l))
                          ).to(x.device)
                x *= lambdas
                y *= lambdas

        return x, y

    def set_content_scale(self, content_scale: float):
        self.content_scale = Tensor([content_scale])

class RandAPEv1(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # self.Wr = nn.Parameter(torch.normal(0, ))
        half_channels = dim // 4
        rho = 10 ** torch.linspace(0, 1, half_channels)
        w_x = rho * torch.cos(torch.arange(half_channels))
        w_y = rho * torch.sin(torch.arange(half_channels))

        self.register_buffer('w_x', w_x)
        self.register_buffer('w_y', w_y)

#     def __init_weights(self):
        # self.Wr
    def __call__(self, x):
        bs, h, w, dim, device, dtype = *x.shape, x.device, x.dtype
        # print(x.device, )
        # idx_x, idx_y = torch.arange(h, device=device), torch.arange(w, device=device)
        # y, x = torch.meshgrid(triangle_wave(idx_x, p=h, a=h/4)+1, triangle_wave(idx_y, p=w, a=w/4)+1, indexing='ij')
        # # y1, x1 = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing='ij')
        # # y1, x1 = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing='ij')
        # yx1 = torch.dstack((y.flatten(-2), x.flatten(-2)))

        # y, x = torch.meshgrid(triangle_wave(idx_x, p=h, cos=True)+1, triangle_wave(idx_y, p=h, cos=True)+1, indexing='ij')

        # yxz1 = self.__cylinder()
        # yx = torch.stack((cylinder(h, w, device=device), cylinder(w, h, device=device)), dim=1)
        # h_range = torch.arange(1, h+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(h,)).to(device=torch.device("cuda")), -0.5, 0.5)
        # w_range = torch.arange(1, w+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(w,)).to(device=torch.device("cuda")), -0.5, 0.5)
        
        # h_s = h / 2
        h_s = h / (2 * torch.pi)
        h_range = torch.arange(h, device=device) / h_s
        # h_range = torch.linspace(-1 , 1, h, device=device)

        # w_s = w / 2
        w_s = w / (2 * torch.pi)
        w_range = torch.arange(w, device=device) / w_s
        # w_range = torch.linspace(-1 , 1, w, device=device)


        y, x = torch.meshgrid(h_range, w_range, indexing='ij')

        # r_g_shift = torch.zeros(bs, 1, 1, device=device)
        # h_g_scale = torch.full((bs, 1, 1), h, device=device)
        # w_g_scale = torch.full((bs, 1, 1), w, device=device)
        r_shift = torch.zeros(bs, 1, 1, device=device)
        r_h_scale, r_w_scale = 1, 1
        r_h_shift, r_w_shift = 0, 0

        if self.training:
            # r_h_shift += torch.rand(bs, h, 1).to(device) * h_s * 0.4
            # r_w_shift += torch.rand(bs, 1, w).to(device) * w_s * 0.4
            r_shift += torch.rand(bs, 1, 1).to(device) * torch.pi
            r_h_shift += torch.rand(bs, h, 1).to(device) * (h_range[1] - h_range[0]) * 0.4
            r_w_shift += torch.rand(bs, 1, w).to(device) * (w_range[1] - w_range[0]) * 0.4
            # r_shift += torch.rand(bs, 1, 1).to(device) - 0.5
            # r_shift += torch.rand(bs, 1, 1, 1).to(device)
            dr_scale = torch.rand(bs, 1, 1).to(device) / 2 - 0.25
            # dr_scale = torch.randint((bs, 1, 1), -2, 3).to(device)
            r_h_scale += dr_scale
            r_w_scale += dr_scale
            # r_h_scale = torch.randint((bs, 1, 1), h-2, h+3).to(device)
            # r_w_scale = torch.randint((bs, 1, 1), w-2, w+3).to(device)

        # y, x = (y + r_h_shift + r_shift + 1) % 2 - 1, (x + r_w_shift + r_shift + 1) % 2 - 1
        y, x = y + r_h_shift + r_shift, x + r_w_shift + r_shift
        # y, x = y + r_shift, x + r_shift
        # y, x = y * r_h_scale, x * r_h_scale
        y1, y2 = torch.cos(y) * r_h_scale, torch.sin(y) * r_h_scale
        x1, x2 = torch.cos(x) * r_w_scale, torch.sin(x) * r_w_scale
        # y1, y2 = torch.cos(self.w_x * y.unsqueeze(-1)) * r_h_scale, torch.sin(self.w_y * y.unsqueeze(-1)) * r_h_scale
        # x1, x2 = torch.cos(self.w_x * x.unsqueeze(-1)) * r_w_scale, torch.sin(self.w_y * x.unsqueeze(-1)) * r_w_scale
        # pos_emb = torch.stack([y1, y2, x1, x2], axis=-1)
        # return pos_emb.reshape(-1, h * w, 4)


        phase1 = (self.w_x * y1.unsqueeze(-1)
                  + self.w_y * y2.unsqueeze(-1)) # * torch.pi
        phase2 = (self.w_x * x1.unsqueeze(-1)
                  + self.w_y * x2.unsqueeze(-1)) # * torch.pi
        # phase = torch.pi * (self.w_y * y.unsqueeze(-1)
        #                     + self.w_x * x.unsqueeze(-1))
        # phase1 = y1 + y2
        # phase2 = x1 + x2

        # pos_emb = torch.cat([torch.cos(phase), torch.sin(phase)], axis=-1)
        pos_emb = torch.cat([torch.cos(phase1), torch.sin(phase1),
                             torch.cos(phase2), torch.sin(phase2)], axis=-1)

        return pos_emb.reshape(-1, h * w, dim)
        # yx1 = torch.dstack((y1.flatten(-2), x1.flatten(-2)))
        # yx2 = torch.dstack((y2.flatten(-2), x2.flatten(-2)))

        # return Y.reshape(-1, h * w, dim)
        

class FourierPosMLP(nn.Module):
    def __init__(self, dim=64, h_dim=8, f_dim=64, scale=5, a=1):
        super().__init__()
        # self.Wr = nn.Parameter(torch.normal(0, ))
        self.Wr = torch.normal(0, scale, size=(f_dim//4, 2)).to(device=torch.device("cuda"))
        # self.W1 = nn.Linear(f_dim, h_dim)
        # self.W2 = nn.Linear(h_dim, dim)
        # self.scale = scale
        self.a = a

        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, dim//2)
        )

#     def __init_weights(self):
        # self.Wr
    def __call__(self, x):
        bs, h, w, dim, device, dtype = *x.shape, x.device, x.dtype
        # print(x.device, )
        # idx_x, idx_y = torch.arange(h, device=device), torch.arange(w, device=device)
        # y, x = torch.meshgrid(triangle_wave(idx_x, p=h, a=h/4)+1, triangle_wave(idx_y, p=w, a=w/4)+1, indexing='ij')
        # # y1, x1 = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing='ij')
        # # y1, x1 = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing='ij')
        # yx1 = torch.dstack((y.flatten(-2), x.flatten(-2)))

        # y, x = torch.meshgrid(triangle_wave(idx_x, p=h, cos=True)+1, triangle_wave(idx_y, p=h, cos=True)+1, indexing='ij')

        # yxz1 = self.__cylinder()
        # yx = torch.stack((cylinder(h, w, device=device), cylinder(w, h, device=device)), dim=1)
        # h_range = torch.arange(1, h+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(h,)).to(device=torch.device("cuda")), -0.5, 0.5)
        # w_range = torch.arange(1, w+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(w,)).to(device=torch.device("cuda")), -0.5, 0.5)
        

        h_s = h / (2 * torch.pi)
        h_range = torch.arange(h, device=device) / h_s

        w_s = w / (2 * torch.pi)
        w_range = torch.arange(w, device=device) / w_s

        y, x = torch.meshgrid(h_range, w_range, indexing='ij')

        # r_g_shift = torch.zeros(bs, 1, 1, device=device)
        # h_g_scale = torch.full((bs, 1, 1), h, device=device)
        # w_g_scale = torch.full((bs, 1, 1), w, device=device)
        r_shift = 0.
        r_h_scale, r_w_scale = h, w

        if self.training:
            # r_shift += torch.rand(bs, 1, 1).to(device) * torch.pi
            r_shift += torch.rand(bs, 1, 1).to(device) - 1

            dr_scale = torch.rand(bs, 1, 1).to(device) * 2 - 1
            # dr_scale = torch.randint((bs, 1, 1), -2, 3).to(device)
            r_h_scale += dr_scale
            r_w_scale += dr_scale
            # r_h_scale = torch.randint((bs, 1, 1), h-2, h+3).to(device)
            # r_w_scale = torch.randint((bs, 1, 1), w-2, w+3).to(device)

        y, x = y + r_shift, x + r_shift

        y1, y2 = torch.cos(y) * r_h_scale, torch.sin(y) * r_h_scale
        x1, x2 = torch.cos(x) * r_w_scale, torch.sin(x) * r_w_scale


        yx1 = torch.dstack((y1.flatten(-2), x1.flatten(-2)))
        yx = torch.dstack((y2.flatten(-2), x2.flatten(-2)))


        # d_angle = 360 / sz * torch.pi / 180 * torch.arange(sz, device=device)
        # d_angle = (360 * torch.pi) / (h * 180) * torch.arange(h, device=device)
        # d_angle = torch.arange(sz, device=device) * 360 * torch.pi 
        # r_offset = torch.rand(bs)
        # r_offset = torch.rand(bs, 1).to(device=torch.device("cuda")) * torch.pi
        # d_angle = d_angle + r_offset

        # cs, sn = torch.cos(d_angle), torch.sin(d_angle)

        #  - r * sn, r * cs

        # h_range = 
        
        # h_range = torch.arange(1, h+1, device=device)
        # w_range = torch.arange(1, w+1, device=device)
        # y, x = torch.meshgrid(h_range, w_range, indexing='ij')

        # r_offset = torch.rand(bs).to(device=torch.device("cuda"))
        # y = (y.flatten()[None, :] + r_offset)[:, :, None] * omega
        # x = (x.flatten()[None, :] + r_offset)[:, :, None] * omega

        # yx = torch.dstack((y.flatten(-2), x.flatten(-2))) + r_offset[:, None, None]
        # 
        # yx = 

        # yx = torch.cat()

        Y = 1 * torch.cat([torch.cos(1.*yx @ self.Wr.T),
                           torch.sin(1.*yx @ self.Wr.T)], dim=-1)

        Y = self.mlp(Y)

        return Y.reshape(-1, h * w, dim)


def circle(sz, device=None):
    r = sz / (2 * torch.pi)
    # d_angle = 360 / sz * torch.pi / 180 * torch.arange(sz, device=device)
    d_angle = (360 * torch.pi) / (sz * 180) * torch.arange(sz, device=device)
    cs, sn = torch.cos(d_angle), torch.sin(d_angle)

    return - r * sn, r * cs

# def meshgrid_tuples(t1, t2):
#     s0 = (1,)

def cylinder(h, w, device=None):
    # r = w / (2*torch.pi)
    r = 1
    d_angle = 360 / w
    d_angle_rad = d_angle * torch.pi / 180 * torch.arange(w, device=device)
    cs, sn = torch.cos(d_angle_rad), torch.sin(d_angle_rad)

    px = - r * sn 
    pz = r * cs

    # xz = np.stack((px, pz))
    # y, x = np.meshgrid(np.arange(h), (px, pz))

    s0 = (1,) * 2
    yxz = [torch.asarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
            for i, x in enumerate((torch.arange(h, device=device)/h, px, pz))]
    # yxz = [np.asanyarray()]
    y, x, z = torch.functional.broadcast_tensors(*yxz)
    yxz = torch.dstack((y.flatten(-2), x.flatten(-2), z.flatten(-2)))

    return yxz


class FourierPos():
    def __init__(self, a=1, scale=1, dim=128, zigzag=True):
        self.a = a
        # self.scale = scale
        self.b = torch.normal(0, scale, size=(dim//2, 2)).to(device=torch.device("cuda"))
        # self.b = torch.normal(0, scale, size=(dim//2, 4)).to(device=torch.device("cuda"))
        # self.map_fn = functorch.vmap(self.__input_encoder)
        # self.noise_param = (torch.tensor(0).to(device=torch.device("cuda")), torch.tensor(scale**0.2).to(device=torch.device("cuda")))

    def __call__(self, x):
        _, h, w, dim, device, dtype = *x.shape, x.device, x.dtype
        # print(x.shape)
        # y1, x1 = torch.meshgrid(triangle_wave(torch.arange(h, device=device), p=h), triangle_wave(torch.arange(w, device=device), p=w))
        # y2, x2 = torch.meshgrid(triangle_wave(torch.arange(h, device=device), p=h, cos=True), triangle_wave(torch.arange(w, device=device), p=w, cos=True))

        # y1, x1 = torch.meshgrid(zigzag(h, device=device), zigzag(w, device=device), indexing='ij')
        # y2, x2 = torch.meshgrid(zigzag(h, y_start=0.5, device=device), zigzag(w, y_start=0.5, device=device), indexing='ij')
        
        # yx = torch.dstack((y1.flatten(-2), x1.flatten(-2), y2.flatten(-2), x2.flatten(-2)))
        
        # return torch.cat([self.a * torch.sin((2.*torch.pi*yx1) @ self.b.T),
        #                   self.a * torch.sin((2.*torch.pi*yx2) @ self.b.T),
        #                   self.a * torch.cos((2.*torch.pi*yx1) @ self.b.T),
        #                   self.a * torch.cos((2.*torch.pi*yx2) @ self.b.T)], dim=-1).reshape(-1, h * w, dim)

        # y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing='ij')
        h_range = torch.arange(1, h+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(h,)).to(device=torch.device("cuda")), -0.5, 0.5)
        w_range = torch.arange(1, w+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(w,)).to(device=torch.device("cuda")), -0.5, 0.5)
        y, x = torch.meshgrid(h_range / (h+1), w_range / (w+1), indexing='ij')
        
        # y, x = torch.meshgrid(triangle_wave(torch.arange(h, device=device), p=h), triangle_wave(torch.arange(w, device=device), p=w))

        yx = torch.dstack((y.flatten(-2), x.flatten(-2)))

        # return torch.cat([self.a * triangle_wave(1.*yx @ self.b.T, cos=False),
        #                   self.a * triangle_wave(1.*yx @ self.b.T, cos=True)], dim=-1).reshape(-1, h * w, dim)

        return self.a * torch.cat([torch.sin((2.*torch.pi*yx) @ self.b.T), 
                                   torch.cos((2.*torch.pi*yx) @ self.b.T)], dim=-1).reshape(-1, h * w, dim)

# helpers

def triangle_wave(x, p=1, a=1, cos=False):
    x = x if cos else x - p / 4
    return 4 * a / p * torch.abs((x % p + p) % p - p / 2)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_lin_2d(patches, temperature=100):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device) / h, torch.arange(w, device = device) / h, indexing = 'ij')
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]



def posemb_sincos_2d(patches, temperature=1000, dtype=torch.float32, k=20):
    bs, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # y, x = cylinder(h, w, device=device)
    # y, x = torch.meshgrid(*circle(h, device=device), indexing='ij')
    # h_range = torch.arange(1, h+1, device=device) + torch.rand(h) - 0.5
    # w_range = torch.arange(1, w+1, device=device) + torch.rand(w) - 0.5
    # mu, sig = torch.tensor([0], device=device), torch.tensor([0], device=device)
    # h_range = torch.arange(1, h+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(h,)).to(device=torch.device("cuda")), -0.5, 0.5)
    # w_range = torch.arange(1, w+1, device=device) + torch.clamp(torch.normal(0, 0.25, size=(w,)).to(device=torch.device("cuda")), -0.5, 0.5)
    # y, x = torch.meshgrid(h_range / (h+1), w_range / (w+1), indexing='ij')
        
    # y1, x1 = cylinder(w, h)


    # y, x = torch.meshgrid(cylinder(), torch.arange(w, device = device), indexing = 'ij')
    y, x = torch.meshgrid(torch.arange(1, h+1, device=device) / (h+1), torch.arange(1, w+1, device=device) / (w+1), indexing='ij')

    # y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    r_offset = torch.rand(bs, 1).to(device=torch.device("cuda"))
    y = (y.flatten()[None, :] + r_offset)[:, :, None] * omega
    x = (x.flatten()[None, :] + r_offset)[:, :, None] * omega

    # y = y.flatten()[:, None] * omega[None, :]
    # x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)

    # return pe.type(dtype)


    # y1, x1 = torch.meshgrid(zigzag(h, device=device), zigzag(w, device=device), indexing = 'ij')
    # y2, x2 = torch.meshgrid(zigzag(h, y_start=0.5, device=device), zigzag(w, y_start=0.5, device=device), indexing = 'ij')
    
    # assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    # omega = torch.arange(dim // 8, device = device) / (dim // 8 - 1)
    # omega = 1. / (temperature ** omega)

    # y1 = y1.flatten()[:, None] * omega[None, :]
    # x1 = x1.flatten()[:, None] * omega[None, :]
    # y2 = y2.flatten()[:, None] * omega[None, :]
    # x2 = x2.flatten()[:, None] * omega[None, :]
    # pe = torch.cat((x1.sin(), x1.cos(), y1.sin(), y1.cos(), x2.sin(), x2.cos(), y2.sin(), y2.cos()), dim = 1)

    return pe.type(dtype)

# classes

# def 

# class posemb_learnable():
#     def __init__(self, n_patches=(8, 8), dim=64):
#         n_x, n_y = n_patches
        
def zigzag(n, y_start=0, y_max=1, device=None):
    x = torch.linspace(0, y_max*2, n+1, device=device)[:-1] + y_start
    # Bounce x from upper bound
    mask_g = x>y_max
    x[mask_g] = -(x[mask_g] - y_max) + y_max
    # Bounce x from lower bound
    mask_l = x<0
    x[mask_l] = torch.abs(x[mask_l])
    # return x
    return (x * n/2).round() + 1
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, attn_drop=0., proj_drop=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.proj_drop(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, drop=0., attn_drop=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, attn_drop=attn_drop, proj_drop=drop),
                FeedForward(dim, mlp_dim, drop=drop)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, dim, 2, stride=2),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, drop=drop_rate, attn_drop=attn_drop_rate)

        # self.posemb = FourierPos(dim=dim)
        # self.posemb = FourierPosMLP(dim=dim, f_dim=dim)
        # self.posemb = posemb_sincos_2d
        # self.posemb = RandeAPEv0(dim=dim, max_global_shift=0., max_local_shift=0., max_global_scaling=1.)
        self.posemb = RandAPEv1(dim=dim)
        self.posdrop = nn.Dropout(drop_rate)
        # self.posemb = posemb_learnable(image_height//patch_height, image_width//patch_width, dim=dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype
        # print(img.shape)
        x = self.to_patch_embedding(img)
        # print(x.shape)
        pe = self.posemb(x)
        pe = self.posdrop(pe)
        x = rearrange(x, 'b ... d -> b (...) d')
        # pe.repeat(x.shape[0], 1, 1)
        # print(pe.shape, x.shape)
        x = x + pe
        # x = torch.cat([x, pe], axis=-1)
        # x = x + pe
        # print(pe.shape, img.shape, torch.flatten(img, start_dim=1).shape, x.shape)
        # x = torch.flatten(img, start_dim=1)
        # pe = torch.arange()
        # x = pe.repeat(x.shape[0], 1, 1)
        # x[:, :, -1] = torch.flatten(img, start_dim=1)
        # pe[:, -1] = torch.flatten(img, start_dim=1)
        # print(pe.shape)
        # pe[:, -1] = img.squeeze().view(-1, h * w)
        # x = pe
        # pe = self.inpu
        # print(x.shape, pe.shape)
        # # print(x.shape)
        # x = x + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)