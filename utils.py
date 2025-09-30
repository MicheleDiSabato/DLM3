import torch
import torch.nn as nn
import math
from torchdiffeq import odeint

class SinusoidalTimeEmbedding(nn.Module):
    """
    Class to embed time
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) in [0,1]
        returns: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
        )
        # [B, half]
        angles = t[:, None] * freqs[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1), mode="constant")
        return emb

class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = SinusoidalTimeEmbedding(dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim*2),
            nn.SiLU(),
            nn.Linear(dim*2, dim)
        )
    def forward(self, t):
        return self.net(self.emb(t))
    

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.block1 = ResBlock(in_ch,  out_ch, t_dim)
        self.block2 = ResBlock(out_ch, out_ch, t_dim)
        self.down   = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        #x = self.block2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block1 = ResBlock(out_ch + skip_ch, out_ch, t_dim)  # <--  in_ch = out_ch+skip_ch
        self.block2 = ResBlock(out_ch, out_ch, t_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)                     # (B, out_ch, H, W)
        assert x.shape[2:] == skip.shape[2:], f"Upsample mismatch: x {x.shape}, skip {skip.shape}"
        x = torch.cat([x, skip], dim=1)    # (B, out_ch+skip_ch, H, W)
        x = self.block1(x, t_emb)
        #x = self.block2(x, t_emb)
        return x

class UNetVectorField(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, levels=3, t_dim=256):
        super().__init__()
        self.t_mlp = TimeMLP(t_dim)
        chs = [base_ch * (2**i) for i in range(levels)]       # [64,128,256]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.downs = nn.ModuleList([Down(chs[i], chs[i+1], t_dim) for i in range(levels-1)])
        self.mid1 = ResBlock(chs[-1], chs[-1], t_dim)
        self.mid2 = ResBlock(chs[-1], chs[-1], t_dim)
        self.ups = nn.ModuleList([
            Up(chs[i+1], chs[i+1], chs[i], t_dim)   # in_ch, skip_ch, out_ch
            for i in reversed(range(levels-1))
        ])
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_ch, 3, padding=1)

    def forward(self, x_t, t):
        t_emb = self.t_mlp(t)
        x = self.in_conv(x_t)
        skips = []
        for d in self.downs:
            x, s = d(x, t_emb); skips.append(s)
        x = self.mid1(x, t_emb); x = self.mid2(x, t_emb)
        for u in self.ups:
            s = skips.pop()
            x = u(x, s, t_emb)
        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


class RHSFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128,dim)
        self.tanh = nn.Tanh()

    def forward(self, t, x):
        x = self.fc1(x)
    #    x = self.tanh(x)
        x = self.fc2(x)
        return x


class CNFModel(nn.Module):
    def __init__(self, dim, T=1):
        super().__init__()
        self.T = T
        self.dim = dim
        self.f = RHSFunc(dim)
        self.base_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)) # Normal distribution with mean 0 and std 1
        self.integration_time = torch.tensor([T, 0]).float()
        
    def forward(self, x):
        """
        The model takes a batch x of our data. x has some distribution that we
        would like to learn. Below, x is transformed into z0, which is a sample
        of our base distribution N(0,1). We integrate backwards since z0 is a
        sample in the "original" distribution.
        Note that df/dz0 is set to zero, so the RHS of eq. (6) in 
        https://arxiv.org/pdf/1806.07366 becocomes log(p(x))=log(p(z0)), meaning
        that the model is not accounting for the volume change in the likelihood
        computation. While this is not entirely accurate, the network is still 
        learning a vector field f describing where samples z0 in the latent 
        space end up, yielding positive results in the end.
        """
        z0 = odeint(self.f, x, self.integration_time)[-1] # Integrating over f from 1 to 0
        return z0

    def log_prob(self, x):
        z0 = self.forward(x)
        return self.base_dist.log_prob(z0).sum(dim=1)
