
# Exact path-likelihood latent sequence model with conditional observation flow:
#   z_t = f_t(y_t),  y_t follows a diagonal Ornstein–Uhlenbeck process, conditioned on y_0 
#   p(z_0) is a learned diagonal Gaussian in z-space.
#   x_t = g(u_t ; z_t),  u_t ~ N(0, I)  (conditional RealNVP observation)
# Loss:
#   E_q[ sum_t log p(x_t|z_t) + log p(z0) + log p(z_{1:T}|z0) - log q(z_{0:T}|x) ]
# Posterior q(z_t|x, t) is an affine Gaussian from a GRU encoder with time-adaptive context. This can be obviously replaced with whatever archecturally.

from __future__ import annotations
from typing import Tuple, Any, Sequence
from abc import ABC, abstractmethod

import math
import torch
from torch import nn, Tensor
from torch import distributions as D

import matplotlib.pyplot as plt
from tqdm import trange


# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed: int = 1234):
    import random, os
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def visualise_data(xs: Tensor, filename: str = "figure.jpg"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for xs_i in xs:
        ax.plot(xs_i[:, 0].detach().cpu(), xs_i[:, 1].detach().cpu(), xs_i[:, 2].detach().cpu())
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel('$z_1$', labelpad=0., fontsize=16)
    ax.set_ylabel('$z_2$', labelpad=.5, fontsize=16)
    ax.set_zlabel('$z_3$', labelpad=0., horizontalalignment='center', fontsize=16)
    plt.savefig(filename, format='jpg', dpi=300)
    plt.close(fig)


# -------------------------------
# Synthetic data generator (stochastic Lorenz -> normalize + noise)
# -------------------------------
class SDE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor: ...
    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor: ...
    def forward(self, z: Tensor, t: Tensor, *args: Any) -> Tuple[Tensor, Tensor]:
        return self.drift(z, t, *args), self.vol(z, t, *args)


class StochasticLorenzSDE(SDE):
    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.15, .15, .15)):
        super().__init__(); self.a = a; self.b = b
    def drift(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        a1, a2, a3 = self.a
        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)
    def vol(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        b1, b2, b3 = self.b
        return torch.cat([x1*b1, x2*b2, x3*b3], dim=1)


@torch.no_grad()
def solve_sde(
    sde: SDE,
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5
    path = [z]
    for t in tt:
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        path.append(z)
    return torch.stack(path)  # (n_steps+1, B, d)


def gen_data(
    batch_size: int,
    ts: float,
    tf: float,
    n_steps: int,
    noise_std: float,
    n_inner_steps: int = 100
) -> Tuple[Tensor, Tensor]:
    sde = StochasticLorenzSDE()
    z0 = torch.randn(batch_size, 3)
    zs = solve_sde(sde, z0, ts, tf, n_steps=n_steps * n_inner_steps)  # (N,B,3)
    zs = zs[::n_inner_steps].permute(1, 0, 2)                         # (B,T,3)
    mean, std = zs.mean(dim=(0, 1)), zs.std(dim=(0, 1))
    xs = (zs - mean) / std + noise_std * torch.randn_like(zs)
    ts_grid = torch.linspace(ts, tf, n_steps + 1, device=xs.device)[None, :, None].repeat(batch_size, 1, 1)
    return xs, ts_grid  # (B,T,3), (B,T,1) in [ts, tf]


# -------------------------------
# Posterior q(z_t | x, t): GRU encoder + time-adaptive affine head
# -------------------------------
class PosteriorEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, x: Tensor) -> Tensor:
        out, h = self.gru(x)
        return torch.cat([h[0, :, None], out], dim=1)  # (B, 1+T, H)


class PosteriorAffine(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, init_logstd: float = -0.5):
        super().__init__()
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 2 * latent_size),
        )
        self.sm = nn.Softmax(dim=-1)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.net[-1]
        with torch.no_grad():
            last.bias[self.latent_size:] = init_logstd
    def get_coeffs(self, ctx: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        l = ctx.shape[1] - 1
        h, out = ctx[:, 0], ctx[:, 1:]
        ts = torch.linspace(0, 1, l, device=ctx.device, dtype=ctx.dtype)[None, :]
        c = self.sm(-(l * (ts - t)) ** 2)
        out = (out * c[:, :, None]).sum(dim=1)
        ctx_t = torch.cat([h + out, t], dim=1)
        m, log_s = self.net(ctx_t).chunk(2, dim=1)
        s = torch.exp(log_s)
        return m, s
    def forward(self, ctx: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.get_coeffs(ctx, t)


# -------------------------------
# Flow utilities
# -------------------------------
def diag_gauss_logprob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp_min(1e-10)
    log2pi = math.log(2.0 * math.pi)
    return -0.5 * (torch.log(var) + log2pi + (x - mean) ** 2 / var).sum(dim=-1)


class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 64, max_freq: float = 16.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0., math.log(max_freq), dim // 2))
        self.register_buffer("freqs", freqs)
    def forward(self, t: Tensor) -> Tensor:   # t: (...,1) in [0,1]
        ang = t.to(self.freqs.dtype) * self.freqs.view(*([1]*(t.dim()-1)), -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# -------------------------------
# Time-conditional flow for latent prior: z_t = f_t(y_t)
# -------------------------------
class AffineCoupling(nn.Module):
    def __init__(self, D: int, hidden: int, tdim: int, mask: Tensor, clamp: float = 1.5):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1).float())
        self.temb = TimeEmbed(tdim)
        self.net = MLP(in_dim=D + tdim, hidden=hidden, out_dim=2 * D)
        self.clamp = clamp
        # identity init
        last = None
        for m in self.net.net:
            if isinstance(m, nn.Linear):
                last = m
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    def forward(self, x: Tensor, t: Tensor, inverse: bool = False) -> Tuple[Tensor, Tensor]:
        m = self.mask
        xa = x * m
        h = torch.cat([xa, self.temb(t)], dim=-1)
        s, b = self.net(h).chunk(2, dim=-1)
        s = torch.tanh(s) * self.clamp
        comp = (1. - m)
        s = s * comp; b = b * comp
        if not inverse:
            y = xa + comp * (x * torch.exp(s) + b)
            logdet = s.sum(dim=-1)
        else:
            y = xa + comp * ((x - b) * torch.exp(-s))
            logdet = (-s).sum(dim=-1)
        return y, logdet


class TimeCondRealNVP(nn.Module):
    def __init__(self, D: int, hidden: int = 160, n_layers: int = 6, tdim: int = 64):
        super().__init__()
        base_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(D)])
        masks = [(base_mask if k % 2 == 0 else 1 - base_mask) for k in range(n_layers)]
        self.layers = nn.ModuleList([AffineCoupling(D, hidden, tdim, m) for m in masks])
    def forward(self, y: Tensor, t: Tensor) -> Tensor:  # y -> z
        x = y
        for layer in self.layers:
            x, _ = layer(x, t, inverse=False)
        return x
    def inverse(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:  # z -> y
        y = x
        logdet = x.new_zeros(x.size(0))
        for layer in reversed(self.layers):
            y, ld = layer(y, t, inverse=True)
            logdet = logdet + ld
        return y, logdet


# -------------------------------
# Conditional observation flow: x = g(u ; z),  u ~ N(0,I)
# -------------------------------
class CondAffineCoupling(nn.Module):
    def __init__(self, D: int, hidden: int, cond_dim: int, mask: Tensor, clamp: float = 1.5):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1).float())
        # conditioning by concatenation of masked x and z
        self.net = MLP(in_dim=D + cond_dim, hidden=hidden, out_dim=2 * D)
        self.clamp = clamp
        # identity init
        last = None
        for m in self.net.net:
            if isinstance(m, nn.Linear):
                last = m
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    def forward(self, x: Tensor, z: Tensor, inverse: bool = False) -> Tuple[Tensor, Tensor]:
        m = self.mask
        xa = x * m
        h = torch.cat([xa, z], dim=-1)         # condition on z only
        s, b = self.net(h).chunk(2, dim=-1)
        s = torch.tanh(s) * self.clamp
        comp = (1. - m)
        s = s * comp; b = b * comp
        if not inverse:
            y = xa + comp * (x * torch.exp(s) + b)
            logdet = s.sum(dim=-1)
        else:
            y = xa + comp * ((x - b) * torch.exp(-s))
            logdet = (-s).sum(dim=-1)
        return y, logdet


class CondRealNVP(nn.Module):
    def __init__(self, D: int, cond_dim: int, hidden: int = 128, n_layers: int = 6):
        super().__init__()
        base_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(D)])
        masks = [(base_mask if k % 2 == 0 else 1 - base_mask) for k in range(n_layers)]
        self.layers = nn.ModuleList([CondAffineCoupling(D, hidden, cond_dim, m) for m in masks])
    # u -> x (generative)
    def forward(self, u: Tensor, z: Tensor) -> Tensor:
        x = u
        for layer in self.layers:
            x, _ = layer(x, z, inverse=False)
        return x
    # x -> u and sum log|det J|
    def inverse(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        u = x
        logdet = x.new_zeros(x.size(0))
        for layer in reversed(self.layers):
            u, ld = layer(u, z, inverse=True)
            logdet = logdet + ld
        return u, logdet


class ObservationFlow(nn.Module):
    """
    Conditional RealNVP observation model p(x|z):
      x = g(u ; z), u ~ N(0,I)
      log p(x|z) = log N(u;0,I) + log|det ∂g^{-1}/∂x|
    """
    def __init__(self, data_size: int, cond_dim: int, hidden: int = 128, n_layers: int = 6):
        super().__init__()
        self.flow = CondRealNVP(D=data_size, cond_dim=cond_dim, hidden=hidden, n_layers=n_layers)
    def log_prob(self, x: Tensor, z: Tensor) -> Tensor:
        u, logdet = self.flow.inverse(x, z)                             # (B, Dx), (B,)
        mean0 = torch.zeros_like(u); var1 = torch.ones_like(u)
        log_pu = diag_gauss_logprob(u, mean0, var1)                     # (B,)
        return log_pu + logdet
    @torch.no_grad()
    def sample(self, z: Tensor, n: int = None) -> Tensor:
        """
        If n is None, draws one sample per z. Otherwise, tiles z to (n * batch) and samples.
        """
        if n is None:
            u = torch.randn_like(z[..., :1]).repeat(1, z.size(-1)*0 + 1)  # placeholder to get device/dtype
        B, Dz = z.shape
        Dx = self.flow.layers[0].mask.numel()
        if n is None:
            u = torch.randn(B, Dx, device=z.device, dtype=z.dtype)
            return self.flow.forward(u, z)
        else:
            zt = z.repeat_interleave(n, dim=0)
            u = torch.randn(B * n, Dx, device=z.device, dtype=z.dtype)
            return self.flow.forward(u, zt)


# -------------------------------
# Diagonal OU base and latent flow prior
# -------------------------------
class DiagOUSDE(nn.Module):
    def __init__(self, D: int, init_mu: float = 0.0, init_logk: float = -0.7, init_logs: float = -1.0):
        super().__init__()
        self.mu = nn.Parameter(torch.full((D,), init_mu))
        self.log_kappa = nn.Parameter(torch.full((D,), init_logk))
        self.log_sigma = nn.Parameter(torch.full((D,), init_logs))
    def _params(self):
        kappa = torch.nn.functional.softplus(self.log_kappa) + 1e-6
        sigma = torch.nn.functional.softplus(self.log_sigma) + 1e-6
        mu = self.mu
        return mu, kappa, sigma
    @torch.no_grad()
    def sample_path_cond(self, ts_grid: Tensor, y0: Tensor) -> Tensor:
        device = ts_grid.device
        n, D = y0.shape
        mu, kappa, sigma = self._params()
        mu = mu.to(device); kappa = kappa.to(device); sigma = sigma.to(device)
        T = ts_grid.size(0)
        y = torch.zeros(n, T, D, device=device)
        y[:, 0, :] = y0
        for k in range(T - 1):
            dt = (ts_grid[k+1] - ts_grid[k]).clamp(min=1e-6)
            Ad = torch.exp(-kappa * dt)
            mean = mu + Ad * (y[:, k, :] - mu)
            q = (sigma**2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
            y[:, k+1, :] = mean + torch.randn_like(mean) * q.sqrt()
        return y
    def path_log_prob_cond(self, y: Tensor, ts_batch: Tensor, y0_given: Tensor) -> Tensor:
        B, T, D = y.shape
        if ts_batch.dim() == 2:
            ts_batch = ts_batch[None, :, :].expand(B, -1, -1)
        mu, kappa, sigma = self._params()
        mu = mu[None, None, :].to(y.device)
        kappa = kappa[None, None, :].to(y.device)
        sigma = sigma[None, None, :].to(y.device)
        t0, t1 = ts_batch[:, :-1, :], ts_batch[:, 1:, :]
        dt = (t1 - t0).clamp(min=1e-6)
        Ad = torch.exp(-kappa * dt)
        prev = torch.cat([y0_given[:, None, :], y[:, 1:T-1, :]], dim=1)
        mean = mu + Ad * (prev - mu)
        q = (sigma**2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
        y_next = y[:, 1:, :]
        lp_trans = diag_gauss_logprob(y_next, mean, q).sum(dim=1)
        return lp_trans


class NF_SDE_Model(nn.Module):
    def __init__(self, D: int, hidden: int = 160, n_layers: int = 6, tdim: int = 64,
                 t_min: float = 0.0, t_max: float = 1.0):
        super().__init__()
        self.flow = TimeCondRealNVP(D, hidden=hidden, n_layers=n_layers, tdim=tdim)
        self.ou = DiagOUSDE(D)
        self.register_buffer("t_min", torch.tensor(float(t_min)))
        self.register_buffer("t_max", torch.tensor(float(t_max)))
    def _cond_t(self, t: Tensor) -> Tensor:
        denom = (self.t_max - self.t_min).clamp(min=1e-8)
        tau = (t - self.t_min) / denom
        return tau.clamp(0., 1.)
    def log_prob_paths_cond(self, z_path: Tensor, ts_batch: Tensor, z0: Tensor) -> Tensor:
        B, T, D = z_path.shape
        if ts_batch.dim() == 2:
            ts_batch = ts_batch[None, :, :].expand(B, -1, -1)
        zf = z_path.reshape(B * T, D)
        tf = ts_batch.reshape(B * T, 1)
        tf_cond = self._cond_t(tf)
        yf, logdetf = self.flow.inverse(zf, tf_cond)
        y = yf.reshape(B, T, D)
        logdet_seq = logdetf.reshape(B, T)[:, 1:].sum(dim=1)
        t0 = ts_batch[:, 0, :]
        t0_cond = self._cond_t(t0)
        y0, _ = self.flow.inverse(z0, t0_cond)
        lp_trans = self.ou.path_log_prob_cond(y, ts_batch, y0)
        return lp_trans + logdet_seq
    @torch.no_grad()
    def sample_paths_cond(self, ts_grid: Tensor, z0: Tensor) -> Tensor:
        n, D = z0.shape
        t0 = ts_grid[0:1, :].expand(n, -1)
        t0_cond = self._cond_t(t0)
        y0, _ = self.flow.inverse(z0, t0_cond)
        y = self.ou.sample_path_cond(ts_grid, y0)
        yf = y.reshape(-1, D)
        tf = ts_grid[None, :, :].expand(n, -1, -1).reshape(-1, 1)
        tf_cond = self._cond_t(tf)
        z = self.flow.forward(yf, tf_cond).reshape(n, ts_grid.size(0), D)
        return z


# -------------------------------
# z0 prior (in z-space)
# -------------------------------
class PriorInitDistribution(nn.Module):
    def __init__(self, latent_size: int, init_log_s: float = -0.2):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1, latent_size))
        self.log_s = nn.Parameter(torch.full((1, latent_size), init_log_s))
    def forward(self) -> D.Distribution:
        m = self.m
        s = torch.exp(self.log_s)
        return D.Independent(D.Normal(m, s), 1)


class FlowPriorMatchingCond(nn.Module):
    """
    E_q[ sum_t log p(x_t|z_t) + log p(z0) + log p(z_{1:T}|z0) - log q(z_{0:T}|x) ]
    """
    def __init__(
        self,
        prior_flow: NF_SDE_Model,
        p_observe_flow: ObservationFlow,
        q_enc: PosteriorEncoder,
        q_affine: PosteriorAffine,
        z0_prior: PriorInitDistribution,
    ):
        super().__init__()
        self.prior = prior_flow
        self.p_obs_flow = p_observe_flow
        self.q_enc = q_enc
        self.q_affine = q_affine
        self.z0_prior = z0_prior

    def _posterior_coeffs_all(self, ctx: Tensor, ts: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, _ = ts.shape
        t_flat = ts.reshape(B*T, 1)
        ctx_rep = ctx.repeat_interleave(T, dim=0)
        m_flat, s_flat = self.q_affine(ctx_rep, t_flat)
        Dz = m_flat.size(-1)
        m = m_flat.view(B, T, Dz)
        s = s_flat.view(B, T, Dz)
        return m, s

    def forward(self, xs: Tensor, ts: Tensor) -> Tuple[Tensor, dict]:
        B, T, Dx = xs.shape

        ctx = self.q_enc(xs)                      # (B,1+T,H)
        m, s = self._posterior_coeffs_all(ctx, ts)
        eps = torch.randn_like(m)
        z = m + s * eps                           # (B,T,Dz)
        Dz = z.size(-1)

        # log p(x|z) via conditional flow
        x_flat = xs.reshape(B*T, Dx)
        z_flat = z.reshape(B*T, Dz)
        log_px_flat = self.p_obs_flow.log_prob(x_flat, z_flat)  # (B*T,)
        log_px = log_px_flat.view(B, T).sum(dim=1)              # (B,)

        # log p(z0) + log p(z_{1:T}|z0)
        z0 = z[:, 0, :]
        log_pz0 = self.z0_prior().log_prob(z0)                  # (B,)
        log_pz_cond = self.prior.log_prob_paths_cond(z, ts, z0) # (B,)

        # -E_q[log q]
        log_q = diag_gauss_logprob(
            z.reshape(B*T, Dz), m.reshape(B*T, Dz), (s.reshape(B*T, Dz) ** 2)
        ).view(B, T).sum(dim=1)

        elbo = log_px + log_pz0 + log_pz_cond - log_q
        loss = -elbo.mean()

        return loss


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data on [0,1]
    batch_size = 2 ** 10
    ts0, tf = 0., 3.
    n_steps = 120
    noise_std = .01
    xs, ts = gen_data(batch_size, ts0, tf, n_steps, noise_std)
    xs, ts = xs.to(device), ts.to(device)
    visualise_data(xs[:6], 'data.jpg')

    # sizes
    data_size = xs.size(-1)     # 3
    latent_size = 4
    hidden_size = 128

    # modules
    prior_flow = NF_SDE_Model(D=latent_size, hidden=hidden_size, n_layers=6, tdim=64,
                              t_min=ts0, t_max=tf).to(device)
    z0_prior = PriorInitDistribution(latent_size, init_log_s=-0.2).to(device)
    p_obs_flow = ObservationFlow(data_size, cond_dim=latent_size, hidden=hidden_size, n_layers=6).to(device)
    q_enc = PosteriorEncoder(data_size, hidden_size).to(device)
    q_affine = PosteriorAffine(latent_size, hidden_size, init_logstd=-0.5).to(device)
    model = FlowPriorMatchingCond(prior_flow, p_obs_flow, q_enc, q_affine, z0_prior).to(device)

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training (exact ELBO)
    iters = 10000
    pbar = trange(iters)
    for i in pbar:
        optim.zero_grad(set_to_none=True)
        loss = model(xs, ts)
        loss.backward()
        optim.step()

        pbar.set_description(f"Loss: {loss.item():.6f}")
        if i % 100 == 0:
            with torch.no_grad():
                T_vis = 1000
                ts_vis = torch.linspace(0., 3., T_vis + 1, device=device).unsqueeze(1)  # (T,1)
                n_paths = 6
                z0_samples = z0_prior().rsample([n_paths])[:, 0, :]                     # (n,Dz)
                z_paths = prior_flow.sample_paths_cond(ts_vis, z0_samples)              # (n,T,Dz)

                Dx = data_size
                u0 = torch.zeros(n_paths*(T_vis+1), Dx, device=device)
                z_flat = z_paths.reshape(-1, latent_size)
                x_flat = p_obs_flow.flow.forward(u0, z_flat)
                x_paths = x_flat.reshape(n_paths, T_vis + 1, Dx)

                visualise_data(x_paths.detach().cpu(), 'samples.jpg')

