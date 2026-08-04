"""
Recovery models = the four disentanglement / batch-correction strategies whose
feasibility frontier we sweep. Each maps to a target lab's question:

  vanilla      : plain VAE, no domain info. CONTROL. It tends to recover `t` at
                 all overlaps because it never tries to remove the domain -- it
                 simply conflates biology and domain. Its "success" is illusory
                 and exists to show the collapse is specific to the correction
                 operation, not to whether a VAE *can represent* t.
  conditional  : scVI-style, domain as a decoder covariate (explain-away).      -> Ding
  contrastive  : contrastiveVI-style shared + salient latents.                  -> NullState / Ding
  adversarial  : DANN-style domain-invariant encoder via gradient reversal.     -> the adversary as a training signal

Extensions (stubs below): mechanism-sparsity penalty (-> Sridhar/Lachapelle),
prior-informed constraint (-> Ding), anchor/paired supervision (-> Sridhar).

IMPORTANT identifiability note: a vanilla VAE with an isotropic prior is provably
NON-identifiable (Khemakhem 2020; Locatello 2019). So we do NOT claim "the VAE
recovers t"; we measure the FEASIBILITY FRONTIER (metrics.py): can ANY model
jointly (a) recover biology and (b) remove the domain? Below a critical overlap,
no model can do both -- that joint infeasibility *is* the identifiability limit,
and it is what sidesteps the vanilla-VAE non-identifiability confound.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    mode: str = "conditional"       # vanilla | conditional | contrastive | adversarial
    shared_dim: int = 2             # dim of the shared (biological) latent
    salient_dim: int = 2            # dim of the salient latent (contrastive only)
    hidden: int = 128
    beta: float = 1.0               # KL weight
    adv_lambda: float = 1.0         # gradient-reversal strength (adversarial only)
    salient_kl: float = 1.0         # extra KL pull on salient for reference cells (contrastive)
    # --- extension hooks (default off) ---
    mechanism_sparsity: float = 0.0  # L1 on decoder domain-shift weights (Sridhar/Lachapelle)
    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 256
    device: str = "cpu"
    seed: int = 0


# --------------------------------------------------------------------------- #
#  gradient reversal (for the adversarial / DANN variant)
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def _mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
#  the VAE
# --------------------------------------------------------------------------- #
class VAE(nn.Module):
    def __init__(self, n_genes: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        z = cfg.shared_dim + (cfg.salient_dim if cfg.mode == "contrastive" else 0)
        self.z_dim = z

        self.encoder = _mlp([n_genes, cfg.hidden, cfg.hidden])
        self.mu = nn.Linear(cfg.hidden, z)
        self.logvar = nn.Linear(cfg.hidden, z)

        dec_in = z + (1 if cfg.mode == "conditional" else 0)   # +domain covariate
        self.decoder = _mlp([dec_in, cfg.hidden, cfg.hidden, n_genes])

        if cfg.mode == "adversarial":
            self.adversary = _mlp([cfg.shared_dim, cfg.hidden, 1])

    # -- pieces ------------------------------------------------------------- #
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, s=None):
        if self.cfg.mode == "conditional":
            z = torch.cat([z, s.view(-1, 1).float()], dim=1)
        return self.decoder(z)

    # -- loss --------------------------------------------------------------- #
    def elbo(self, x, s):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)

        # contrastive: zero the salient block for reference (background) cells
        if self.cfg.mode == "contrastive":
            sd, td = self.cfg.shared_dim, self.cfg.salient_dim
            mask = s.view(-1, 1).float()                     # 1 for target/in-vitro
            z = torch.cat([z[:, :sd], z[:, sd:sd + td] * mask], dim=1)

        xhat = self.decode(z, s)
        recon = F.mse_loss(xhat, x, reduction="none").sum(1)   # Gaussian NLL up to const

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())    # per-dim KL
        if self.cfg.mode == "contrastive":
            sd = self.cfg.shared_dim
            # extra pull on salient KL for reference cells (salient should be ~0 there)
            w = torch.ones_like(kl)
            w[:, sd:] = 1.0 + self.cfg.salient_kl * (1.0 - s.view(-1, 1).float())
            kl = (kl * w)
        kl = kl.sum(1)

        loss = (recon + self.cfg.beta * kl).mean()

        # adversarial: encoder tries to make the SHARED latent domain-invariant
        if self.cfg.mode == "adversarial":
            shared = mu[:, :self.cfg.shared_dim]
            logits = self.adversary(grad_reverse(shared, self.cfg.adv_lambda)).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(logits, s.float())

        # mechanism-sparsity extension: L1 on the first decoder layer's weights
        # acting on the domain covariate column (only meaningful for conditional).
        if self.cfg.mechanism_sparsity > 0 and self.cfg.mode == "conditional":
            first = self.decoder[0].weight            # [hidden, dec_in]
            loss = loss + self.cfg.mechanism_sparsity * first[:, -1].abs().sum()

        return loss

    # -- inference: return the SHARED (biological) latent -------------------- #
    @torch.no_grad()
    def embed(self, x):
        mu, _ = self.encode(x)
        return mu[:, :self.cfg.shared_dim].cpu().numpy()


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
def _prep(X: np.ndarray) -> np.ndarray:
    """log1p + per-cell median scaling (standard lognorm)."""
    lib = X.sum(1, keepdims=True)
    med = np.median(lib[lib > 0]) if (lib > 0).any() else 1.0
    return np.log1p(X / (lib + 1e-8) * med).astype(np.float32)


def train_model(X: np.ndarray, s: np.ndarray, cfg: ModelConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    Xn = _prep(X)
    dev = torch.device(cfg.device)
    model = VAE(Xn.shape[1], cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    Xt = torch.tensor(Xn, device=dev)
    st = torch.tensor(s, device=dev)
    n = Xt.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = model.elbo(Xt[idx], st[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model, Xn


# --------------------------------------------------------------------------- #
#  EXTENSION STUBS  (fill in for the method-comparison arm)
# --------------------------------------------------------------------------- #
def prior_informed_hook():
    """Ding: inject partial true biology as a constraint (e.g. supervise a few
    latent dims with known markers) and re-measure the frontier. TODO."""
    raise NotImplementedError


def anchor_informed_hook():
    """Sridhar: add K cross-domain matched pairs and a pairing loss; test whether
    paired observations restore identifiability below threshold. TODO."""
    raise NotImplementedError


def scvi_spotcheck_hook():
    """Credibility spot-check at a few rho values with real scvi-tools SCVI/SCANVI
    and contrastiveVI. Verify the current API against docs before using:
        import scvi; scvi.model.SCVI.setup_anndata(adata, batch_key="s"); ...
    Keep this OUT of the dense sweep (slow); the custom VAE above is the workhorse.
    TODO."""
    raise NotImplementedError
