"""
STAGE-3 MODELS: VAEs with a genuine count likelihood, plus the diagnostics that
prove they actually train as VAEs.

WHY v1's MODELS WERE VOID. v1 used an MSE reconstruction summed over 500 genes
against a beta = 1 KL. Relative to a correctly scaled likelihood that is an
EFFECTIVE beta of ~0.003: the model trained as a deterministic autoencoder.
Two consequences the audit confirmed: (a) the Locatello/Khemakhem framing did
not apply, because there was no meaningful posterior; and (b) the adversarial
strength adv_lambda sat below the gradient-noise floor and was INERT -- the
"feasibility frontier" swept a knob that did nothing.

v2 fixes:

  * NEGATIVE-BINOMIAL likelihood on RAW COUNTS with a per-gene learned inverse
    dispersion, and the observed library size as the size factor (scVI-style,
    Lopez et al. 2018). The decoder emits a composition via softmax, exactly
    matching how dgp2 generates.
  * DIAGNOSTICS returned from every fit: per-dimension KL in nats, active units
    (Burda et al. 2016, Var_x(E_q[z|x]) > 0.01), the rate-distortion point, and
    the effective beta. A run that sits at either degenerate corner -- posterior
    collapse (KL -> 0) or deterministic AE (KL >> 0 with no stochasticity) --
    is reported, not silently used.
  * PRIOR MODES. `iso` is the standard isotropic N(0, I), which is provably
    NON-identifiable (Khemakhem 2020, Locatello 2019). `cond` is the iVAE
    domain-conditional factorised prior p(z | u) = N(mu_u, diag(sigma_u^2))
    over the environment index u -- the theory-matching arm. With n_env =
    2*latent_dim + 1 the variability condition (Thm 1, assumption (iv)) is
    satisfiable; with n_env = 2 it provably is not, which is the deliberately
    violating arm.
  * adv_lambda liveness is asserted, not assumed (see `adversary_is_live`).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Model2Config:
    mode: str = "conditional"     # vanilla | conditional | adversarial | ivae
    prior: str = "iso"            # iso | cond   ('cond' == iVAE conditional prior)
    latent_dim: int = 2
    hidden: int = 128
    beta: float = 1.0             # KL weight on the TRUE NB ELBO
    adv_lambda: float = 1.0
    n_env: int = 5                # size of the auxiliary variable u
    epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 256
    device: str = "cpu"
    seed: int = 0


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


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


def nb_nll(x, mu, theta, eps=1e-8):
    """Negative binomial NLL, summed over genes. theta = inverse dispersion."""
    lg = (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0))
    ll = (lg + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
          + x * (torch.log(mu + eps) - torch.log(theta + mu + eps)))
    return -ll.sum(1)


class CountVAE(nn.Module):
    def __init__(self, n_genes: int, cfg: Model2Config):
        super().__init__()
        self.cfg = cfg
        z = cfg.latent_dim
        self.encoder = _mlp([n_genes, cfg.hidden, cfg.hidden])
        self.mu = nn.Linear(cfg.hidden, z)
        self.logvar = nn.Linear(cfg.hidden, z)

        dec_in = z + (1 if cfg.mode == "conditional" else 0)
        self.decoder = _mlp([dec_in, cfg.hidden, cfg.hidden, n_genes])
        self.log_theta = nn.Parameter(torch.zeros(n_genes))     # per-gene dispersion

        if cfg.prior == "cond":                                  # iVAE prior p(z|u)
            self.prior_net = nn.Linear(cfg.n_env, 2 * z)
        if cfg.mode == "adversarial":
            self.adversary = _mlp([z, cfg.hidden, 1])

    def encode(self, xn):
        h = self.encoder(xn)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def prior_params(self, u_onehot, z_dim, device):
        if self.cfg.prior == "cond":
            p = self.prior_net(u_onehot)
            return p[:, :z_dim], p[:, z_dim:].clamp(-8, 8)
        zero = torch.zeros(u_onehot.shape[0], z_dim, device=device)
        return zero, zero

    def elbo(self, x_raw, xn, s, u_onehot):
        mu, logvar = self.encode(xn)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        dec_in = torch.cat([z, s.view(-1, 1).float()], 1) if self.cfg.mode == "conditional" else z
        px_scale = F.softmax(self.decoder(dec_in), dim=1)
        lib = x_raw.sum(1, keepdim=True)                   # observed size factor
        px_rate = px_scale * lib
        recon = nb_nll(x_raw, px_rate, torch.exp(self.log_theta).clamp(1e-3, 1e4))

        # KL( q(z|x) || p(z|u) ) for a diagonal Gaussian prior
        pm, plv = self.prior_params(u_onehot, mu.shape[1], mu.device)
        kl_d = 0.5 * (plv - logvar + (logvar.exp() + (mu - pm) ** 2) / plv.exp() - 1.0)
        kl = kl_d.sum(1)

        loss = (recon + self.cfg.beta * kl).mean()
        if self.cfg.mode == "adversarial":
            logits = self.adversary(grad_reverse(mu, self.cfg.adv_lambda)).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(logits, s.float())
        if self.cfg.mode == "mixing":
            # POSITIVE CONTROL: a moment-matching integration objective that
            # forces the two domains' shared-latent distributions to coincide
            # (mean + covariance), which is what MNN/Harmony-style integration
            # does. At full overlap the domains already share a t-distribution,
            # so matching is free; below full overlap, matching REQUIRES mapping
            # A's high-t cells onto B's low-t cells -- misaligning biology. This
            # is the objective that should make the MCC gap grow as rho falls,
            # and it is the control that makes the faithful-encoder null
            # interpretable (the metric CAN see the effect when it is present).
            m0 = mu[s == 0]; m1 = mu[s == 1]
            if len(m0) > 2 and len(m1) > 2:
                mean_gap = ((m0.mean(0) - m1.mean(0)) ** 2).sum()
                c0 = torch.cov(m0.T); c1 = torch.cov(m1.T)
                cov_gap = ((c0 - c1) ** 2).sum()
                loss = loss + self.cfg.adv_lambda * (mean_gap + cov_gap)
        return loss, recon.mean().detach(), kl_d.mean(0).detach()

    @torch.no_grad()
    def embed(self, xn):
        return self.encode(xn)[0].cpu().numpy()


def prep(X: np.ndarray) -> np.ndarray:
    """log1p median-scaled input for the ENCODER only. The likelihood is
    evaluated on raw counts, so this is a feature transform, not a rescaling of
    the objective (which is what produced v1's effective beta of 0.003)."""
    lib = X.sum(1, keepdims=True)
    med = np.median(lib[lib > 0]) if (lib > 0).any() else 1.0
    return np.log1p(X / (lib + 1e-8) * med).astype(np.float32)


def train_model2(X, s, e, cfg: Model2Config):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)
    Xn = prep(X)
    xr = torch.tensor(np.asarray(X, np.float32), device=dev)
    xn = torch.tensor(Xn, device=dev)
    st = torch.tensor(np.asarray(s), device=dev)
    ue = torch.tensor(np.asarray(e), device=dev, dtype=torch.long)
    uh = F.one_hot(ue, num_classes=cfg.n_env).float()

    model = CountVAE(Xn.shape[1], cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = xr.shape[0]
    hist = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, device=dev)
        agg = []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss, rec, kld = model.elbo(xr[idx], xn[idx], st[idx], uh[idx])
            loss.backward()
            opt.step()
            agg.append((float(rec), float(kld.sum())))
        if ep % 20 == 0 or ep == cfg.epochs - 1:
            a = np.array(agg).mean(0)
            hist.append(dict(epoch=ep, recon=a[0], kl=a[1]))
    model.eval()
    return model, xn, hist


# --------------------------------------------------------------------------- #
#  VAE-behaviour diagnostics  (STAGE-3 GATE)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def vae_diagnostics(model, xn, xr, s, uh) -> dict:
    """Rate-distortion point, per-dimension KL, and active units.

    GATE: the run must sit away from BOTH degenerate corners.
      posterior collapse  -> total KL ~ 0 and 0 active units
      deterministic AE    -> posterior variance ~ 0 (mean |logvar| very negative)
    """
    mu, logvar = model.encode(xn)
    pm, plv = model.prior_params(uh, mu.shape[1], mu.device)
    kl_d = 0.5 * (plv - logvar + (logvar.exp() + (mu - pm) ** 2) / plv.exp() - 1.0)
    kl_per_dim = kl_d.mean(0).cpu().numpy()

    au = mu.var(0).cpu().numpy()                       # Burda active-units statistic
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    dec_in = (torch.cat([z, s.view(-1, 1).float()], 1)
              if model.cfg.mode == "conditional" else z)
    px = F.softmax(model.decoder(dec_in), 1) * xr.sum(1, keepdim=True)
    recon = float(nb_nll(xr, px, torch.exp(model.log_theta).clamp(1e-3, 1e4)).mean())

    # Stochasticity ratio: posterior sd relative to the latent's OWN marginal
    # spread. An absolute threshold on logvar is the wrong test -- at a healthy
    # rate of ~3 nats/dim the posterior is legitimately ~20x narrower than the
    # unit prior, which an absolute rule misreads as a deterministic AE. What
    # actually makes a model deterministic is sampling noise that is negligible
    # against the signal it encodes.
    post_sd = torch.exp(0.5 * logvar).mean(0).cpu().numpy()
    stoch = post_sd / (np.sqrt(au) + 1e-12)

    return dict(
        kl_per_dim=[float(x) for x in kl_per_dim],
        kl_total=float(kl_per_dim.sum()),
        kl_per_dim_min=float(kl_per_dim.min()),
        active_units=int((au > 0.01).sum()),
        active_unit_vars=[float(x) for x in au],
        recon_nats=recon,
        rate_over_distortion=float(kl_per_dim.sum() / max(recon, 1e-9)),
        mean_post_logvar=float(logvar.mean()),
        stochasticity_ratio=[float(x) for x in stoch],
        effective_beta=float(model.cfg.beta),
        # the two degenerate corners
        posterior_collapse=bool(kl_per_dim.min() < 0.01),
        deterministic_ae=bool(np.max(stoch) < 0.01),
    )


def _domain_predictability(z, s):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import balanced_accuracy_score
    pred = cross_val_predict(HistGradientBoostingClassifier(max_depth=4, random_state=0),
                             z, s, cv=KFold(3, shuffle=True, random_state=0))
    return float(balanced_accuracy_score(s, pred))


def conditioning_is_live(X, s, e, cfg: Model2Config) -> dict:
    """Assert the DOMAIN-CONDITIONING knob v2 actually uses is live.

    v2's arms remove the batch effect via a decoder covariate (conditional /
    scVI-style) or a conditional prior (iVAE), NOT via the adversary. So the
    load-bearing liveness check is: at rho=1/delta=large (domain predictable
    ONLY through the removable batch effect), does conditioning make the shared
    latent LESS domain-predictable than a vanilla VAE that has no way to
    explain-away the batch? A vanilla iso-prior VAE must encode the offset to
    reconstruct it; the conditional decoder absorbs it, so its latent should be
    markedly less domain-predictable. That gap is the mechanism v2 relies on.
    """
    out = {}
    for name, mode, prior in (("vanilla", "vanilla", "iso"),
                              ("conditional", "conditional", "iso"),
                              ("ivae", "ivae", "cond")):
        c = Model2Config(**{**cfg.__dict__, "mode": mode, "prior": prior})
        m, xn, _ = train_model2(X, s, e, c)
        out[name] = _domain_predictability(m.embed(xn), s)
    # conditioning must reduce domain-predictability vs the vanilla baseline
    out["cond_reduces"] = bool(out["vanilla"] - out["conditional"] > 0.05)
    out["ivae_reduces"] = bool(out["vanilla"] - out["ivae"] > 0.05)
    out["live"] = bool(out["cond_reduces"] or out["ivae_reduces"])
    return out


def adversary_is_live(X, s, e, cfg: Model2Config,
                      lambdas=(0.0, 1.0, 10.0, 100.0)) -> dict:
    """DEPRECATED for v2 (kept for the record). The adversarial variant is NOT
    part of v2's main sweep. In the 2-D-latent count-VAE regime it is
    structurally weak: reconstruction requires encoding the removable offset and
    there is no spare latent capacity, so the reconstruction gradient (~1000+
    nats) swamps the adversary's BCE (<= ln 2). This is a DIFFERENT failure from
    v1's inert adv_lambda (v1's effective beta was 0.003, an autoencoder); here
    the VAE is healthy and the adversary is simply outgunned. Use
    conditioning_is_live instead -- it tests the knob v2 actually uses."""
    out = {}
    for lam in lambdas:
        c = Model2Config(**{**cfg.__dict__, "mode": "adversarial", "adv_lambda": lam})
        m, xn, _ = train_model2(X, s, e, c)
        out[lam] = _domain_predictability(m.embed(xn), s)
    vals = list(out.values())
    out["live"] = bool(vals[0] - vals[-1] > 0.05)
    return out
