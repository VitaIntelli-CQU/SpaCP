from __future__ import annotations

import torch

__all__ = ["gaussian_prior", "all_zeros", "PriorSampler", "Interpolant"]

def gaussian_prior(shape):
    return torch.randn(shape)

def all_zeros(shape):
    return torch.zeros(shape)

class Interpolant:
    def __init__(self, prior_sample_type, normalize=True, **kwargs):
        if (torch.cuda.is_available()):
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.prior_sampler = PriorSampler(prior_sample_type, device=self.device, **kwargs)
        self.normalize = normalize
    
    def sample_from_prior(self, shape):
        exp = self.prior_sampler.sample(shape).to(self.device)
        if self.normalize:
            exp = torch.log(exp + 1)
        return exp

    def sample_t(self, shape):
        return torch.rand(shape)

    def corrupt_exp(self, exp):
        # exp: [B, n_cells, n_genes]
        t = self.sample_t((exp.shape[0],)).to(self.device)
        if exp.shape[0] > 1:
            t = t.squeeze(-1)
        exp_0 = self.sample_from_prior(exp.shape).to(self.device)
        return exp, exp_0 * (1 - t[:, None, None]) + exp * t[:, None, None], t

    def denoise(self, exp_1, exp_t, t, d_t):
        # exp_1: [B, n_cells, n_genes]
        # exp_t: [B, n_cells, n_genes]
        # t: [B]
        # d_t: [B]
        exp_vf = (exp_1 - exp_t) / (1 - t[:, None, None])
        return exp_t + d_t[:, None, None] * exp_vf

class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            # https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L433
            from scvi.distributions import ZeroInflatedNegativeBinomial

            prior_sampler = ZeroInflatedNegativeBinomial(
                                        total_count=kwargs.get("total_count", None),
                                        logits=kwargs.get("logits", None),
                                        zi_logits=kwargs.get("zi_logits", None),  # real number
                                    )
            self.prior_sampler = lambda shape: prior_sampler.sample(shape).squeeze(-1)
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        return self.prior_sampler(shape)
