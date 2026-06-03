import torch
import torch.nn.functional as F
import math

class VMFLikelihood(torch.nn.Module):
    def __init__(self, d=256, eps=1e-7):
        super().__init__()
        self.d = d
        self.eps = eps

    def log_partition_function(self, kappa):
        """
        Approximates log(Z_d(kappa)) in a fully differentiable, numerically stable way.
        Uses the integral of the Amos (1974) stable ratio approx for A_d(kappa):
        A_d(kappa) = I_{d/2}(kappa)/I_{d/2-1}(kappa) ≈ kappa / (v + sqrt(kappa^2 + v^2))
        where v = (d-1)/2. Integrating this yields the log-partition function mathematically.
        """
        v = (self.d - 1) / 2.0
        
        # Use simple eps to prevent zero gradient exactly at kappa=0
        kappa_eps = kappa + self.eps
        y = torch.sqrt(kappa_eps**2 + v**2)
        
        # Integral of Amos approximation formulation
        integral = y - v * torch.log(v + y)
        
        # Integration constant c = log Z_d(0) - integral(0)
        # Z_d(0) is the surface area of the d-dimensional unit hypersphere.
        log_z0 = math.log(2.0) + (self.d / 2.0) * math.log(math.pi) - math.lgamma(self.d / 2.0)
        integral_0 = v - v * math.log(2.0 * v)
        c = log_z0 - integral_0
        
        return integral + c

    def forward(self, mu, kappa, target):
        """
        mu: [batch, d] - Predicted mean (must be L2 normalized)
        kappa: [batch, 1] - Predicted concentration (from Softplus)
        target: [batch, d] - Ground truth vector (must be L2 normalized)
        """
        # 1. Ensure inputs are normalized (defensive)
        mu = F.normalize(mu, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        
        # 2. Dot product (Cosine Similarity)
        dot_prod = torch.sum(mu * target, dim=-1, keepdim=True)
        
        # 3. vMF Log-Likelihood
        # NLL = -kappa * cos_sim + log_Z_d(kappa)
        log_z_d = self.log_partition_function(kappa)
        
        loss = -kappa * dot_prod + log_z_d
        
        return loss.mean()