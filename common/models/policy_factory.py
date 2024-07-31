import torch as ch

from common.models.gaussian_policy_diag import GaussianPolicyDiag
from common.models.gaussian_policy_diag_squashed import GaussianPolicyDiagSquashed
from common.models.gaussian_policy_full import GaussianPolicyFull
from common.models.gaussian_policy_full_squashed import GaussianPolicyFullSquashed
from common.models.gaussian_policy_sqrt import GaussianPolicySqrt
from common.models.gaussian_policy_sqrt_squashed import GaussianPolicySqrtSquashed
from common.models.gaussian_policy_full_indep_chol import GaussianPolicyFullIndepChol
from common.models.gaussian_policy_full_filmed import FilmedGaussianPolicyFull
from common.models.gaussian_policy_diag_filmed import FilmedGaussianPolicyDiag
from common.models.gaussian_policy_full_gmm_filmed import FilmedGMMPolicyFull
from common.models.gaussian_policy_diag_gmm_filmed import FilmedGMMPolicyDiag
from common.models.gaussian_policy_full_gmm_transformer import TransformerGMMPolicyFull
from common.models.gaussian_policy_diag_gmm_transformer import TransformerGMMPolicyDiag


def get_policy_network(cov_type, proj_type, squash=False, device: ch.device = "cpu", dtype=ch.float32, **kwargs):
    """
    Policy network factory
    Args:
        cov_type: 'full' or 'diag' covariance
        proj_type: Which projection is used.
        squash: Gaussian policy with tanh transformation
        device: torch device
        dtype: torch dtype
        **kwargs: policy arguments

    Returns:
        Gaussian Policy instance
    """
    if isinstance(dtype, str):
        dtype = ch.float32 if dtype == "float32" else ch.float64

    if squash:
        if cov_type == "full":
            policy = GaussianPolicySqrtSquashed(**kwargs) if "w2" in proj_type else GaussianPolicyFullSquashed(**kwargs)
        elif cov_type == "diag":
            policy = GaussianPolicyDiagSquashed(**kwargs)
        else:
            raise ValueError(f"Invalid policy type {cov_type}. Select one of 'full', 'diag'.")
    else:

        if cov_type == "full":
            policy = GaussianPolicySqrt(**kwargs) if "w2" in proj_type else GaussianPolicyFull(**kwargs)
        elif cov_type == "indep_full":
            policy = GaussianPolicyFullIndepChol(**kwargs)
        elif cov_type == "film_full":
            policy = FilmedGaussianPolicyFull(**kwargs)
        elif cov_type == "gmm_full":
            policy = FilmedGMMPolicyFull(**kwargs)
        elif cov_type == "gmm_diag":
            policy = FilmedGMMPolicyDiag(**kwargs)
        elif cov_type == "film_diag":
            policy = FilmedGaussianPolicyDiag(**kwargs)
        elif cov_type == "diag":
            policy = GaussianPolicyDiag(**kwargs)
        elif cov_type == "transformer_gmm_full":
            policy = TransformerGMMPolicyFull(**kwargs)
        elif cov_type == "transformer_gmm_diag":
            policy = TransformerGMMPolicyDiag(**kwargs)
        else:
            raise ValueError(f"Invalid policy type {cov_type}")

    return policy.to(device, dtype)
