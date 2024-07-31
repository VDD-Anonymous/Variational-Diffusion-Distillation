from common.models.residual.residual_gaussian_diag import ResidualGaussianPolicyDiag


def get_residual_policy_network(pretrained_policy, **kwargs):
    """
    Policy network factory
    Args:
        pretrained_policy: pretrained policy
        **kwargs: policy arguments

    Returns:
        Gaussian Policy instance
    """
    if kwargs['cov_type'] != 'diag':
        raise NotImplementedError("Only diagonal covariance is supported for residual policies")

    return ResidualGaussianPolicyDiag(pretrained_policy=pretrained_policy, **kwargs)