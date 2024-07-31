import torch as ch

import math

def kl_loss(pred_mean: ch.Tensor, pred_cov: ch.Tensor, target_mean: ch.Tensor, target_cov: ch.Tensor) -> ch.Tensor:
    """
    Compute the KL divergence between two gaussian distributions in batch
    """
    inv_target_cov = ch.inverse(target_cov)
    det_pred_cov = ch.det(pred_cov)
    det_target_cov = ch.det(target_cov)

    det_term = ch.log(det_target_cov) - ch.log(det_pred_cov)
    trace_term = ch.einsum('bii->bi', (inv_target_cov @ pred_cov)).sum(dim=-1)
    mean_term = (target_mean - pred_mean).unsqueeze(-1).swapaxes(-1, -2) @ inv_target_cov @ (target_mean - pred_mean).unsqueeze(-1)
    k = target_cov.shape[-1]

    batch_kl = 0.5 * (det_term + trace_term + mean_term.squeeze() - k)

    if ch.isnan(batch_kl).any():
        print('det_term', det_term)
        print('trace_term', trace_term)
        print('mean_term', mean_term)
        print('k', k)
        exit()

    return batch_kl.mean()

def kl_loss_chol(mean_0: ch.Tensor, chol_0: ch.Tensor, mean_1: ch.Tensor, chol_1: ch.Tensor) -> ch.Tensor:
    """
    Compute the KL divergence between two gaussian distributions in batch
    """
    dim = chol_0.shape[-1]

    # compute the determinant terms using the cholesky decomposition
    # det_chol_0 = ch.det(chol_0) ** 2
    det_chol_0 = 2 * chol_0.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    # det_chol_1 = ch.det(chol_1) ** 2
    det_chol_1 = 2 * chol_1.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    # det_term = ch.log(det_chol_1) - ch.log(det_chol_0)
    det_term = det_chol_1 - det_chol_0

    # compute the trace term
    inv_L1 = ch.inverse(chol_1)
    inv_Sigma_1 = inv_L1.transpose(-1, -2) @ inv_L1
    Sigma_0 = chol_0 @ chol_0.transpose(-1, -2)
    trace_term = ch.einsum('bii->bi', (inv_Sigma_1 @ Sigma_0)).sum(dim=-1)

    # Compute the mean term
    diff = (mean_1 - mean_0).unsqueeze(-1)
    quadratic_term = diff.transpose(-1, -2) @ inv_Sigma_1 @ diff
    quadratic_term = quadratic_term.squeeze(-1, -2)

    batch_kl = 0.5 * (det_term + trace_term + quadratic_term - dim)

    if ch.isnan(batch_kl).any():
        print('det_term', det_term)
        print('trace_term', trace_term)
        print('mean_term', quadratic_term)
        print('k', dim)

    return batch_kl.mean()


def gaussian_entropy_chol(cholesky_matrices):
    b, n, _ = cholesky_matrices.shape  # b is batch size, n is dimensionality
    # Determinant of L is the product of its diagonal elements
    det_L = ch.prod(ch.diagonal(cholesky_matrices, dim1=-2, dim2=-1), dim=-1)
    # Determinant of the covariance matrix is square of the determinant of L
    det_covariances = det_L ** 2
    entropy = 0.5 * ch.log((2 * ch.pi * ch.e) ** n * det_covariances)
    return entropy


def gaussian_entropy_cov(covariance_matrices):
    b, n, _ = covariance_matrices.shape  # b is batch size, n is dimensionality
    det_covariances = ch.det(covariance_matrices)  # Determinants of the covariance matrices in the batch
    entropy = 0.5 * ch.log((2 * ch.pi * ch.e) ** n * det_covariances)
    return entropy
