import torch


def kl_divergence_of_two_gaussians(m1, s1, m2, s2):
    """
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    return torch.log(s2 / s1) + (s1 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def compute_covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    num_of_random_variables = multiply_dims(x.shape[1:])
    num_of_samples = x.shape[0]
    matrix = x.reshape(num_of_samples, num_of_random_variables)

    mean_of_each_random_var = torch.mean(matrix, dim=0)
    matrix = matrix - mean_of_each_random_var
    covm = 1 / (num_of_samples - 1) * matrix.transpose(-1, -2) @ matrix
    return covm


def calculate_covscores(cov_matrix: torch.Tensor, convert_abs: bool = False) -> torch.Tensor:
    if convert_abs:
        cov_matrix = torch.abs(cov_matrix)
    cm_scores = torch.sum(cov_matrix, dim=0) - torch.diag(cov_matrix)
    return cm_scores


def multiply_dims(dims: torch.Size):
    m = 1
    for d in dims:
        m = m * d
    return m