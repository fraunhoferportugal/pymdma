from typing import Optional, Sequence

import torch
from torch import nn


# ======= Multikernel with Gaussian kernels =======
class GaussianKernel(nn.Module):
    """Gaussian Kernel Module for Maximum Mean Discrepancy (MMD) computation.

    Parameters
    ----------
    sigma : float, optional
        Bandwidth \\( \\sigma \\). Default is None.
    track_running_stats : bool, optional
        If ``True``, this module tracks the running mean of \\( \\sigma^2 \\).
        Otherwise, it won't track such statistics and always uses fixed \\( \\sigma^2 \\). Default is True.
    alpha : float, optional
        \\( \alpha \\) which decides the magnitude of \\( \\sigma^2 \\) when `track_running_stats` is set to True.

    References
    ---------
    Code adapted from:
    https://github.com/lhoyer/MIC/blob/master/cls/dalib/modules/kernels.py
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        track_running_stats: Optional[bool] = True,
        alpha: Optional[float] = 1.0,
    ):
        super().__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, x_feat: torch.Tensor) -> torch.Tensor:
        """Gaussian Kernel Matrix.

        Parameters
        ----------
        x_feat : torch.Tensor
            Tensor of features

        Returns
        -------
        torch.Tensor
            Gaussian kernel Matrix
        """
        l2_distance_square = ((x_feat.unsqueeze(0) - x_feat.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """Multiple Kernel Maximum Mean Discrepancy (MK-MMD).

    Parameters
    ----------
    kernels : Sequence[nn.Module]
        A sequence of kernel functions to be used in the MK-MMD computation.
    linear : bool, optional
        If True, use a linear version of the MK-MMD loss. Otherwise, use a nonlinear version. Default is False.

    References
    ---------
    Long M. et al., Learning Transferable Features with Deep Adaptation Networks (ICML 2015)
    https://arxiv.org/pdf/1502.02791

    Code adapted from:
    Transfer Learning Library by the THUML Group.
    https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dan.py
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super().__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def _update_index_matrix(
        self,
        batch_size: int,
        index_matrix: Optional[torch.Tensor] = None,
        linear: Optional[bool] = True,
    ) -> torch.Tensor:
        """Update the `index_matrix` which converts `kernel_matrix` to loss.

        Parameters
        ----------
        batch_size: int
            The batch size of the input data.
        index_matrix: torch.Tensor, optional
            A precomputed index matrix. If provided and its shape is (2 x batch_size, 2 x batch_size),
            it will be used as is. Otherwise, a new index matrix will be created.
        linear: bool, optional
            If True, a linear index matrix will be used. Otherwise, use a nonlinear index matrix will be used.
            Defaults to True.

        Returns
        -------
        torch.Tensor
            The updated index matrix with shape (2 x batch_size, 2 x batch_size).
        """

        if index_matrix is None or index_matrix.size(0) != batch_size * 2:
            index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
            if linear:
                for i in range(batch_size):
                    s1, s2 = i, (i + 1) % batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    index_matrix[s1, s2] = 1.0 / float(batch_size)
                    index_matrix[t1, t2] = 1.0 / float(batch_size)
                    index_matrix[s1, t2] = -1.0 / float(batch_size)
                    index_matrix[s2, t1] = -1.0 / float(batch_size)
            else:
                for i in range(batch_size):
                    for ii in range(batch_size):
                        if i != ii:
                            index_matrix[i][ii] = 1.0 / float(batch_size * (batch_size - 1))
                            index_matrix[i + batch_size][ii + batch_size] = 1.0 / float(batch_size * (batch_size - 1))
                for i in range(batch_size):
                    for ii in range(batch_size):
                        index_matrix[i][ii + batch_size] = -1.0 / float(batch_size * batch_size)
                        index_matrix[i + batch_size][ii] = -1.0 / float(batch_size * batch_size)
        return index_matrix

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Compute the MK-MMD between features between features from a source
        distribution (z_s) and a target distribution (z_t). This loss measures
        the discrepancy between the two distributions using multiple kernel
        functions.

        Parameters
        ----------
        z_s : torch.Tensor
            Source features with shape (batch_size, feature_dim).
        z_t : torch.Tensor
            Target features with shape (batch_size, feature_dim).

        Returns
        -------
        torch.Tensor
            The MK-MMD value.
        """
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = self._update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2.0 / float(batch_size - 1)

        return loss
