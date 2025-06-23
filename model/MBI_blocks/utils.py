import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
import math
import numpy as np
import torch.nn.functional as F


def get_mask(attn: torch.Tensor, ratio: float = 0.1):
    """
    根据 attn 数值，将最小的 `ratio` 部分的值 mask 掉。

    参数:
        attn (Tensor): [N, 1] 形状的张量
        ratio (float): 需要 mask 的比例，例如 0.1 代表最小的 10%

    返回:
        mask (Tensor): [N, 1] 形状的布尔张量，True 表示保留，False 表示被 mask
    """
    threshold = torch.quantile(attn, ratio)  # 计算前 ratio% 的分位数
    mask = attn > threshold  # 只保留大于阈值的部分

    return mask


class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
                np.pi
                / r_max
                * torch.linspace(
            start=1.0,
            end=num_basis,
            steps=num_basis,
            dtype=torch.get_default_dtype(),
        )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class PolynomialCutoff(torch.nn.Module):
    """
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


class BesselEmbeddingBlock(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            num_bessel: int,
    ):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)

        self.cutoff_fn = CosineCutoff(r_max)
        self.out_dim = num_bessel

    def forward(
            self,
            edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]

        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]

        return radial * cutoff  # [n_edges, n_basis]


class CosineCutoff(nn.Module):

    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class VecNorm(nn.Module):
    def __init__(self, hidden_channels, norm_type="norm"):
        super(VecNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12

        self.dist_norm = nn.LayerNorm(hidden_channels)

        if norm_type == "norm":
            self.norm = self.vec_norm
        else:
            self.norm = self.none_norm

        self.reset_parameters()

    def reset_parameters(self):
        self.dist_norm.reset_parameters()

    @torch.jit.export
    def none_norm(self, vec):
        return vec

    @torch.jit.export
    def vec_norm(self, vec):

        eps = self.eps

        dist = torch.norm(vec, dim=1, keepdim=True)  # (num_atoms, 1, hidden_channels)
        if torch.all(dist == 0):
            return torch.zeros_like(vec)

        # 方向向量
        radial = vec / dist.clamp(min=eps)

        # 每个样本的 hidden_channel 维度上做 min-max 归一化
        min_val = torch.min(dist, dim=-1, keepdim=True)[0]
        max_val = torch.max(dist, dim=-1, keepdim=True)[0]
        dist_norm = (dist - min_val) / (max_val - min_val).clamp(min=eps)

        norm_vec = dist_norm * radial

        return norm_vec

    def forward(self, vec):

        vec = self.norm(vec)

        return vec


class Edge_Connect(nn.Module):
    def __init__(self, cutoff, max_neighbors=32, loop=True):
        """
        Distance module for computing pairwise distances in molecular graphs.

        Args:
            cutoff (float): Maximum distance threshold for edge connections.
            max_neighbors (int, optional): Maximum number of neighbors per node. Default is 10.
            include_self (bool, optional): Whether to include self-loops. Default is True.
        """
        super(Edge_Connect, self).__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.loop = loop

    def forward(self, positions, edge_indices, shift):
        """
        Computes edge index, edge distances, and edge vectors.

        Args:
            positions (torch.Tensor): Atomic positions of shape (num_atoms, 3).
            batch (torch.Tensor): Batch indices of shape (num_atoms,).
            edge_indices (torch.Tensor, optional): Precomputed edge indices of shape (2, num_edges).

        Returns:
            edge_indices (torch.Tensor): Edge indices of shape (2, num_edges).
            edge_distances (torch.Tensor): Computed Euclidean distances for edges.
            edge_vectors (torch.Tensor): Edge vectors between connected atoms.
        """
        # Compute edge vectors
        row, col = edge_indices[0], edge_indices[1]

        if shift is not None:
            edge_vectors = positions[row] - positions[col] - shift
        else:
            edge_vectors = positions[row] - positions[col]

        mask = (row != col).unsqueeze(-1).to(positions.dtype)  # [E,1]
        norms = torch.norm(edge_vectors, dim=-1, keepdim=True) * mask
        edge_vectors = edge_vectors / torch.where(mask == 1, norms, 1.0)
        edge_distances = norms.squeeze(-1)

        return edge_indices, edge_distances, edge_vectors


def radius_graph_range(positions, min_radius, max_radius, loop, batch, max_neighbors=100):
    """
    Generates edge indices for pairs of atoms whose distances fall within [min_radius, max_radius).

    Args:
        positions (torch.Tensor): Atomic coordinates of shape (num_atoms, 3).
        min_radius (float): Minimum distance threshold.
        max_radius (float): Maximum distance threshold.
        loop (bool): Whether to include self-loops.
        batch (torch.Tensor): Batch indices of shape (num_atoms,).
        max_neighbors (int, optional): Maximum number of neighbors per atom.

    Returns:
        torch.Tensor: Edge indices of shape (2, num_edges) within the specified distance range.
    """
    # Compute edges within max_radius
    edge_indices_max = radius_graph(positions, r=max_radius, loop=loop,
                                    batch=batch, max_num_neighbors=max_neighbors)

    # Compute edges within min_radius
    edge_indices_min = radius_graph(positions, r=min_radius, loop=False,
                                    batch=batch, max_num_neighbors=max_neighbors)

    # Convert edge indices to sets for filtering
    edge_set_max = set(map(tuple, edge_indices_max.t().tolist()))
    edge_set_min = set(map(tuple, edge_indices_min.t().tolist()))

    # Compute the difference set to get edges in range [min_radius, max_radius)
    edge_set_filtered = edge_set_max - edge_set_min

    # Convert back to tensor
    edge_indices_filtered = torch.tensor(list(edge_set_filtered), dtype=torch.long).t()

    return edge_indices_filtered


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
