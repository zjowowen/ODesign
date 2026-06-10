# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.model.rmsd import weighted_rigid_align
from src.model.modules.frames import (
    express_coordinates_in_frame,
    gather_frame_atom_by_indices,
)
from src.utils.model.misc import expand_at_dim
from src.utils.openfold_local.utils.checkpointing import get_checkpoint_fn
from src.utils.model.torch_utils import cdist

from src.utils.data.constants import BOND_TYPE
from src.api.model_interface import (
    LossInput,
    GroundTruth,
    ODesignOutput,
)

def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """reduction wrapper

    Args:
        loss (torch.Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


def _valid_resolution_mask(
    resolution: torch.Tensor, min_resolution: float, max_resolution: float
) -> torch.Tensor:
    """Return a float per-example mask for resolutions inside the allowed range."""
    resolution = resolution.reshape(-1)
    return ((resolution >= min_resolution) & (resolution <= max_resolution)).to(
        dtype=torch.float32,
        device=resolution.device,
    )


def _apply_resolution_gate(
    loss: torch.Tensor,
    has_valid_resolution: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reduce loss over examples with valid resolution only."""
    valid_resolution = has_valid_resolution.to(device=loss.device, dtype=loss.dtype)
    valid_resolution = valid_resolution.reshape(-1)

    if valid_resolution.numel() == 1:
        return loss * valid_resolution.squeeze()
    if loss.dim() == 0 or loss.shape[0] != valid_resolution.numel():
        raise ValueError(
            "Resolution-gated losses must be scalar with a single resolution mask "
            "or have a leading per-example dimension matching the resolution mask."
        )

    per_example_loss = loss.reshape(valid_resolution.numel(), -1).mean(dim=-1)
    valid_loss = per_example_loss * valid_resolution
    valid_count = valid_resolution.sum()
    if valid_count == 0:
        return valid_loss.sum() * 0.0

    if reduction is None:
        return valid_loss
    if reduction == "mean":
        return valid_loss.sum() / valid_count
    if reduction in ["sum", "add"]:
        return valid_loss.sum()

    return loss_reduction(per_example_loss[valid_resolution.bool()], method=reduction)


def _apply_last_dim_mask(
    tensor: torch.Tensor, mask: torch.Tensor, name: str = "mask"
) -> torch.Tensor:
    """Apply a last-dimension mask without flattening ragged batched selections."""
    if mask.shape[-1] != tensor.shape[-1]:
        raise ValueError(
            f"{name} last dimension ({mask.shape[-1]}) must match tensor last "
            f"dimension ({tensor.shape[-1]})."
        )

    expanded_mask = mask.to(device=tensor.device, dtype=torch.bool)
    while expanded_mask.dim() < tensor.dim():
        expanded_mask = expanded_mask.unsqueeze(-2)

    try:
        torch.broadcast_shapes(tensor.shape, expanded_mask.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(mask.shape)} cannot be broadcast to tensor "
            f"shape {tuple(tensor.shape)} along the last dimension."
        ) from exc

    if tensor.dtype == torch.bool:
        return tensor & expanded_mask
    return tensor * expanded_mask.to(dtype=tensor.dtype)


def _diffusion_condition_align_mask(
    is_condition_atom: torch.Tensor,
    atom_padding_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.3,
) -> torch.Tensor:
    condition_mask = is_condition_atom.bool()
    if atom_padding_mask is None:
        valid_atom_mask = torch.ones_like(condition_mask, dtype=torch.bool)
    else:
        valid_atom_mask = ~atom_padding_mask.bool()

    condition_count = (condition_mask & valid_atom_mask).sum(dim=-1)
    valid_count = valid_atom_mask.sum(dim=-1).clamp_min(1)
    condition_fraction = condition_count / valid_count
    use_condition_align = condition_fraction >= threshold
    return torch.where(
        use_condition_align.unsqueeze(dim=-1),
        condition_mask,
        torch.ones_like(condition_mask, dtype=torch.bool),
    )


def _broadcast_prefix(
    tensor: torch.Tensor,
    prefix_shape: torch.Size,
    event_ndim: int,
    name: str,
) -> torch.Tensor:
    """Broadcast leading dimensions to a known prefix without changing event dims."""
    tensor_prefix = tensor.shape[:-event_ndim]
    event_shape = tensor.shape[-event_ndim:]
    try:
        common_prefix = torch.broadcast_shapes(tensor_prefix, prefix_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} prefix shape {tuple(tensor_prefix)} cannot broadcast to "
            f"{tuple(prefix_shape)}."
        ) from exc
    if common_prefix != prefix_shape:
        raise ValueError(
            f"{name} prefix shape {tuple(tensor_prefix)} cannot broadcast to "
            f"{tuple(prefix_shape)}."
        )

    padded_prefix = (1,) * (len(prefix_shape) - len(tensor_prefix)) + tuple(
        tensor_prefix
    )
    return tensor.reshape(padded_prefix + tuple(event_shape)).expand(
        tuple(prefix_shape) + tuple(event_shape)
    )


class BondTypeLoss(nn.Module):
    # Copyright 2025 ODesign Team and/or its affiliates.
    # Licensed under the Apache License, Version 2.0 (the "License");
    """
    BondTypeLoss computes softmax cross entropy loss over predicted bond types
    """

    def __init__(
        self,
        num_classes: int = len(BOND_TYPE),
        eps: float = 1e-6,
        k: float = 2.0,
        alpha: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            num_classes (int, optional): Number of bond types.
            eps (float, optional): Small epsilon to avoid divide by zero.
            enable_t_weight (bool, optional): Whether to weigh loss according to sampled time.
            reduction (str, optional): Reduction method.
        """
        super(BondTypeLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.k = k
        self.alpha = alpha
        self.reduction = reduction

    def _match_target_batch(
        self,
        target: torch.Tensor,
        logits: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        expected_pair_shape = logits.shape[1:3]
        if target.dim() == 2:
            target = target.unsqueeze(0).expand(batch_size, -1, -1)
        elif target.dim() == 3:
            if target.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size ({target.shape[0]}) must match logits "
                    f"batch size ({batch_size})."
                )
        else:
            raise ValueError(
                f"{name} must have rank 2 [N, N] or rank 3 [B, N, N], "
                f"got rank {target.dim()}."
            )

        if target.shape[-2:] != expected_pair_shape:
            raise ValueError(
                f"{name} pair shape {tuple(target.shape[-2:])} must match logits "
                f"pair shape {tuple(expected_pair_shape)}."
            )
        return target

    def forward(
        self,
        logits: torch.Tensor,          # [B, N, N, num_classes]
        bond_labels: torch.Tensor,     # [N, N]
        bond_gen_flag: torch.Tensor,   # [N, N] 
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits for bond types.
            bond_labels: Ground truth labels as integer indices.
            bond_gen_flag: Mask indicating which positions to compute loss.

        Returns:
            Reduced loss.
        """
        # Expand legacy unbatched labels/masks or validate already-batched inputs.
        bond_labels = self._match_target_batch(
            bond_labels,
            logits,
            name="bond_labels",
        ).to(device=logits.device, dtype=torch.long)
        bond_gen_flag = self._match_target_batch(
            bond_gen_flag,
            logits,
            name="bond_gen_flag",
        ).to(device=logits.device, dtype=logits.dtype)

        # One-hot labels: [B, N, N, num_classes]
        bond_labels_one_hot = F.one_hot(
            bond_labels, num_classes=self.num_classes
        ).to(dtype=logits.dtype)

        # Compute per-position softmax cross entropy
        # logits: [B, N, N, num_classes]
        # labels_one_hot: [B, N, N, num_classes]
        errors = softmax_cross_entropy(
            logits=logits,
            labels=bond_labels_one_hot
        )  # [B, N, N]

        # Mask and normalize
        masked_errors = errors * bond_gen_flag    # [B, N, N]
        denom = self.eps + torch.sum(bond_gen_flag, dim=(-1, -2))  # [B]
        loss = torch.sum(masked_errors, dim=(-1, -2)) / denom      # [B]

        return loss_reduction(loss, method=self.reduction)

class SmoothLDDTLoss(nn.Module):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None):
        if c_lm is not None:
            true_distance = true_distance.unsqueeze(dim=-3)
        dist_diff = torch.abs(pred_distance - true_distance)
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)

        # Compute mean
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(c_lm, dim=(-1, -2)) + self.eps
            )  # [..., N_sample]
        else:
            # It's for sparse forward mode
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        if diffusion_chunk_size is None:
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            # Default use checkpoint for saving memory
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    c_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        if lddt_mask.dim() > 2:
            prefix_shape = lddt_mask.shape[:-2]
            pred_coordinate = _broadcast_prefix(
                pred_coordinate,
                prefix_shape,
                event_ndim=3,
                name="pred_coordinate",
            )
            true_coordinate = _broadcast_prefix(
                true_coordinate,
                prefix_shape,
                event_ndim=2,
                name="true_coordinate",
            )

            flat_lddt_mask = lddt_mask.reshape(-1, *lddt_mask.shape[-2:])
            flat_pred_coordinate = pred_coordinate.reshape(
                -1, *pred_coordinate.shape[-3:]
            )
            flat_true_coordinate = true_coordinate.reshape(
                -1, *true_coordinate.shape[-2:]
            )
            flat_losses = [
                self.sparse_forward(
                    pred_coordinate=flat_pred_coordinate[i],
                    true_coordinate=flat_true_coordinate[i],
                    lddt_mask=flat_lddt_mask[i],
                    diffusion_chunk_size=diffusion_chunk_size,
                )
                for i in range(flat_lddt_mask.shape[0])
            ]
            stacked_losses = torch.stack(flat_losses).reshape(prefix_shape)
            if self.reduction is None:
                return stacked_losses
            return loss_reduction(stacked_losses, method=self.reduction)

        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        if lddt_indices[0].numel() == 0:
            zero_loss = pred_coordinate.sum(dim=(-1, -2, -3)) * 0.0
            return loss_reduction(zero_loss, method=self.reduction)

        true_coords_l = true_coordinate.index_select(-2, lddt_indices[0])
        true_coords_m = true_coordinate.index_select(-2, lddt_indices[1])
        true_distance_sparse_lm = torch.norm(true_coords_l - true_coords_m, p=2, dim=-1)
        if diffusion_chunk_size is None:
            pred_coords_l = pred_coordinate.index_select(-2, lddt_indices[0])
            pred_coords_m = pred_coordinate.index_select(-2, lddt_indices[1])
            # \delta x_{lm} and \delta x_{lm}^{GT} in the Algorithm 27
            pred_distance_sparse_lm = torch.norm(
                pred_coords_l - pred_coords_m, p=2, dim=-1
            )
            lddt = self._chunk_forward(
                pred_distance_sparse_lm, true_distance_sparse_lm, c_lm=None
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_coords_i_l = pred_coordinate[
                    ...,
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                    :,
                    :,
                ].index_select(-2, lddt_indices[0])
                pred_coords_i_m = pred_coordinate[
                    ...,
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                    :,
                    :,
                ].index_select(-2, lddt_indices[1])

                # \delta x_{lm} and \delta x_{lm}^{GT} in the Algorithm 27
                pred_distance_sparse_i_lm = torch.norm(
                    pred_coords_i_l - pred_coords_i_m, p=2, dim=-1
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_sparse_i_lm,
                    true_distance_sparse_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def dense_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        true_distance = torch.cdist(true_coordinate, true_coordinate)
        if diffusion_chunk_size is None:
            pred_distance = torch.cdist(pred_coordinate, pred_coordinate)
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_coordinate_i = pred_coordinate[
                    ...,
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                    :,
                    :,
                ]
                pred_distance_i = torch.cdist(pred_coordinate_i, pred_coordinate_i)
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_i,
                    true_distance,
                    c_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)


class BondLoss(nn.Module):
    """
    Implements Formula 5 [BondLoss] in AF3
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        """BondLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(BondLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, bond_mask):
        # Distance squared error
        # [...,  N_sample , N_atom, N_atom]
        dist_squared_err = (pred_distance - true_distance.unsqueeze(dim=-3)) ** 2
        bond_loss = torch.sum(dist_squared_err * bond_mask, dim=(-1, -2)) / torch.sum(
            bond_mask + self.eps, dim=(-1, -2)
        )  # [..., N_sample]
        return bond_loss

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """BondLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = (bond_mask * distance_mask).unsqueeze(
            dim=-3
        )  # [1, N_atom, N_atom] or [..., 1, N_atom, N_atom]
        # Bond Loss
        if diffusion_chunk_size is None:
            bond_loss = self._chunk_forward(
                pred_distance=pred_distance,
                true_distance=true_distance,
                bond_mask=bond_mask,
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            bond_loss = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                bond_loss_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    bond_mask,
                )
                bond_loss.append(bond_loss_i)
            bond_loss = torch.cat(bond_loss, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """BondLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = bond_mask * distance_mask
        if bond_mask.dim() > 2:
            prefix_shape = bond_mask.shape[:-2]
            pred_coordinate = _broadcast_prefix(
                pred_coordinate,
                prefix_shape,
                event_ndim=3,
                name="pred_coordinate",
            )
            true_coordinate = _broadcast_prefix(
                true_coordinate,
                prefix_shape,
                event_ndim=2,
                name="true_coordinate",
            )
            if per_sample_scale is not None:
                per_sample_scale = _broadcast_prefix(
                    per_sample_scale,
                    prefix_shape,
                    event_ndim=1,
                    name="per_sample_scale",
                )

            flat_bond_mask = bond_mask.reshape(-1, *bond_mask.shape[-2:])
            flat_pred_coordinate = pred_coordinate.reshape(
                -1, *pred_coordinate.shape[-3:]
            )
            flat_true_coordinate = true_coordinate.reshape(
                -1, *true_coordinate.shape[-2:]
            )
            flat_per_sample_scale = (
                None
                if per_sample_scale is None
                else per_sample_scale.reshape(-1, per_sample_scale.shape[-1])
            )
            flat_losses = [
                self.sparse_forward(
                    pred_coordinate=flat_pred_coordinate[i],
                    true_coordinate=flat_true_coordinate[i],
                    distance_mask=torch.ones_like(flat_bond_mask[i]),
                    bond_mask=flat_bond_mask[i],
                    per_sample_scale=(
                        None
                        if flat_per_sample_scale is None
                        else flat_per_sample_scale[i]
                    ),
                )
                for i in range(flat_bond_mask.shape[0])
            ]
            stacked_losses = torch.stack(flat_losses).reshape(prefix_shape)
            if self.reduction is None:
                return stacked_losses
            return loss_reduction(stacked_losses, method=self.reduction)

        bond_indices = torch.nonzero(bond_mask, as_tuple=True)
        pred_coords_i = pred_coordinate.index_select(-2, bond_indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, bond_indices[1])
        true_coords_i = true_coordinate.index_select(-2, bond_indices[0])
        true_coords_j = true_coordinate.index_select(-2, bond_indices[1])

        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)
        dist_squared_err_sparse = (pred_distance_sparse - true_distance_sparse) ** 2
        # Protecting special data that has size: tensor([], size=(x, 0), grad_fn=<PowBackward0>)
        if dist_squared_err_sparse.numel() == 0:
            return pred_coordinate.sum(dim=(-1, -2, -3)) * 0.0
        bond_loss = torch.mean(dist_squared_err_sparse, dim=-1)  # [..., N_sample]
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return bond_loss


def compute_lddt_mask(
    true_distance: torch.Tensor,
    distance_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> torch.Tensor:
    """calculate the atom pair mask with the bespoke radius

    Args:
        true_distance (torch.Tensor): the ground truth coordinates
            [..., N_atom, N_atom]
        distance_mask (torch.Tensor): whether true coordinates exist.
            [..., N_atom, N_atom] or [N_atom, N_atom]
        is_nucleotide (torch.Tensor): Indicator for nucleotide atoms.
            [..., N_atom] or [N_atom]
        is_nucleotide_threshold (float): Threshold distance for nucleotide atoms. Defaults to 30.0.
        is_not_nucleotide_threshold (float): Threshold distance for non-nucleotide atoms. Defaults to 15.0.

    Returns:
        c_lm (torch.Tenson): the atom pair mask c_lm, not symmetric
            [..., N_atom, N_atom]
    """
    # Restrict to bespoke inclusion radius
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None] + (
        true_distance < is_not_nucleotide_threshold
    ) * (
        ~is_nucleotide_mask[..., None]
    )  # [..., N_atom, N_atom]

    # Zero-out diagonals of c_lm and cast to float
    c_lm = c_lm * (
        1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=true_distance.dtype)
    )
    # Zero-out atom pairs without true coordinates
    # Note: the sparsity of c_lm is ~10% in 5000 atom-pairs,
    # and becomes more sparse as the number of atoms increases,
    # change to sparse implementation can reduce cuda memory
    c_lm = c_lm * distance_mask  # [..., N_atom, N_atom]
    return c_lm


def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softmax cross entropy

    Args:
        logits (torch.Tensor): classification logits
            [..., num_class]
        labels (torch.Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        torch.Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


class DistogramLoss(nn.Module):
    """
    Implements DistogramLoss in AF3
    """

    def __init__(
        self,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """Distogram loss
        This head and loss are identical to AlphaFold 2, where the pairwise token distances use the representative atom for each token:
            Cβ for protein residues (Cα for glycine),
            C4 for purines and C2 for pyrimidines.
            All ligands already have a single atom per token.

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 2.3125.
            max_bin (float, optional): max boundary of bins. Defaults to 21.6875.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(DistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def update_label(
        self,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins - 1,
            device=true_coordinate.device,
        )

        rep_atom_mask = rep_atom_mask.bool()
        if rep_atom_mask.dim() > 1:
            prefix_shape = rep_atom_mask.shape[:-1]
            if true_coordinate.shape[:-2] != prefix_shape:
                true_coordinate = _broadcast_prefix(
                    true_coordinate,
                    prefix_shape,
                    event_ndim=2,
                    name="true_coordinate",
                )
            if coordinate_mask.shape[:-1] != prefix_shape:
                coordinate_mask = _broadcast_prefix(
                    coordinate_mask,
                    prefix_shape,
                    event_ndim=1,
                    name="coordinate_mask",
                )

            max_rep_atom = int(rep_atom_mask.sum(dim=-1).max().item())
            rep_coordinate = true_coordinate.new_zeros(
                *prefix_shape,
                max_rep_atom,
                true_coordinate.size(-1),
            )
            token_mask = coordinate_mask.new_zeros(*prefix_shape, max_rep_atom)

            flat_rep_coordinate = rep_coordinate.reshape(
                -1, max_rep_atom, true_coordinate.size(-1)
            )
            flat_token_mask = token_mask.reshape(-1, max_rep_atom)
            flat_true_coordinate = true_coordinate.reshape(
                -1, *true_coordinate.shape[-2:]
            )
            flat_coordinate_mask = coordinate_mask.reshape(
                -1, coordinate_mask.shape[-1]
            )
            flat_rep_atom_mask = rep_atom_mask.reshape(-1, rep_atom_mask.shape[-1])
            for index, mask in enumerate(flat_rep_atom_mask):
                n_rep_atom = int(mask.sum().item())
                if n_rep_atom == 0:
                    continue
                flat_rep_coordinate[index, :n_rep_atom] = flat_true_coordinate[
                    index, mask
                ]
                flat_token_mask[index, :n_rep_atom] = flat_coordinate_mask[index, mask]
            true_coordinate = rep_coordinate
        else:
            true_coordinate = true_coordinate[..., rep_atom_mask, :]  # [..., N_token, 3]
            token_mask = coordinate_mask[..., rep_atom_mask]

        # Compute label: the true bins
        # True distance
        gt_dist = cdist(true_coordinate, true_coordinate)  # [..., N_token, N_token]
        # Assign distance to bins
        true_bins = torch.sum(
            gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1
        )  # range in [0, no_bins-1], shape = [..., N_token, N_token]

        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(
        self,
        logits: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Distogram loss

        Args:
            logits (torch.Tensor): logits.
                [..., N_token, N_token, no_bins]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask.
                [N_atom]

        Returns:
            torch.Tensor: the return loss.
                [...] if self.reduction is not None else []
        """

        with torch.no_grad():
            true_bins, pair_mask = self.update_label(
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_token, N_token]

        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2))
        loss = loss / denom

        return loss_reduction(loss, method=self.reduction)


# Algorithm 30 Compute alignment error
def compute_alignment_error_squared(
    pred_coordinate: torch.Tensor,
    true_coordinate: torch.Tensor,
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
) -> torch.Tensor:
    """Implements Algorithm 30 Compute alignment error, but do not take the square root

    Args:
        pred_coordinate (torch.Tensor): the predict coords [frame center]
            [..., N_sample, N_token, 3]
        true_coordinate (torch.Tensor): the ground truth coords [frame center]
            [..., N_token, 3]
        pred_frames (torch.Tensor): the predict frame
            [..., N_sample, N_frame, 3, 3]
        true_frames (torch.Tensor): the ground truth frame
            [..., N_frame, 3, 3]

    Returns:
        torch.Tensor: the computed alignment error
            [..., N_sample, N_frame, N_token]
    """
    x_transformed_pred = express_coordinates_in_frame(
        coordinate=pred_coordinate, frames=pred_frames
    )  # [..., N_sample, N_frame, N_token, 3]
    x_transformed_true = express_coordinates_in_frame(
        coordinate=true_coordinate, frames=true_frames
    )  # [..., N_frame, N_token, 3]
    squared_pae = torch.sum(
        (x_transformed_pred - x_transformed_true.unsqueeze(dim=-4)) ** 2, dim=-1
    )  # [..., N_sample, N_frame, N_token]
    return squared_pae


class MSELoss(nn.Module):
    # Copyright 2025 ODesign Team and/or its affiliates.
    # Licensed under the Apache License, Version 2.0 (the "License");
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(
        self,
        weight_mse: float = 1 / 3,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
    ) -> None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction

    def weighted_rigid_align(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        align_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        N_sample = pred_coordinate.size(-3)
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [N_atom] or [..., N_atom]

        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        align_weight = _apply_last_dim_mask(
            weight,
            align_mask,
            name="align_mask",
        )
        true_coordinate = true_coordinate * coordinate_mask.unsqueeze(dim=-1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]

        # Reshape to add "N_sample" dimension
        true_coordinate = expand_at_dim(
            true_coordinate, dim=-3, n=N_sample
        )  # [..., N_sample, N_atom, 3]
        if len(weight.shape) > 1:
            weight = expand_at_dim(
                weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]
            align_weight = expand_at_dim(
                align_weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]

        # Align GT coords to predicted coords
        d = pred_coordinate.dtype
        # Some ops in weighted_rigid_align do not support BFloat16 training
        with torch.cuda.amp.autocast(enabled=False):
            true_coordinate_aligned = weighted_rigid_align(
                x=true_coordinate.to(torch.float32),  # [..., N_sample, N_atom, 3]
                x_target=pred_coordinate.to(
                    torch.float32
                ),  # [..., N_sample, N_atom, 3]


                atom_weight=align_weight.to(
                    torch.float32
                ),  # [N_atom] or [..., N_sample, N_atom]
                stop_gradient=True,
            )  # [..., N_sample, N_atom, 3]
            true_coordinate_aligned = true_coordinate_aligned.to(d)

        return (true_coordinate_aligned.detach(), weight.detach())

    def calc_mse(self, pred_x, true_x, weight, coordinate_mask, per_sample_scale=None):
        """Calculate MSE loss

        Args:
            pred_x (torch.Tensor): predicted coordinates
                [..., N_sample, N_atom, 3]
            true_x (torch.Tensor): true coordinates
                [..., N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss
                [...] if reduction is None else []
        """
        if pred_x.dim() == true_x.dim() + 1:
            true_x = true_x.unsqueeze(dim=-3)
        per_atom_se = ((pred_x - true_x) ** 2).sum(dim=-1)  # [..., N_sample, N_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (
            coordinate_mask.sum(dim=-1, keepdim=True) + self.eps
        )  # [..., N_sample]
        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale
        weighted_mse_loss = self.weight_mse * (per_sample_weighted_mse).mean(
            dim=-1
        )
        # [..., N_sample]
        loss = loss_reduction(weighted_mse_loss, method=self.reduction)
        return loss

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        not_condition_atom,
        align_mask,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # True_coordinate_aligned: [..., N_sample, N_atom, 3]
        # Weight: [N_atom] or [..., N_sample, N_atom]

        with torch.no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
                align_mask=align_mask,
            )

        no_condition = align_mask.all()

        # Calculate MSE loss global align
        global_align_loss = self.calc_mse(
            pred_x=pred_coordinate,
            true_x=true_coordinate_aligned,
            weight=weight,
            coordinate_mask=coordinate_mask,
            per_sample_scale=per_sample_scale,
        )

        # Calculate partial align loss
        condition_align_loss = self.calc_mse(
            pred_x=pred_coordinate,
            true_x=true_coordinate,
            weight=weight,
            coordinate_mask=coordinate_mask,
            per_sample_scale=per_sample_scale,
        )

        non_condition_coordinate_mask = _apply_last_dim_mask(
            coordinate_mask,
            not_condition_atom,
            name="not_condition_atom",
        )
        non_condition_weight = _apply_last_dim_mask(
            weight,
            not_condition_atom,
            name="not_condition_atom",
        )

        # Calculate loss for non-condition atoms with condition alignment
        condition_align_loss_wo_condition = self.calc_mse(
            pred_x=pred_coordinate,
            true_x=true_coordinate,
            weight=non_condition_weight,
            coordinate_mask=non_condition_coordinate_mask,
            per_sample_scale=per_sample_scale,
        )

        # Calculate loss for non-condition atoms with global alignment
        global_align_loss_wo_condition = self.calc_mse(
            pred_x=pred_coordinate,
            true_x=true_coordinate_aligned,
            weight=non_condition_weight,
            coordinate_mask=non_condition_coordinate_mask,
            per_sample_scale=per_sample_scale,
        )

        # Align non-condition atoms independently
        with torch.no_grad():
            gen_alig_true_coordinate_aligned, gen_weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=non_condition_coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
                align_mask=not_condition_atom,
            )
        gen_align_loss_wo_condition = self.calc_mse(
            pred_x=pred_coordinate,
            true_x=gen_alig_true_coordinate_aligned,
            weight=gen_weight,
            coordinate_mask=non_condition_coordinate_mask,
            per_sample_scale=per_sample_scale,
        )

        return (
            gen_align_loss_wo_condition,
            global_align_loss,
            condition_align_loss,
            global_align_loss_wo_condition,
            condition_align_loss_wo_condition,
            no_condition,
        )


def calculate_atom_bespoke_lddt(
    pred_coordinate: torch.Tensor,
    true_coordinate: torch.Tensor,
    is_nucleotide: torch.Tensor,
    is_polymer: torch.Tensor,
    rep_atom_mask: torch.Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> torch.Tensor:
    """calculate the bespoke lddt as described in Sec 4.3.1.
    Args:
        pred_coordinate (torch.Tensor):
            [..., N_sample, N_atom, 3]
        true_coordinate (torch.Tensor):
            [..., N_atom]
        is_nucleotide (torch.Tensor):
            [N_atom] or [..., N_atom]
        is_polymer (torch.Tensor):
            [N_atom]
        rep_atom_mask (torch.Tensor):
            [N_atom]
    Returns:
        torch.Tensor: per-atom lddt
            [..., N_sample, N_atom, 1]
        torch.Tensor: per-atom lddt weight
            [..., N_sample, N_atom, 1]
    """

    N_atom = true_coordinate.size(-2)
    atom_m_mask = (rep_atom_mask * is_polymer).bool()  # [N_atom]
    # Distance: d_lm
    pred_d_lm = torch.cdist(
        pred_coordinate, pred_coordinate[..., atom_m_mask, :]
    )  # [..., N_sample, N_atom, N_atom(m)]
    true_d_lm = torch.cdist(
        true_coordinate, true_coordinate[..., atom_m_mask, :]
    )  # [..., N_atom, N_atom(m)]
    delta_d_lm = torch.abs(
        pred_d_lm - true_d_lm.unsqueeze(dim=-3)
    )  # [..., N_sample, N_atom, N_atom(m)]
    # Pair-wise lddt
    thresholds = [0.5, 1, 2, 4]
    lddt_lm = (
        torch.stack([delta_d_lm < t for t in thresholds], dim=-1)
        .to(dtype=delta_d_lm.dtype)
        .mean(dim=-1)
    )  # [..., N_sample, N_atom, N_atom(m)]
    # Select atoms that are within certain threshold to l in ground truth
    # Restrict to bespoke inclusion radius
    is_nucleotide = is_nucleotide[
        ..., atom_m_mask
    ].bool()  # [N_atom(m)] or [..., N_atom(m)]
    locality_mask = (true_d_lm < is_nucleotide_threshold) * is_nucleotide.unsqueeze(
        dim=-2
    ) + (true_d_lm < is_not_nucleotide_threshold) * (
        ~is_nucleotide.unsqueeze(dim=-2)
    )  # [..., N_atom, N_atom(m)]
    # Remove self-distance computation
    diagonal_mask = ((1 - torch.eye(n=N_atom)).bool().to(true_d_lm.device))[
        ..., atom_m_mask
    ]  # [N_atom, N_atom(m)]
    pair_mask = (locality_mask * diagonal_mask).unsqueeze(
        dim=-3
    )  # [..., 1, N_atom, N_atom(m)]
    per_atom_lddt = torch.sum(
        lddt_lm * pair_mask, dim=-1, keepdim=True
    )  # [...,  N_sample, N_atom, 1]
    per_atom_weight = torch.sum(pair_mask.to(dtype=lddt_lm.dtype), dim=-1, keepdim=True)
    return per_atom_lddt, per_atom_weight


class ODesignLoss(nn.Module):
    # Copyright 2025 ODesign Team and/or its affiliates.
    # Licensed under the Apache License, Version 2.0 (the "License");
    """Aggregation of the various losses"""

    def __init__(self, configs) -> None:
        super(ODesignLoss, self).__init__()
        self.configs = configs

        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt
        self.alpha_bond_type = self.configs.loss.weight.alpha_bond_type

        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.loss_weight = {
            # diffusion
            "mse_loss": self.alpha_diffusion,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion
            * self.weight_smooth_lddt,  # Different from AF3 appendix eq(6), where smooth_lddt has no weight
            # distogram
            "distogram_loss": self.alpha_distogram,
            # bond type
            "bond_type_loss": self.alpha_bond_type,
        }

        # Loss
        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        self.bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.smooth_lddt_loss = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)
        self.distogram_loss = DistogramLoss(**configs.loss.distogram)
        self.bond_type_loss = BondTypeLoss(**configs.loss.bond_type)

    def update_label(
        self,
        loss_input: LossInput,
        ground_truth: GroundTruth,
    ) -> dict[str, Any]:
        """calculate true distance, and atom pair mask

        Args:
            feat_dict (dict): Feature dictionary containing additional features.
            label_dict (dict): Label dictionary containing ground truth data.

        Returns:
            label_dict (dict): with the following updates:
                distance (torch.Tensor): true atom-atom distance.
                    [..., N_atom, N_atom]
                distance_mask (torch.Tensor): atom-atom mask indicating whether true distance exists.
                    [..., N_atom, N_atom]
        """
        legacy_unbatched = ground_truth.coordinate.dim() == 2
        coordinate = (
            ground_truth.coordinate.unsqueeze(0)
            if legacy_unbatched
            else ground_truth.coordinate
        )
        coordinate_mask = (
            ground_truth.coordinate_mask.unsqueeze(0)
            if legacy_unbatched
            else ground_truth.coordinate_mask
        )
        is_nucleotide = torch.logical_or(loss_input.is_rna, loss_input.is_dna)
        if legacy_unbatched:
            is_nucleotide = is_nucleotide.unsqueeze(0)

        atom_padding_mask = getattr(loss_input, "atom_padding_mask", None)
        if legacy_unbatched and atom_padding_mask is not None:
            atom_padding_mask = atom_padding_mask.unsqueeze(0)

        batch_shape = coordinate.shape[:-2]
        num_atom = coordinate.shape[-2]
        distance = coordinate.new_zeros(*batch_shape, num_atom, num_atom)
        distance_mask = coordinate_mask.new_zeros(*batch_shape, num_atom, num_atom)
        lddt_mask = coordinate.new_zeros(*batch_shape, num_atom, num_atom)

        flat_coordinate = coordinate.reshape(-1, num_atom, coordinate.shape[-1])
        flat_coordinate_mask = coordinate_mask.reshape(-1, num_atom)
        flat_is_nucleotide = is_nucleotide.reshape(-1, num_atom)
        flat_distance = distance.reshape(-1, num_atom, num_atom)
        flat_distance_mask = distance_mask.reshape(-1, num_atom, num_atom)
        flat_lddt_mask = lddt_mask.reshape(-1, num_atom, num_atom)
        flat_atom_padding_mask = (
            None
            if atom_padding_mask is None
            else atom_padding_mask.reshape(-1, num_atom)
        )

        # Compute label distances per real sample. This keeps LDDT labels
        # independent of co-batched padding shape, so bsz1 and padded bsz2 see
        # the same per-sample atom-pair mask.
        for sample_idx in range(flat_coordinate.shape[0]):
            if flat_atom_padding_mask is None:
                real_atom_len = num_atom
            else:
                real_atom_len = int(
                    (~flat_atom_padding_mask[sample_idx].bool()).sum().item()
                )
            if real_atom_len == 0:
                continue

            sample_coordinate = flat_coordinate[
                sample_idx : sample_idx + 1, :real_atom_len
            ]
            sample_coordinate_mask = flat_coordinate_mask[
                sample_idx : sample_idx + 1, :real_atom_len
            ]
            sample_distance_mask = (
                sample_coordinate_mask[..., None]
                * sample_coordinate_mask[..., None, :]
            )
            # Note: we convert to bf16 for saving cuda memory, if performance drops, do not convert it
            sample_distance = (
                cdist(sample_coordinate, sample_coordinate) * sample_distance_mask
            ).to(ground_truth.coordinate.dtype)
            sample_lddt_mask = compute_lddt_mask(
                true_distance=sample_distance,
                distance_mask=sample_distance_mask,
                is_nucleotide=flat_is_nucleotide[
                    sample_idx : sample_idx + 1, :real_atom_len
                ],
                **self.lddt_radius,
            )

            flat_distance[
                sample_idx, :real_atom_len, :real_atom_len
            ] = sample_distance.squeeze(0)
            flat_distance_mask[
                sample_idx, :real_atom_len, :real_atom_len
            ] = sample_distance_mask.squeeze(0)
            flat_lddt_mask[
                sample_idx, :real_atom_len, :real_atom_len
            ] = sample_lddt_mask.squeeze(0)

        if legacy_unbatched:
            distance = distance.squeeze(0)
            distance_mask = distance_mask.squeeze(0)
            lddt_mask = lddt_mask.squeeze(0)

        ground_truth.update(
            {
                "lddt_mask": lddt_mask,
                "distance_mask": distance_mask,
            }
        )
        if not self.configs.model.loss_metrics_sparse_enable:
            ground_truth.update({"distance": distance})
        del distance, distance_mask, lddt_mask
        return ground_truth

    def calculate_prediction(
        self,
        pred_output: ODesignOutput,
    ) -> dict[str, torch.Tensor]:
        """get more predictions used for calculating difference losses

        Args:
            pred_dict (dict[str, torch.Tensor]): raw prediction dict given by the model

        Returns:
            dict[str, torch.Tensor]: updated predictions
        """
        if not self.configs.model.loss_metrics_sparse_enable:
            distance = torch.cdist(
                pred_output["coordinate"], pred_output["coordinate"]
            ).to(
                pred_output["coordinate"].dtype
            )  # [..., N_atom, N_atom]
            pred_output.update({"distance": distance})  
        return pred_output

    def aggregate_losses(
        self, loss_fns: dict, has_valid_resolution: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.
            has_valid_resolution (Optional[torch.Tensor]): Tensor indicating valid resolutions. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
        """
        cum_loss = 0.0
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple) and loss_name != 'mse_loss':
                loss, metrics = loss_outputs
            elif loss_name == 'mse_loss':
                (
                    gen_align_loss_wo_condition, 
                    global_align_loss, 
                    condition_align_loss, 
                    global_align_loss_wo_condition, 
                    condition_align_loss_wo_condition, 
                    no_condition
                 ) = loss_outputs
                loss = gen_align_loss_wo_condition
                metrics = {}
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            if (
                (has_valid_resolution is not None)
                and (
                    loss_name in ["plddt_loss", "pde_loss", "resolved_loss", "pae_loss"]
                )
            ):
                loss = _apply_resolution_gate(loss, has_valid_resolution)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning(f"{loss_name} loss is NaN. Skipping...")

            all_metrics[loss_name] = loss.detach().clone()
            all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()
            if loss_name == 'mse_loss':
                all_metrics[f"global_align_wo_condition_{loss_name}"] = global_align_loss_wo_condition.detach().clone()
                all_metrics[f"condition_align_wo_condition_{loss_name}"] = condition_align_loss_wo_condition.detach().clone()
                all_metrics[f'global_align_{loss_name}'] = global_align_loss.detach().clone()
                all_metrics[f'condition_align_{loss_name}'] = condition_align_loss.detach().clone()
                all_metrics[f'gen_align_{loss_name}'] = gen_align_loss_wo_condition.detach().clone()
            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics

    def calculate_losses(
        self,
        loss_input: LossInput,
        pred_output: ODesignOutput,
        ground_truth: GroundTruth,
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ["train", "eval", "inference"]
        if mode == "train":
            # Scale diffusion loss with noise-level
            diffusion_per_sample_scale = (
                pred_output["noise_level"] ** 2 + self.configs.model.sigma_data**2
            ) / (self.configs.model.sigma_data * pred_output["noise_level"]) ** 2

        else:
            # No scale is required
            diffusion_per_sample_scale = None

        # Diffusion Loss: SmoothLDDTLoss / BondLoss / MSELoss
        loss_fns = {}
        if self.configs.loss.diffusion_lddt_loss_dense:
            loss_fns.update(
                {
                    "smooth_lddt_loss": lambda: self.smooth_lddt_loss.dense_forward(
                        pred_coordinate=pred_output["coordinate"],
                        true_coordinate=ground_truth.coordinate,
                        lddt_mask=ground_truth.lddt_mask,
                        diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                    )  # it's faster is not OOM
                }
            )
        elif self.configs.loss.diffusion_sparse_loss_enable:
            loss_fns.update(
                {
                    "smooth_lddt_loss": lambda: self.smooth_lddt_loss.sparse_forward(
                        pred_coordinate=pred_output["coordinate"],
                        true_coordinate=ground_truth.coordinate,
                        lddt_mask=ground_truth.lddt_mask,
                        diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                    )
                }
            )
        else:
            loss_fns.update(
                {
                    "smooth_lddt_loss": lambda: self.smooth_lddt_loss(
                        pred_distance=pred_output["distance"],
                        true_distance=ground_truth.distance,
                        distance_mask=ground_truth.distance_mask,
                        lddt_mask=ground_truth.lddt_mask,
                        diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size,
                    )
                }
            )

        # 0.3 is empirical parameter
        if (
            set(self.configs.data_condition) & set(['diffusion'])
        ):
            align_mask = _diffusion_condition_align_mask(
                is_condition_atom=loss_input.is_condition_atom,
                atom_padding_mask=loss_input.atom_padding_mask,
                threshold=0.3,
            )
        else:  
            align_mask = torch.ones_like(loss_input.is_condition_atom, dtype=torch.bool)

        loss_fns.update(
            {
                "bond_loss": lambda: (
                    self.bond_loss.sparse_forward(
                        pred_coordinate=pred_output["coordinate"],
                        true_coordinate=ground_truth.coordinate,
                        distance_mask=ground_truth.distance_mask,
                        bond_mask=ground_truth.ligand_bond_mask,
                        per_sample_scale=diffusion_per_sample_scale,
                    )
                    if self.configs.loss.diffusion_sparse_loss_enable
                    else self.bond_loss(
                        pred_distance=pred_output["distance"],
                        true_distance=ground_truth.distance,
                        distance_mask=ground_truth.distance_mask,
                        bond_mask=ground_truth.ligand_bond_mask,
                        per_sample_scale=diffusion_per_sample_scale,
                        diffusion_chunk_size=self.configs.loss.diffusion_bond_chunk_size,
                    )
                ),
                "mse_loss": lambda: self.mse_loss(
                    pred_coordinate=pred_output["coordinate"],
                    true_coordinate=ground_truth.coordinate,
                    coordinate_mask=ground_truth.coordinate_mask,
                    is_rna=loss_input.is_rna,
                    is_dna=loss_input.is_dna,
                    is_ligand=loss_input.is_ligand,
                    not_condition_atom=torch.logical_not(loss_input.is_condition_atom),
                    align_mask=align_mask,
                    per_sample_scale=diffusion_per_sample_scale,
                ),
            }
        )
        # Distogram Loss
        if "distogram" in pred_output and pred_output["distogram"] is not None:
            loss_fns.update(
                {
                    "distogram_loss": lambda: self.distogram_loss(
                        logits=pred_output["distogram"],
                        true_coordinate=ground_truth.coordinate,
                        coordinate_mask=ground_truth.coordinate_mask,
                        rep_atom_mask=loss_input.distogram_rep_atom_mask,
                    )
                }
            )

        # Confidence Loss:
        # Only when resolution is in [min_resolution, max_resolution] the confidence loss is considered.
        has_valid_resolution = _valid_resolution_mask(
            loss_input.resolution,
            min_resolution=self.configs.loss.resolution.min,
            max_resolution=self.configs.loss.resolution.max,
        ).to(dtype=ground_truth.coordinate.dtype, device=ground_truth.coordinate.device)

        if self.configs.bond_reconstruction:
            loss_fns.update({
                "bond_type_loss": lambda: self.bond_type_loss(
                    pred_output["token_bond_type_logits"],
                    ground_truth.token_bond_type_label,
                    pred_output["token_bond_gen_mask"],
                )
            })
        
        cum_loss, metrics = self.aggregate_losses(
            loss_fns,
            has_valid_resolution,
        )
        return cum_loss, metrics

    def forward(
        self,
        loss_input: LossInput,
        pred_output: ODesignOutput,
        ground_truth: GroundTruth,
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ["train", "eval", "inference"]
        # Pre-computations
        with torch.no_grad():
            ground_truth = self.update_label(loss_input, ground_truth)

        pred_output = self.calculate_prediction(pred_output)

        # Calculate losses
        cum_loss, losses = self.calculate_losses(
            loss_input=loss_input,
            pred_output=pred_output,
            ground_truth=ground_truth,
            mode=mode,
        )

        return cum_loss, losses
