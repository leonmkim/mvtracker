import logging
import os
from collections import defaultdict
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import nn as nn

from mvtracker.datasets.utils import transform_scene
from mvtracker.models.core.cotracker2.blocks import Attention, FlashAttention
from mvtracker.models.core.cotracker2.blocks import EfficientUpdateFormer
from mvtracker.models.core.embeddings import (
    get_3d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid,
    get_3d_embedding,
)
from mvtracker.models.core.model_utils import smart_cat, init_pointcloud_from_rgbd, save_pointcloud_to_ply
from mvtracker.models.core.spatracker.blocks import BasicEncoder
from mvtracker.utils.basic import time_now


# ---------- KNN backends ----------
def _knn_pointops(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    """
    Efficient batched KNN using pointops library.

    This is slightly faster than torch.cdist + torch.topk and uses less memory:

    Example::

        Benchmarking KNN with different methods (HALF_PRECISION=True):
        torch.cdist+torch.topk   | Avg Time: 0.008380 s | Peak Memory: 1151.19 MB (min: 1151.19, max: 1151.19)
        pointops.knn_query       | Avg Time: 0.007477 s | Peak Memory:  47.22 MB (min:  47.22, max:  47.22)

        Benchmarking KNN with different methods (HALF_PRECISION=False):
        torch.cdist+torch.topk   | Avg Time: 0.014090 s | Peak Memory: 2249.88 MB (min: 2249.88, max: 2249.88)
        pointops.knn_query       | Avg Time: 0.007368 s | Peak Memory:  43.62 MB (min:  43.62, max:  43.62)

    Args:
        xyz_ref (Tensor): (B, N, 3)
        xyz_query (Tensor): (B, M, 3)

    Returns:
        Tuple[Tensor, Tensor]:
            - dist (Tensor): (B, M, k)
            - idx (Tensor): (B, M, k) int32 — indices into dimension N
    """
    # Fallback if tensors are not on CUDA
    if not xyz_ref.is_cuda:
        return _knn_torch(k, xyz_ref, xyz_query)

    from pointops import knn_query
    B, N, _ = xyz_ref.shape
    _, M, _ = xyz_query.shape
    orig_dtype = xyz_ref.dtype

    xyz_ref_flat = xyz_ref.contiguous().view(B * N, 3).to(torch.float32)
    xyz_query_flat = xyz_query.contiguous().view(B * M, 3).to(torch.float32)

    offset = torch.arange(1, B + 1, device=xyz_ref.device) * N
    new_offset = torch.arange(1, B + 1, device=xyz_query.device) * M
    idx, dists = knn_query(k, xyz_ref_flat, offset, xyz_query_flat, new_offset)

    # Remap global indices to local per-batch
    idx = idx.view(B, M, k)
    idx = idx - (torch.arange(B, device=idx.device).view(B, 1, 1) * N)
    dists = dists.view(B, M, k).to(orig_dtype)

    return dists, idx


def _knn_torch(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
    """Fallback KNN using torch.cdist + topk."""
    dists = torch.cdist(xyz_query, xyz_ref, p=2)  # (B, M, N)
    sorted_dists, indices = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
    return sorted_dists, indices


# Select backend once (safe if pointops missing).
try:
    import importlib

    importlib.import_module("pointops")
    knn = _knn_pointops
except Exception:
    logging.warning("pointops not found, falling back to slower KNN implementation.")
    knn = _knn_torch


class MVTracker(nn.Module):
    def __init__(
            self,
            sliding_window_len=12,
            stride=4,
            normalize_scene_in_fwd_pass=False,
            fmaps_dim=128,
            add_space_attn=True,
            num_heads=6,
            hidden_size=384,
            space_depth=6,
            time_depth=6,
            num_virtual_tracks=64,
            use_flash_attention=True,
            corr_n_groups=1,
            corr_n_levels=4,
            corr_neighbors=16,
            corr_add_neighbor_offset=True,
            corr_add_neighbor_xyz=False,
            corr_filter_invalid_depth=False,
    ):
        super().__init__()

        self.S = sliding_window_len
        self.stride = stride
        self.normalize_scene_in_fwd_pass = normalize_scene_in_fwd_pass
        self.latent_dim = fmaps_dim
        self.flow_embed_dim = 64
        self.b_latent_dim = self.latent_dim // 3
        self.corr_n_groups = corr_n_groups
        self.corr_n_levels = corr_n_levels
        self.corr_neighbors = corr_neighbors
        self.corr_pos_emb_size = 0
        self.corr_add_neighbor_offset = corr_add_neighbor_offset
        self.corr_add_neighbor_xyz = corr_add_neighbor_xyz
        self.corr_filter_invalid_depth = corr_filter_invalid_depth
        self.add_space_attn = add_space_attn
        self.updateformer_input_dim = (
            # The positional encoding of the 3D flow from t=i to t=0
                + (self.flow_embed_dim + 1) * 3

                # The correlation features (LRR) for the three planes (xy, yz, xz), concatenated
                + self.corr_neighbors * self.corr_n_levels
                * (self.corr_n_groups
                   + 3 * self.corr_add_neighbor_offset
                   + 3 * self.corr_add_neighbor_xyz
                   + self.corr_pos_emb_size)

                # The features of the tracked points, one for each of the three planes
                + self.latent_dim

                # The visibility mask
                + 1

                # The whether-the-point-is-tracked mask
                + 1
        )

        # Feature encoder
        self.fnet = BasicEncoder(
            input_dim=3,
            output_dim=self.latent_dim,
            norm_fn="instance",
            dropout=0,
            stride=self.stride,
            Embed3D=False,
        )

        # Transformer for iterative updates
        self.updateformer_hidden_size = hidden_size
        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.updateformer_input_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=3 + self.latent_dim,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
            attn_class=FlashAttention if use_flash_attention else Attention,
            linear_layer_for_vis_conf=False,
        )

        # Feature update + visibility
        self.ffeats_norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeats_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())
        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        self.stats_pyramid = None
        self.stats_depth = None

    def fnet_fwd(self, rgbs_normalized, image_features=None):
        b, v, t, _, h, w = rgbs_normalized.shape
        rgbs_normalized = rgbs_normalized.reshape(-1, 3, h, w)
        return self.fnet(rgbs_normalized)

    def init_stats(self):
        self.stats_pyramid = defaultdict(list)
        self.stats_depth = []

    def consume_stats(self):
        # Per-pyramid-level summary of neighbor distances
        level_to_norms = defaultdict(list)
        for (level, _), norm_lists in self.stats_pyramid.items():
            level_to_norms[level].extend(norm_lists)
        level_summary = []
        for level, norm_lists in level_to_norms.items():
            norms = np.concatenate(norm_lists).astype(float)
            stats = pd.Series(norms).describe(percentiles=[.25, .5, .75])
            level_summary.append({
                "level": level,
                "count": int(stats["count"]),
                "mean": round(float(stats["mean"] * 100), 1),
                "std": round(float(stats["std"] * 100), 1),
                "min": round(float(stats["min"] * 100), 1),
                "25%": round(float(stats["25%"] * 100), 1),
                "50%": round(float(stats["50%"] * 100), 1),
                "75%": round(float(stats["75%"] * 100), 1),
                "max": round(float(stats["max"] * 100), 1),
            })
        df_level_summary = pd.DataFrame(level_summary).sort_values("level")
        logging.info(f"Neighbor distances across pyramid levels:\n{df_level_summary}")

        # Per-pyramid-level and per-iteration summary of neighbor distances
        summary = []
        for (level, it), norm_lists in self.stats_pyramid.items():
            norms = np.concatenate(norm_lists).astype(float)
            stats = pd.Series(norms).describe(percentiles=[.25, .5, .75])
            summary.append({
                "level": level,
                "iteration": it,
                "count": int(stats["count"]),
                "mean": round(float(stats["mean"] * 100), 1),
                "std": round(float(stats["std"] * 100), 1),
                "min": round(float(stats["min"] * 100), 1),
                "25%": round(float(stats["25%"] * 100), 1),
                "50%": round(float(stats["50%"] * 100), 1),
                "75%": round(float(stats["75%"] * 100), 1),
                "max": round(float(stats["max"] * 100), 1),
            })
        df_summary = pd.DataFrame(summary).sort_values(["level", "iteration"])
        logging.info(f"Neighbor distances across pyramid levels and iterations (in cm):\n{(df_summary)}")

        # Valid vs invalid depth stats
        depth_stats = pd.Series(self.stats_depth).describe(percentiles=[.25, .5, .75]).astype(float).round(1)
        logging.info(f"Depth stats (valid vs invalid):\n{depth_stats}")

        self.stats_pyramid = None
        self.stats_depth = None

    def forward_iteration(
            self,
            fmaps,
            depths,
            intrs,
            extrs,
            coords_init,
            vis_init,
            track_mask,
            iters=4,
            feat_init=None,
            save_debug_logs=False,
            debug_logs_path="",
            debug_logs_prefix="",
            debug_logs_window_idx=None,
            save_rerun_logs: bool = False,
            rerun_fmap_coloring_fn: Optional[Callable] = None,
    ):
        B, V, S, D, H, W = fmaps.shape
        N = coords_init.shape[2]
        device = fmaps.device
        if coords_init.shape[1] < S:
            coords = torch.cat([coords_init, coords_init[:, -1].repeat(1, S - coords_init.shape[1], 1, 1)], dim=1)
            vis_init = torch.cat([vis_init, vis_init[:, -1].repeat(1, S - vis_init.shape[1], 1, 1)], dim=1)
        else:
            coords = coords_init.clone()
        if track_mask.shape[1] < S:
            track_mask = torch.cat([
                track_mask,
                torch.zeros_like(track_mask[:, 0]).repeat(1, S - track_mask.shape[1], 1, 1),
            ], dim=1)
        assert B == 1
        assert D == self.latent_dim
        assert fmaps.shape == (B, V, S, D, H, W)
        assert depths.shape == (B, V, S, 1, H, W)
        assert intrs.shape == (B, V, S, 3, 3)
        assert extrs.shape == (B, V, S, 3, 4)
        assert coords.shape == (B, S, N, 3)
        assert vis_init.shape == (B, S, N, 1)
        assert track_mask.shape == (B, S, N, 1)
        assert feat_init is None or feat_init.shape == (B, S, N, self.latent_dim)

        assert track_mask.any(1).all(), "All points should be requested to be tracked at least for one frame"

        intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
        extrs_square = torch.eye(4).to(extrs.device)[None].repeat(B, V, S, 1, 1)
        extrs_square[:, :, :, :3, :] = extrs
        extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)
        assert intrs_inv.shape == (B, V, S, 3, 3)
        assert extrs_square.shape == (B, V, S, 4, 4)
        assert extrs_inv.shape == (B, V, S, 4, 4)

        fcorr_fns = {}
        for lvl in range(self.corr_n_levels):
            pc = init_pointcloud_from_rgbd(
                fmaps=fmaps,
                depths=depths,
                intrs=intrs,
                extrs=extrs,
                stride=self.stride,
                level=lvl,
                return_validity_mask=self.corr_filter_invalid_depth or save_rerun_logs,
            )
            if self.corr_filter_invalid_depth or save_rerun_logs:
                pc_xyz, pc_fvec, pc_valid = pc
            else:
                pc_xyz, pc_fvec = pc
                pc_valid = None
            fcorr_fns[lvl] = PointcloudCorrBlock(
                k=self.corr_neighbors,
                groups=self.corr_n_groups,
                xyz=pc_xyz,
                fvec=pc_fvec,
                filter_invalid=self.corr_filter_invalid_depth,
                valid=pc_valid,
                corr_add_neighbor_offset=self.corr_add_neighbor_offset,
                corr_add_neighbor_xyz=self.corr_add_neighbor_xyz,
                rerun_fmap_coloring_fn=rerun_fmap_coloring_fn,
            )

        # Positional/time embeddings (keep shapes identical to before)
        embed_dim = self.updateformer_input_dim
        if embed_dim % 6 != 0:
            embed_dim += 6 - (embed_dim % 6)
        pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, coords[:, 0:1]).float()[:, 0].permute(0, 2, 1)
        if embed_dim > self.updateformer_input_dim:
            pos_embed = pos_embed[:, :self.updateformer_input_dim, :]
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1) / S
        embed_dim = self.updateformer_input_dim
        if embed_dim % 2 != 0:
            embed_dim += 2 - (embed_dim % 2)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        if embed_dim > self.updateformer_input_dim:
            times_embed = times_embed[:, :, :self.updateformer_input_dim]

        coord_predictions = []

        ffeats = feat_init.clone()
        track_mask_and_vis = torch.cat([track_mask, vis_init], dim=3).permute(0, 2, 1, 3).reshape(B * N, S, 2)
        for it in range(iters):
            coords = coords.detach()

            # Sample correlation features around each point
            fcorrs = []
            for lvl in range(self.corr_n_levels):
                fcorr_fn = fcorr_fns[lvl]
                fcorrs_level = (
                    fcorr_fn
                    .corr_sample(
                        targets=ffeats.reshape(B * S, N, self.latent_dim),
                        coords_world_xyz=coords.reshape(B * S, N, 3),
                        save_debug_logs=False,
                        debug_logs_path=debug_logs_path,
                        debug_logs_prefix=debug_logs_prefix + f"__iter_{it}__pyramid_level_{lvl}",
                        save_rerun_logs=save_rerun_logs,
                    )
                    .reshape(B, S, N, -1)
                )
                fcorrs.append(fcorrs_level)
                if self.stats_pyramid is not None:
                    self.stats_pyramid[(lvl, it)] += [
                        np.linalg.norm(fcorrs_level.reshape(-1, 4)[:, 1:].detach().cpu().numpy(), axis=-1)
                    ]
            fcorrs = torch.cat(fcorrs, dim=-1)
            LRR = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)

            # Flow embedding
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 3)
            flows_ = get_3d_embedding(flows_, self.flow_embed_dim, cat_coords=True)

            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            transformer_input = torch.cat([flows_, fcorrs_, ffeats_, track_mask_and_vis], dim=2)
            assert transformer_input.shape[-1] == pos_embed.shape[-1]
            x = transformer_input + pos_embed + times_embed
            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta = self.updateformer(x)
            delta = rearrange(delta, " b n t d -> (b n) t d")

            d_coord = delta[:, :, :3].reshape(B, N, S, 3).permute(0, 2, 1, 3)

            d_feats = delta[:, :, 3:self.latent_dim + 3]
            d_feats = self.ffeats_norm(d_feats.view(-1, self.latent_dim))
            d_feats = self.ffeats_updater(d_feats).view(B, N, S, self.latent_dim).permute(0, 2, 1, 3)

            coords = coords + d_coord
            ffeats = ffeats + d_feats

            if torch.isnan(coords).any():
                logging.error("Got NaN values in coords, perhaps the training exploded")
                import ipdb
                ipdb.set_trace()

            coord_predictions.append(coords.clone())

        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)

        return coord_predictions, vis_e, feat_init

    def forward(
            self,
            rgbs,
            depths,
            query_points,
            intrs,
            extrs,
            iters=4,
            image_features=None,
            is_train=False,
            save_debug_logs=False,
            debug_logs_path="",
            save_rerun_logs: bool = False,
            save_rerun_logs_output_rrd_path: Optional[str] = None,
            previous_state: Optional[dict[str, torch.Tensor]] = None,
            persistent_query_count: Optional[int] = None,
            return_rolling_state: bool = False,
            **kwargs,
    ):
        device = extrs.device
        if save_debug_logs:
            if kwargs:
                logging.info(f"Unused kwargs: {kwargs.keys()}")

        batch_size, num_views, num_frames, _, height, width = rgbs.shape
        _, num_points, _ = query_points.shape
        logging.info(f"FWD pass: {num_views=} {num_frames=} {num_points=} "
                     f"{height=} {width=} {iters=} {num_points=} {rgbs.dtype=}")

        # I made a video tutorial here if it is easier to follow: https://www.youtube.com/watch?v=dQw4w9WgXcQ

        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
        assert query_points.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)

        if persistent_query_count is None:
            persistent_query_count = num_points if (previous_state is not None or return_rolling_state) else 0
        persistent_query_count = int(persistent_query_count)
        if persistent_query_count < 0 or persistent_query_count > num_points:
            raise ValueError(
                f"persistent_query_count must be in [0, {num_points}] but got {persistent_query_count}"
            )

        if save_debug_logs:
            os.makedirs(debug_logs_path, exist_ok=True)

        if save_rerun_logs:
            assert save_rerun_logs_output_rrd_path is not None
            import rerun as rr
            rr.init("3dpt", recording_id="v0.16")
            rr.set_time_seconds("frame", 0)

        if self.stats_depth is not None:
            self.stats_depth += [(depths == 0).float().mean().item() * 100]

        # Scene normalization (optional): Rigid transformation to center first camera and rescale the scene like VGGT
        qp_range_before = np.stack([
            query_points[0, :, 1:].min(0).values.cpu().numpy().round(2),
            query_points[0, :, 1:].max(0).values.cpu().numpy().round(2),
        ])
        if self.normalize_scene_in_fwd_pass:
            assert batch_size == 1, "VGGT normalization assumes batch size 1"
            max_depth = 24
            _d = depths.clone()
            _d[_d < max_depth] = max_depth
            T_scale, T_rot, T_translation = compute_vggt_scene_normalization_transform(
                _d[0], extrs[0].to(_d.device), intrs[0].to(_d.device)
            )
            T_scale_inv = 1 / T_scale
            T_rot_inv = T_rot.transpose(0, 1)
            T_translation_inv = -T_translation @ T_rot_inv

            query_points, extrs = query_points[0], extrs[0]  # Remove batch dimension
            extrs, query_points, _, _ = transform_scene(T, extrs, query_points, None, None)
            query_points, extrs = query_points[None], extrs[None]  # Add batch dimension
        qp_range_after = np.stack([
            query_points[0, :, 1:].min(0).values.cpu().numpy().round(2),
            query_points[0, :, 1:].max(0).values.cpu().numpy().round(2),
        ])
        if save_debug_logs:
            logging.info(f"Query points range before normalization:\n{qp_range_before}")
            logging.info(f"Query points range after normalization: \n{qp_range_after}")

        self.is_train = is_train

        # Unpack the query points
        query_points_t = query_points[:, :, :1].long()
        query_points_xyz_worldspace = query_points[:, :, 1:]

        # Invert intrinsics and extrinsics
        intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
        extrs_square = torch.eye(4).to(extrs.device)[None].repeat(batch_size, num_views, num_frames, 1, 1)
        extrs_square[:, :, :, :3, :] = extrs
        extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)

        # Interpolate the rgbs and depthmaps to the stride of the SpaTracker
        strided_height = height // self.stride
        strided_width = width // self.stride
        half_window = self.S // 2

        # Filter the points that never appear during 1 - T
        assert batch_size == 1, "Batch size > 1 is not supported yet"
        query_points_t = query_points_t.squeeze(0).squeeze(-1)  # BN1 --> N
        ind_array = torch.arange(num_frames, device=query_points.device)
        ind_array = ind_array[None, :, None].repeat(batch_size, 1, num_points)
        track_mask = (ind_array >= query_points_t[None, None, :]).unsqueeze(-1)  # TODO: >= or >?

        # Prepare the initial coordinates and visibility
        coords_init = query_points_xyz_worldspace.unsqueeze(1).repeat(1, self.S, 1, 1)
        vis_init = query_points.new_ones((batch_size, self.S, num_points, 1)) * 10

        # Sort the queries via their first appeared time
        sort_inds = torch.argsort(query_points_t, dim=0, descending=False, stable=True)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        assert torch.allclose(query_points_t, query_points_t[sort_inds][inv_sort_inds])

        query_points_t_ = query_points_t[sort_inds]
        query_points_xyz_worldspace_ = query_points_xyz_worldspace[..., sort_inds, :]
        coords_init_ = coords_init[..., sort_inds, :].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()
        track_mask_ = track_mask[:, :, sort_inds].clone()
        if persistent_query_count > 0:
            expected_persistent = torch.arange(persistent_query_count, device=sort_inds.device, dtype=sort_inds.dtype)
            if not torch.equal(sort_inds[:persistent_query_count], expected_persistent):
                raise ValueError(
                    "Persistent queries must remain the first entries after stable time-sorting. "
                    "Use query timesteps that keep the primary queries ahead of support/grid queries."
                )

        # Delete the unsorted variables (for safety)
        del coords_init, vis_init, query_points_t, query_points, query_points_xyz_worldspace, track_mask

        # Placeholders for the results (for the sorted points)
        traj_e_ = coords_init_.new_zeros((batch_size, num_frames, num_points, 3))
        vis_e_ = coords_init_.new_zeros((batch_size, num_frames, num_points))

        w_idx_start = int(query_points_t_.min().item())
        p_idx_start = 0
        vis_predictions = []
        coord_predictions = []
        p_idx_end_list = []
        fmaps_seq, depths_seq, feat_init, rerun_fmap_coloring_fn = None, None, None, None
        previous_coords_state = None
        previous_vis_state = None
        has_prior_window_predictions = False
        if previous_state is not None:
            required_keys = {"fmaps_seq", "depths_seq", "feat_init", "last_coords", "last_vis"}
            missing_keys = required_keys.difference(previous_state)
            if missing_keys:
                raise ValueError(f"previous_state is missing keys: {sorted(missing_keys)}")
            if w_idx_start != 0:
                raise ValueError("previous_state requires query_points to start at local frame 0")
            if persistent_query_count == 0:
                raise ValueError("previous_state requires persistent_query_count > 0")
            if not torch.all(query_points_t_[:persistent_query_count] == w_idx_start):
                raise ValueError(
                    "Persistent queries must begin at the first local frame when previous_state is provided"
                )

            fmaps_seq = previous_state["fmaps_seq"]
            depths_seq = previous_state["depths_seq"]
            feat_init = previous_state["feat_init"]
            previous_coords_state = previous_state["last_coords"]
            previous_vis_state = previous_state["last_vis"]

            if fmaps_seq.shape != (
                batch_size,
                num_views,
                half_window,
                self.latent_dim,
                strided_height,
                strided_width,
            ):
                raise ValueError(
                    "previous_state['fmaps_seq'] has incompatible shape "
                    f"{tuple(fmaps_seq.shape)} for batch/view/window configuration"
                )
            if depths_seq.shape != (
                batch_size,
                num_views,
                half_window,
                1,
                strided_height,
                strided_width,
            ):
                raise ValueError(
                    "previous_state['depths_seq'] has incompatible shape "
                    f"{tuple(depths_seq.shape)} for batch/view/window configuration"
                )
            if feat_init.shape != (batch_size, self.S, persistent_query_count, self.latent_dim):
                raise ValueError(
                    "previous_state['feat_init'] has incompatible shape "
                    f"{tuple(feat_init.shape)} for the persistent query count"
                )
            if previous_coords_state.shape != (batch_size, half_window, persistent_query_count, 3):
                raise ValueError(
                    "previous_state['last_coords'] has incompatible shape "
                    f"{tuple(previous_coords_state.shape)} for the persistent query count"
                )
            if previous_vis_state.shape != (batch_size, half_window, persistent_query_count, 1):
                raise ValueError(
                    "previous_state['last_vis'] has incompatible shape "
                    f"{tuple(previous_vis_state.shape)} for the persistent query count"
                )
            p_idx_start = persistent_query_count

        last_window_fmaps = None
        last_window_depths = None
        last_window_len = 0
        last_window_coords = None
        last_window_vis_logits = None
        while w_idx_start < num_frames - half_window:
            curr_wind_points = torch.nonzero(query_points_t_ < w_idx_start + self.S)
            assert curr_wind_points.shape[0] > 0
            p_idx_end = curr_wind_points[-1].item() + 1
            p_idx_end_list.append(p_idx_end)

            intrs_seq = intrs[:, :, w_idx_start:w_idx_start + self.S]
            extrs_seq = extrs[:, :, w_idx_start:w_idx_start + self.S]

            # Compute fmaps and interpolated depth on a rolling basis
            # to reduce peak GPU memory consumption, but don't recompute
            # for the overlapping part of a window
            is_first_stateful_window = previous_state is not None and not has_prior_window_predictions
            if is_first_stateful_window:
                new_seq_t0 = w_idx_start + half_window
            elif fmaps_seq is None:
                assert depths_seq is None
                new_seq_t0 = w_idx_start
            else:
                fmaps_seq = fmaps_seq[:, :, half_window:]
                depths_seq = depths_seq[:, :, half_window:]
                new_seq_t0 = w_idx_start + half_window
            new_seq_t1 = min(w_idx_start + self.S, num_frames)

            if new_seq_t1 > new_seq_t0:
                _depths_seq_new = nn.functional.interpolate(
                    input=depths[:, :, new_seq_t0:new_seq_t1].to(device).reshape(-1, 1, height, width),
                    scale_factor=1.0 / self.stride,
                    mode="nearest",
                ).reshape(batch_size, num_views, -1, 1, strided_height, strided_width)
                depths_seq = smart_cat(depths_seq, _depths_seq_new, dim=2)

                _fmaps_seq_new = self.fnet_fwd(
                    (2 * (rgbs[:, :, new_seq_t0:new_seq_t1].to(device) / 255.0) - 1.0),
                    image_features,
                )
                _fmaps_seq_new = nn.functional.interpolate(
                    input=_fmaps_seq_new,
                    size=(strided_height, strided_width),
                    mode="bilinear",
                ).reshape(batch_size, num_views, -1, self.latent_dim, strided_height, strided_width)
                fmaps_seq = smart_cat(fmaps_seq, _fmaps_seq_new, dim=2)
            else:
                _depths_seq_new = depths.new_zeros(
                    batch_size, num_views, 0, 1, strided_height, strided_width
                )
                if fmaps_seq is None:
                    raise ValueError("No feature maps available for the current window")
                _fmaps_seq_new = fmaps_seq.new_zeros(
                    batch_size, num_views, 0, self.latent_dim, strided_height, strided_width
                )

            if save_rerun_logs and rerun_fmap_coloring_fn is None:
                valid_depths_mask = depths_seq.detach().cpu().squeeze(3) > 0
                fvec_flat = fmaps_seq.detach().cpu().permute(0, 1, 2, 4, 5, 3)[valid_depths_mask].numpy()
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=3)
                reducer.fit(fvec_flat)
                fvec_reduced = reducer.transform(fvec_flat)
                reducer_min = fvec_reduced.min(axis=0)
                reducer_max = fvec_reduced.max(axis=0)

                def fvec_to_rgb(fvec):
                    input_shape = fvec.shape
                    assert input_shape[-1] == self.latent_dim
                    fvec_reduced = reducer.transform(fvec.reshape(-1, self.latent_dim))
                    fvec_reduced = np.clip(fvec_reduced, reducer_min[None, :], reducer_max[None, :])
                    fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                    fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                    fvec_reduced_rgb = fvec_reduced_rgb.reshape(input_shape[:-1] + (3,))
                    return fvec_reduced_rgb

                rerun_fmap_coloring_fn = fvec_to_rgb

            S_local = fmaps_seq.shape[2]
            last_window_len = S_local
            last_window_fmaps = fmaps_seq[:, :, :S_local].detach().clone()
            last_window_depths = depths_seq[:, :, :S_local].detach().clone()
            if S_local < self.S:
                diff = self.S - S_local
                fmaps_seq = torch.cat([fmaps_seq, fmaps_seq[:, :, -1:].repeat(1, 1, diff, 1, 1, 1)], 2)
                depths_seq = torch.cat([depths_seq, depths_seq[:, :, -1:].repeat(1, 1, diff, 1, 1, 1)], 2)
                intrs_seq = torch.cat([intrs_seq, intrs_seq[:, :, -1:].repeat(1, 1, diff, 1, 1)], 2)
                extrs_seq = torch.cat([extrs_seq, extrs_seq[:, :, -1:].repeat(1, 1, diff, 1, 1)], 2)

            # Compute the feature vector initialization for the new query points
            if p_idx_end - p_idx_start > 0:
                init_fmaps = _fmaps_seq_new
                init_depths = _depths_seq_new
                init_intrs = intrs[:, :, new_seq_t0:new_seq_t1]
                init_extrs = extrs[:, :, new_seq_t0:new_seq_t1]
                init_t0 = new_seq_t0
                init_t1 = new_seq_t1
                if is_first_stateful_window:
                    init_fmaps = last_window_fmaps
                    init_depths = last_window_depths
                    init_intrs = intrs[:, :, w_idx_start:w_idx_start + S_local]
                    init_extrs = extrs[:, :, w_idx_start:w_idx_start + S_local]
                    init_t0 = w_idx_start
                    init_t1 = w_idx_start + S_local

                rgbd_xyz, rgbd_fvec = init_pointcloud_from_rgbd(
                    fmaps=init_fmaps,
                    depths=init_depths,
                    intrs=init_intrs,
                    extrs=init_extrs,
                    stride=self.stride,
                )

                init_num_frames = init_fmaps.shape[2]
                rgbd_xyz = rgbd_xyz.reshape(batch_size, init_num_frames, num_views, strided_height * strided_width, 3)
                rgbd_fvec = rgbd_fvec.reshape(batch_size, init_num_frames, num_views, strided_height * strided_width,
                                              self.latent_dim)

                _feat_init_new = torch.zeros(batch_size, p_idx_end - p_idx_start, self.latent_dim,
                                             device=init_fmaps.device, dtype=init_fmaps.dtype)
                assert batch_size == 1
                assert (
                    (query_points_t_[p_idx_start:p_idx_end] >= init_t0)
                    & (query_points_t_[p_idx_start:p_idx_end] < init_t1)
                ).all()
                batch_idx = 0
                for t in range(init_t0, init_t1):
                    query_mask = query_points_t_[p_idx_start:p_idx_end] == t
                    if query_mask.sum() == 0:
                        continue
                    query_points_world = query_points_xyz_worldspace_[batch_idx, p_idx_start:p_idx_end][query_mask]

                    rgbd_xyz_current = rgbd_xyz[batch_idx, t - init_t0].reshape(-1, 3)  # Combine views for frame
                    rgbd_fvec_current = rgbd_fvec[batch_idx, t - init_t0].reshape(-1, self.latent_dim)

                    k = 1
                    neighbor_dists, neighbor_indices = knn(k, rgbd_xyz_current[None],
                                                           query_points_world[None])
                    assert k == 1, "If k > 1, the code below should be modified to handle multiple neighbors -- how to combine the features of multiple neighbors?"
                    neighbor_xyz = rgbd_xyz_current[neighbor_indices[0, :, 0]]
                    neighbor_fvec = rgbd_fvec_current[neighbor_indices[0, :, 0]]

                    _feat_init_new[batch_idx, query_mask] = neighbor_fvec

                feat_init = smart_cat(feat_init, _feat_init_new.repeat(1, self.S, 1, 1), dim=2)

            # Update the initial coordinates and visibility for non-first windows
            if has_prior_window_predictions:
                last_coords = coords[-1][:, half_window:].clone()  # Take the predicted coords from the last window
                coords_init_[:, :half_window, :p_idx_start] = last_coords
                coords_init_[:, half_window:, :p_idx_start] = last_coords[:, -1].repeat(1, half_window, 1, 1)

                last_vis = vis[:, half_window:][..., None]
                vis_init_[:, :half_window, :p_idx_start] = last_vis
                vis_init_[:, half_window:, :p_idx_start] = last_vis[:, -1].repeat(1, half_window, 1, 1)
            elif previous_state is not None and p_idx_start > 0:
                if previous_coords_state is None or previous_vis_state is None:
                    raise ValueError("previous_state did not provide the expected carry-over tensors")
                coords_init_[:, :half_window, :p_idx_start] = previous_coords_state
                coords_init_[:, half_window:, :p_idx_start] = previous_coords_state[:, -1].repeat(1, half_window, 1, 1)
                vis_init_[:, :half_window, :p_idx_start] = previous_vis_state
                vis_init_[:, half_window:, :p_idx_start] = previous_vis_state[:, -1].repeat(1, half_window, 1, 1)

            track_mask_current = track_mask_[:, w_idx_start: w_idx_start + self.S, :p_idx_end]
            if S_local < self.S:
                track_mask_current = torch.cat([
                    track_mask_current,
                    track_mask_current[:, -1:].repeat(1, self.S - S_local, 1, 1),
                ], 1)

            coords, vis, _ = self.forward_iteration(
                fmaps=fmaps_seq,
                depths=depths_seq,
                intrs=intrs_seq,
                extrs=extrs_seq,
                coords_init=coords_init_[:, :, :p_idx_end],
                feat_init=feat_init[:, :, :p_idx_end],
                vis_init=vis_init_[:, :, :p_idx_end],
                track_mask=track_mask_current,
                iters=iters,
                save_debug_logs=save_debug_logs,
                debug_logs_path=debug_logs_path,
                debug_logs_prefix=f"__widx-{w_idx_start}_pidx-{p_idx_start}-{p_idx_end}",
                debug_logs_window_idx=w_idx_start,
                save_rerun_logs=save_rerun_logs,
                rerun_fmap_coloring_fn=rerun_fmap_coloring_fn,
            )

            if is_train:
                coord_predictions.append([
                    coord[:, :S_local]
                    if not self.normalize_scene_in_fwd_pass
                    else transform_scene(T_scale_inv, T_rot_inv, T_translation_inv,
                                         None, None, None, coord[:, :S_local][0], None)[2][None]
                    for coord in coords
                ])
                vis_predictions.append(vis[:, :S_local])

            traj_e_[:, w_idx_start:w_idx_start + self.S, :p_idx_end] = coords[-1][:, :S_local]
            vis_e_[:, w_idx_start:w_idx_start + self.S, :p_idx_end] = torch.sigmoid(vis[:, :S_local])
            last_window_coords = coords[-1][:, :S_local].detach().clone()
            last_window_vis_logits = vis[:, :S_local].detach().clone()

            track_mask_[:, : w_idx_start + self.S, :p_idx_end] = 0.0
            w_idx_start = w_idx_start + half_window

            p_idx_start = p_idx_end
            has_prior_window_predictions = True

        if save_debug_logs:
            import gpustat
            torch.cuda.empty_cache()
            logging.info(f"Forward pass GPU usage: {gpustat.new_query()}")

        if save_rerun_logs:
            import rerun as rr
            rr.save(save_rerun_logs_output_rrd_path)
            logging.info(f"Saved Rerun recording to: {os.path.abspath(save_rerun_logs_output_rrd_path)}.")

        traj_e = traj_e_[:, :, inv_sort_inds]
        vis_e = vis_e_[:, :, inv_sort_inds]

        # Un-normalize the scene
        if self.normalize_scene_in_fwd_pass:
            traj_e = transform_scene(T_scale_inv, T_rot_inv, T_translation_inv,
                                     None, None, None, traj_e[0], None)[2][None]

        results = {
            "traj_e": traj_e,
            "feat_init": feat_init,
            "vis_e": vis_e,
        }
        if return_rolling_state:
            if last_window_fmaps is None or last_window_depths is None:
                raise RuntimeError("Unable to export rolling_state because no local window was processed")
            carry_len = min(half_window, last_window_len)
            if carry_len <= 0:
                raise RuntimeError("Unable to export rolling_state because the last window was empty")
            carry_start = last_window_len - carry_len

            feat_init_unsorted = feat_init[:, :, inv_sort_inds] if feat_init is not None else None
            if last_window_coords is None or last_window_vis_logits is None or feat_init_unsorted is None:
                raise RuntimeError("Rolling-state export is missing point-wise tensors from the last window")
            last_coords_unsorted = last_window_coords[:, :, inv_sort_inds]
            last_vis_unsorted = last_window_vis_logits[:, :, inv_sort_inds][..., None]

            results["rolling_state"] = {
                "fmaps_seq": last_window_fmaps[:, :, carry_start:].detach().clone().contiguous(),
                "depths_seq": last_window_depths[:, :, carry_start:].detach().clone().contiguous(),
                "feat_init": feat_init_unsorted[:, :, :persistent_query_count].detach().clone().contiguous(),
                "last_coords": last_coords_unsorted[:, carry_start:, :persistent_query_count].detach().clone().contiguous(),
                "last_vis": last_vis_unsorted[:, carry_start:, :persistent_query_count].detach().clone().contiguous(),
            }
        if self.is_train:
            results["train_data"] = {
                "vis_predictions": vis_predictions,
                "coord_predictions": coord_predictions,
                "attn_predictions": None,
                "p_idx_end_list": p_idx_end_list,
                "sort_inds": sort_inds,
                "Rigid_ln_total": None,
            }
        return results


def compute_vggt_scene_normalization_transform(depths, extrs, intrs):
    V, T, _, H, W = depths.shape
    device = depths.device

    extrs_square = torch.eye(4, device=device)[None, None].repeat(V, T, 1, 1)
    extrs_square[:, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)

    intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    homog = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().reshape(-1, 3)
    homog = homog[None].expand(V, -1, -1).type(depths.dtype)

    cam_points = torch.einsum("vij,vnj->vni", intrs_inv[:, 0], homog) * depths[:, 0].reshape(V, -1, 1)
    cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[..., :1])], dim=-1)
    world_points_h = torch.einsum("vij,vnj->vni", extrs_inv[:, 0], cam_points_h)

    world_points_in_first = torch.einsum("ij,vnj->vni", extrs[0, 0], world_points_h)

    mask = (depths[:, 0] > 0).reshape(V, -1)
    valid_points = world_points_in_first[mask]
    avg_dist = valid_points.norm(dim=1).mean()
    scale = 1.0 / avg_dist

    rot = extrs[0, 0, :3, :3]
    translation = extrs[0, 0, :3, 3] * scale
    return scale, rot, translation


class PointcloudCorrBlock:
    def __init__(
            self,
            k: int,
            groups,
            xyz: torch.Tensor,
            fvec: torch.Tensor,
            corr_add_neighbor_offset: bool,
            corr_add_neighbor_xyz: bool,
            filter_invalid: bool = False,
            valid: Optional[torch.Tensor] = None,
            rerun_fmap_coloring_fn: Optional[Callable] = None,
    ):
        self.B, self.N, self.C = fvec.shape
        assert xyz.shape == (self.B, self.N, 3)
        assert fvec.shape == (self.B, self.N, self.C)
        assert k <= self.N, "k should be less than or equal to N"
        assert groups <= self.C, "number of correlation groups should not be larger than the number of channels"
        assert self.C % groups == 0, "number of channels must be divisible by the number of groups (for convenience)"
        assert not filter_invalid or valid is not None

        self.k = k
        self.groups = groups
        self.xyz = xyz
        self.fvec = fvec
        self.corr_add_neighbor_offset = corr_add_neighbor_offset
        self.corr_add_neighbor_xyz = corr_add_neighbor_xyz
        self.filter_invalid = filter_invalid
        self.valid = valid
        self.rerun_fmap_coloring_fn = rerun_fmap_coloring_fn

    def corr_sample(
            self,
            targets: torch.Tensor,
            coords_world_xyz: torch.Tensor,
            save_debug_logs=False,
            debug_logs_path=".",
            debug_logs_prefix="corr",
            save_rerun_logs=False,
    ):
        # Check inputs
        _, M, _ = targets.shape
        assert targets.shape == (self.B, M, self.C)
        assert coords_world_xyz.shape == (self.B, M, 3)

        # Find neighbors for each of the N target points
        if not self.filter_invalid:
            neighbor_dists, neighbor_indices = knn(self.k, self.xyz, coords_world_xyz)
        else:
            neighbor_dists = []
            neighbor_indices = []
            for xyz_i, valid_i, coords_world_xyz_i in zip(self.xyz, self.valid, coords_world_xyz):
                xyz_i = xyz_i[valid_i]
                neighbor_dists_i, neighbor_indices_i = knn(self.k, xyz_i[None], coords_world_xyz_i[None])
                neighbor_dists.append(neighbor_dists_i)
                neighbor_indices.append(neighbor_indices_i)
            neighbor_dists = torch.cat(neighbor_dists)
            neighbor_indices = torch.cat(neighbor_indices)
        batch_idx = torch.arange(self.B, device=self.xyz.device)[:, None, None]
        neighbor_xyz = self.xyz[batch_idx, neighbor_indices]
        neighbor_fvec = self.fvec[batch_idx, neighbor_indices]

        # Compute the local correlations
        targets_grouped = targets.view(self.B, M, self.groups, -1)
        neighbor_fvec_grouped = neighbor_fvec.view(self.B, M, self.k, self.groups, -1)
        corrs = torch.einsum('BMGc,BMKGc->BMKG', targets_grouped, neighbor_fvec_grouped)
        corrs = corrs / ((self.C / self.groups) ** 0.5)

        output = corrs

        # Append the distance/direction features to the correlation
        neighbor_offset_in_world_xyz = neighbor_xyz - coords_world_xyz[..., None, :]
        if self.corr_add_neighbor_offset:
            output = torch.cat([corrs, neighbor_offset_in_world_xyz], -1)

        # Append the neighbor xyz to the correlation
        if self.corr_add_neighbor_xyz:
            output = torch.cat([output, neighbor_xyz], -1)

        if save_debug_logs:

            from sklearn.decomposition import PCA
            fvec_flat = self.fvec.reshape(-1, self.C).detach().cpu().numpy()
            reducer = PCA(n_components=3)
            reducer.fit(fvec_flat)

            fvec_reduced = reducer.transform(fvec_flat)
            reducer_min = fvec_reduced.min(axis=0)
            reducer_max = fvec_reduced.max(axis=0)

            def fvec_to_rgb(fvec):
                fvec_reduced = reducer.transform(fvec)
                fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                return fvec_reduced_rgb

            for b in [0, self.B - 1]:
                # Save all points
                xyz = self.xyz[b].detach().cpu().numpy()
                xyz_colors = fvec_to_rgb(self.fvec[b].detach().cpu().numpy())
                save_pointcloud_to_ply(os.path.join(debug_logs_path, f"{time_now()}{debug_logs_prefix}_all_b{b}.ply"),
                                       xyz, xyz_colors)

                for n in range(3):
                    neighbors = neighbor_xyz[b, n].detach().cpu().numpy()
                    neighbors_colors = fvec_to_rgb(neighbor_fvec[b, n].detach().cpu().numpy())
                    save_pointcloud_to_ply(
                        os.path.join(debug_logs_path, f"{time_now()}{debug_logs_prefix}_neighbors_b{b}_n{n}.ply"),
                        neighbors, neighbors_colors)

                for n in range(3):
                    neighbors = neighbor_xyz[b, n].detach().cpu().numpy()
                    neighbors_colors = fvec_to_rgb(neighbor_fvec[b, n].detach().cpu().numpy())
                    query_point = coords_world_xyz[b, n].detach().cpu().numpy()
                    query_point_color = fvec_to_rgb(targets[b, n].detach().cpu().numpy().reshape(1, -1))
                    combined_points = np.vstack([query_point, neighbors])
                    combined_colors = np.vstack([query_point_color, neighbors_colors])
                    query_point_index = 0
                    neighbor_indices = np.arange(1, len(neighbors) + 1)
                    edges = np.array([[query_point_index, i] for i in neighbor_indices])
                    save_pointcloud_to_ply(os.path.join(debug_logs_path,
                                                        f"{time_now()}{debug_logs_prefix}_query_b{b}_n{n}_with_edges.ply"),
                                           combined_points, combined_colors, edges=edges)

        # Visualize the results with rerun.io
        if save_rerun_logs:
            import rerun as rr
            import re

            assert self.C > 1
            rerun_fps = 30
            log_feature_maps = True
            log_knn_neighbors = False
            knn_line_coloring = "static"
            knn_neighbors_to_log = 6

            logging.info(f"rerun for {debug_logs_prefix} started")

            ## Mask out target scene area
            # xyz = self.xyz.detach().cpu().numpy()
            # bbox = np.array([[-4, 4], [-3, 3.7], [1.2, 5.2]]) # Softball bbox
            # mask = (
            #         (xyz[..., 0] > bbox[0, 0])
            #         & (xyz[..., 0] < bbox[0, 1])
            #         & (xyz[..., 1] > bbox[1, 0])
            #         & (xyz[..., 1] < bbox[1, 1])
            #         & (xyz[..., 2] > bbox[2, 0])
            #         & (xyz[..., 2] < bbox[2, 1])
            # )
            xyz = self.xyz.detach().cpu().numpy()
            mask = np.ones_like(xyz[..., 0]).astype(bool)
            if self.valid is not None:
                mask = self.valid.detach().cpu().numpy()

            # PCA-based feature coloring
            if self.rerun_fmap_coloring_fn is None:
                fvec_flat = self.fvec.detach().cpu().numpy()[mask]
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=3)
                reducer.fit(fvec_flat)
                fvec_reduced = reducer.transform(fvec_flat)
                reducer_min = fvec_reduced.min(axis=0)
                reducer_max = fvec_reduced.max(axis=0)

                def fvec_to_rgb(fvec):
                    input_shape = fvec.shape
                    assert input_shape[-1] == self.C
                    fvec_reduced = reducer.transform(fvec.reshape(-1, self.C))
                    fvec_reduced = np.clip(fvec_reduced, reducer_min[None, :], reducer_max[None, :])
                    fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                    fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                    fvec_reduced_rgb = fvec_reduced_rgb.reshape(input_shape[:-1] + (3,))
                    return fvec_reduced_rgb

                self.rerun_fmap_coloring_fn = fvec_to_rgb

            fvec_colors = self.rerun_fmap_coloring_fn(self.fvec.detach().cpu().numpy())
            targets_colors = self.rerun_fmap_coloring_fn(targets.detach().cpu().numpy())
            neighbor_fvec_colors = self.rerun_fmap_coloring_fn(neighbor_fvec.detach().cpu().numpy())

            import re
            pattern = r'__widx-(\d+)_pidx-(\d+)-(\d+)__iter_(\d+)__pyramid_level_(\d+)'
            match = re.search(pattern, debug_logs_prefix)
            assert match
            t_start = int(match.group(1))
            pidx_start = int(match.group(2))
            pidx_end = int(match.group(3))
            iteration = int(match.group(4))
            pyramid_level = int(match.group(5))

            # # Log fmaps as images for the pipeline figure
            # import os
            # from PIL import Image
            # png_outdir = os.path.join(debug_logs_path, "feature_maps_pngs_2")
            # os.makedirs(png_outdir, exist_ok=True)
            # if pyramid_level == 0 and iteration == 0:
            #     for b in range(self.B):
            #         t = t_start + b
            #         for v in range(8):
            #             fvec_rgb_uint8 = fvec_colors[b].reshape(8, 96, 128, 3)[v].astype(np.uint8)
            #             fname = f"fmap__view{v:02d}__frame{t:05d}.png"
            #             fpath = os.path.join(png_outdir, fname)
            #             Image.fromarray(fvec_rgb_uint8).save(fpath)

            # Log feature map points
            # if log_feature_maps and pyramid_level in [0, 1, 2, 3] and iteration == 0:
            if log_feature_maps and pyramid_level in [0] and iteration == 0:
                if t_start > 0:
                    bs = range(self.B)
                else:
                    bs = range(self.B // 2, self.B)
                for b in bs:
                    rr.set_time_seconds("frame", (t_start + b) / rerun_fps)
                    rr.log(f"fmaps/pyramid-{pyramid_level}", rr.Points3D(
                        xyz[b][mask[b]],
                        colors=fvec_colors[b][mask[b]],
                        radii=0.042,
                        # radii=-2.53,
                    ))

            # Log neighbors
            if log_knn_neighbors and pyramid_level in [0, 1, 2, 3] and iteration in [0]:
                for b in range(self.B):
                    rr.set_time_seconds("frame", (t_start + b) / rerun_fps)
                    for n in range(min(neighbor_xyz.shape[1], knn_neighbors_to_log)):  # Iterate over queries
                        prefix = f"knn/track-{n:03d}/iter-{iteration}/pyramid-{pyramid_level}"
                        rr.log(f"{prefix}/queries", rr.Points3D(
                            coords_world_xyz[b, n].cpu().numpy(),
                            colors=targets_colors[b, n],
                            radii=0.072,
                            # radii=-9.0,
                        ))

                        rr.log(f"{prefix}/neighbors", rr.Points3D(
                            neighbor_xyz[b, n].cpu().numpy(),
                            colors=neighbor_fvec_colors[b, n],
                            radii=0.054,
                            # radii=-5.0,
                        ))

                        if knn_line_coloring == "correlation":
                            # Compute correlation strength for line coloring
                            corr_strength = corrs[b, n,].squeeze(-1).cpu().numpy()
                            corr_strength_normalized = (corr_strength / corr_strength.max()) * 1.0 + 0.0
                            line_colors = (corr_strength_normalized[:, None] * np.array([9, 208, 239])).astype(int)
                            line_colors = np.hstack([line_colors, np.full((line_colors.shape[0], 1), 204)])  # RGBA 80%

                        elif knn_line_coloring == "static":
                            # Make the lines sun flower yellow (241, 196, 15)
                            line_colors = np.array([241, 196, 15])[None].repeat(self.k, 0).astype(int)

                        # Draw edges between query and its neighbors
                        strips = np.stack([
                            coords_world_xyz[b, n].cpu().numpy()[None].repeat(neighbor_xyz.shape[2], axis=0),
                            neighbor_xyz[b, n].cpu().numpy(),
                        ], axis=-2)
                        rr.log(f"{prefix}/arrows", rr.Arrows3D(
                            origins=strips[:, 0],
                            vectors=strips[:, 1] - strips[:, 0],
                            colors=line_colors,
                            radii=0.016,
                            # radii=-1.2,
                        ))
            logging.info(f"rerun for {debug_logs_prefix} done")
        return output
