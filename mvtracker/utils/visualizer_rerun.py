from typing import Union, Optional, List, Dict, Any

import matplotlib
import numpy as np
import pandas as pd
import rerun as rr
import seaborn as sns
import torch
from matplotlib import pyplot as plt, colors as mcolors, cm as cm
from sklearn.decomposition import PCA


def setup_libs(latex=False):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_style("ticks")
    sns.set_palette("flare")

    if latex:
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
        plt.rc('text', usetex=True)
    plt.rcParams.update({
        'figure.titlesize': '28',
        'axes.titlesize': '22',
        'axes.titlepad': '10',
        'legend.title_fontsize': '16',
        'legend.fontsize': '14',
        'axes.labelsize': '18',
        'xtick.labelsize': '16',
        'ytick.labelsize': '16',
        'figure.dpi': 200,
    })


def _set_time_seconds_compat(timeline: str, seconds: float) -> None:
    # Rerun <=0.21 used set_time_seconds, while newer versions use set_time(..., duration=...).
    if hasattr(rr, "set_time"):
        rr.set_time(timeline, duration=float(seconds))
    else:
        rr.set_time_seconds(timeline, seconds)


def log_pointclouds_to_rerun(
        dataset_name: str,
        datapoint_idx: Union[int, str],
        rgbs: torch.Tensor,
        depths: torch.Tensor,
        intrs: torch.Tensor,
        extrs: torch.Tensor,
        depths_conf: Optional[torch.Tensor] = None,
        conf_thrs: Optional[List[float]] = None,
        log_only_confident_pc: bool = False,
        radii: float = -2.45,
        fps: float = 30.0,
        bbox_crop: Optional[torch.Tensor] = None,  # e.g., np.array([[-4, 4], [-3, 3.7], [1.2, 5.2]])
        sphere_radius_crop: Optional[float] = None,  # e.g., 6.0
        sphere_center_crop: Optional[np.ndarray] = np.array([0, 0, 0]),
        log_rgb_image: bool = False,
        log_depthmap_as_image_v1: bool = False,
        log_depthmap_as_image_v2: bool = False,
        log_camera_frustrum: bool = True,
        log_rgb_pointcloud: bool = True,
        timesteps_to_log: Optional[List[int]] = None,
):
    # Set the up-axis for the world
    # Log coordinate axes for reference
    _set_time_seconds_compat("frame", 0)
    B, V, T, _, H, W = rgbs.shape
    assert rgbs.shape == (B, V, T, 3, H, W)
    assert depths.shape == (B, V, T, 1, H, W)
    assert depths_conf is None or depths_conf.shape == (B, V, T, 1, H, W)
    assert intrs.shape == (B, V, T, 3, 3)
    assert extrs.shape == (B, V, T, 3, 4)
    assert B == 1
    # Compute inverse intrinsics and extrinsics
    intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
    extrs_square = torch.eye(4).to(extrs.device)[None].repeat(B, V, T, 1, 1)
    extrs_square[:, :, :, :3, :] = extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)
    assert intrs_inv.shape == (B, V, T, 3, 3)
    assert extrs_inv.shape == (B, V, T, 4, 4)
    for v in range(V):  # Iterate over views
        for t in range(T):  # Iterate over frames

            if timesteps_to_log is not None and t not in timesteps_to_log:
                continue

            _set_time_seconds_compat("frame", t / fps)

            # Log RGB image
            rgb_image = rgbs[0, v, t].permute(1, 2, 0).cpu().numpy()
            if log_rgb_image:
                rr.log(f"sequence-{datapoint_idx}/{dataset_name}/image/view-{v}/rgb", rr.Image(rgb_image))

            # Log Depth map
            depth_map = depths[0, v, t, 0].cpu().numpy()
            if log_depthmap_as_image_v1:
                rr.log(f"sequence-{datapoint_idx}/{dataset_name}/image/view-{v}/depth",
                       rr.DepthImage(depth_map, point_fill_ratio=0.2))

            # Log Depth map as RGB
            d_min, d_max = depth_map.min(), depth_map.max()
            norm = mcolors.Normalize(vmin=d_min, vmax=d_max)
            turbo_cmap = cm.get_cmap("turbo")  # "viridis", "plasma", etc.
            depth_color_rgba = turbo_cmap(norm(depth_map))
            depth_color_rgb = (depth_color_rgba[..., :3] * 255).astype(np.uint8)
            if log_depthmap_as_image_v2:
                rr.log(f"sequence-{datapoint_idx}/{dataset_name}/image/view-{v}/deptha-as-rgb",
                       rr.Image(depth_color_rgb))

            # Log Camera
            K = intrs[0, v, t].cpu().numpy()
            world_T_cam = np.eye(4)
            world_T_cam[:3, :3] = extrs_inv[0, v, t, :3, :3].cpu().numpy()
            world_T_cam[:3, 3] = extrs_inv[0, v, t, :3, 3].cpu().numpy()
            if log_camera_frustrum:
                rr.log(f"sequence-{datapoint_idx}/{dataset_name}/image/view-{v}",
                       rr.Pinhole(image_from_camera=K, width=W, height=H))
                rr.log(f"sequence-{datapoint_idx}/{dataset_name}/image/view-{v}",
                       rr.Transform3D(translation=world_T_cam[:3, 3], mat3x3=world_T_cam[:3, :3]))

            # Generate and log point cloud colored by RGB values
            # Compute 3D points from depth map
            y, x = np.indices((H, W))
            homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
            depth_values = depth_map.ravel()
            cam_coords = (intrs_inv[0, v, t].cpu().numpy() @ homo_pixel_coords) * depth_values
            cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
            world_coords = (world_T_cam @ cam_coords)[:3].T
            rgb_colors = rgb_image.reshape(-1, 3).astype(np.uint8)

            # Log point clouds
            if log_rgb_pointcloud:
                # Filter out points with zero depth
                valid_mask = depth_values > 0

                # Filter out points outside this bbox
                # bbox_crop = np.array([[-4, 4], [-3, 3.7], [1.2, 5.2]])
                if bbox_crop is not None:
                    bbox_mask = (
                            (world_coords[..., 0] > bbox_crop[0, 0])
                            & (world_coords[..., 0] < bbox_crop[0, 1])
                            & (world_coords[..., 1] > bbox_crop[1, 0])
                            & (world_coords[..., 1] < bbox_crop[1, 1])
                            & (world_coords[..., 2] > bbox_crop[2, 0])
                            & (world_coords[..., 2] < bbox_crop[2, 1])
                    )
                    valid_mask = valid_mask & bbox_mask

                # Lightweight Kubric and DexYCB
                if sphere_radius_crop is not None:
                    assert sphere_center_crop is not None
                    sphere_mask = ((world_coords - sphere_center_crop) ** 2).sum(-1) < sphere_radius_crop ** 2
                    valid_mask = valid_mask & sphere_mask

                # Filter out points with confidence below threshold
                pc_name__mask__tuples = []
                if not (log_only_confident_pc and depths_conf is not None):
                    pc_name__mask__tuples += [("point_cloud", valid_mask)]
                if depths_conf is not None:
                    confs = depths_conf[0, v, t, 0].cpu().numpy()
                    assert conf_thrs is not None
                    for thr in conf_thrs:
                        name = f"point_cloud__conf-{thr}"
                        mask = valid_mask & (confs.ravel() > thr)
                        if (valid_mask == mask).all():
                            continue
                        pc_name__mask__tuples += [(name, mask)]
                for pc_name, mask in pc_name__mask__tuples:
                    rr.log(f"sequence-{datapoint_idx}/{dataset_name}/{pc_name}/view-{v}",
                           rr.Points3D(world_coords[mask], colors=rgb_colors[mask], radii=radii))


def _log_tracks_to_rerun(
        tracks: np.ndarray,
        visibles: np.ndarray,
        query_timestep: np.ndarray,
        colors: np.ndarray,
        track_names=None,
        fps=30.0,

        entity_format_str="{}",

        log_points=True,
        points_radii=-3.6,

        log_line_strips=True,
        max_strip_length_past=10,
        max_strip_length_future=0,
        strips_radii=-1.8,

        log_error_lines=False,
        error_lines_radii=0.0042,
        error_lines_color=[1., 0., 0.],
        gt_for_error_lines=None,
) -> None:
    """
    Log tracks to Rerun.

    Parameters:
        tracks: Shape (T, N, 3), the 3D trajectories of points.
        visibles: Shape (T, N), boolean visibility mask for each point at each timestep.
        query_timestep: Shape (T, N), the frame index after which the tracks start.
        colors: Shape (N, 4), RGBA colors for each point.
    """
    T, N, _ = tracks.shape
    assert tracks.shape == (T, N, 3)
    assert visibles.shape == (T, N)
    assert query_timestep.shape == (N,)
    assert query_timestep.min() >= 0
    assert query_timestep.max() < T
    assert colors.shape == (N, 4)

    for n in range(N):
        track_name = track_names[n] if track_names is not None else f"track-{n}"
        rr.log(entity_format_str.format(track_name), rr.Clear(recursive=True))
        for t in range(query_timestep[n], T):
            # if t not in [0] + [T * (x + 1) // 3 - 1 for x in range(3)]:
            # if t not in [T - 1]:
            #     continue
            _set_time_seconds_compat("frame", t / fps)

            # Log the point (special handling for invisible points)
            if log_points:
                rr.log(
                    entity_format_str.format(f"{track_name}/point"),
                    rr.Points3D(
                        positions=[tracks[t, n]],
                        colors=[colors[n, :3]] if visibles[t, n] else [colors[n, :3] * 0.7],
                        radii=points_radii,
                    ),
                )

            # Log line segments for visible tracks
            if log_line_strips and t > query_timestep[n]:
                strip_t_start = max(t - max_strip_length_past, query_timestep[n].item())
                strip_t_end = min(t + max_strip_length_future, T - 1)

                strips = np.stack([
                    tracks[strip_t_start:strip_t_end, n],
                    tracks[strip_t_start + 1:strip_t_end + 1, n],
                ], axis=-2)
                strips_visibility = visibles[strip_t_start + 1:strip_t_end + 1, n]
                strips_colors = np.where(
                    strips_visibility[:, None],
                    colors[None, n, :3],
                    colors[None, n, :3] * 0.7,
                )

                rr.log(
                    entity_format_str.format(f"{track_name}/line"),
                    rr.LineStrips3D(strips=strips, colors=strips_colors, radii=strips_radii),
                )

            if log_error_lines:
                assert gt_for_error_lines is not None
                strips = np.stack([
                    tracks[t, n],
                    gt_for_error_lines[t, n],
                ], axis=-2)
                rr.log(
                    entity_format_str.format(f"{track_name}/error"),
                    rr.LineStrips3D(strips=strips, colors=error_lines_color, radii=error_lines_radii),
                )


def _log_tracks_to_rerun_lightweight(
        tracks: np.ndarray,
        visibles: np.ndarray,
        query_timestep: np.ndarray,
        colors: np.ndarray,
        track_names=None,
        fps=30.0,

        entity_format_str="{}",

        log_points=True,
        points_radii=0.01,

        log_line_strips=True,
        max_strip_length_past=24,
        max_strip_length_future=0,
        strips_radii=0.0042,

        log_error_lines=False,
        error_lines_radii=0.0010,
        error_lines_color=[1., 0., 0.],
        gt_for_error_lines=None,
) -> None:
    """
    Log tracks to Rerun.

    Parameters:
        tracks: Shape (T, N, 3), the 3D trajectories of points.
        visibles: Shape (T, N), boolean visibility mask for each point at each timestep.
        query_timestep: Shape (T, N), the frame index after which the tracks start.
        colors: Shape (N, 4), RGBA colors for each point.
    """
    T, N, _ = tracks.shape
    assert tracks.shape == (T, N, 3)
    assert visibles.shape == (T, N)
    assert query_timestep.shape == (N,)
    assert query_timestep.min() >= 0
    assert query_timestep.max() < T
    assert colors.shape == (N, 4)

    for t in range(T):
        _set_time_seconds_compat("frame", t / fps)
        points_list, points_colors = [], []
        strips_list, strips_colors_list = [], []
        errors_list = []
        for n in range(N):
            if t > query_timestep[n]:
                strip_t_start = max(t - max_strip_length_past, query_timestep[n].item())
                strip_t_end = min(t + max_strip_length_future, T - 1)

                strips = np.stack([
                    tracks[strip_t_start:strip_t_end, n],
                    tracks[strip_t_start + 1:strip_t_end + 1, n],
                ], axis=-2)
                strips_visibility = visibles[strip_t_start + 1:strip_t_end + 1, n]
                strips_colors = np.where(
                    strips_visibility[:, None],
                    colors[None, n, :3],
                    colors[None, n, :3] * 0.7,
                )
                if log_line_strips:
                    strips_list.append(strips)
                    strips_colors_list.append(strips_colors)

                for t_ in range(strip_t_start, strip_t_end + 1):
                    if log_points:
                        points_list += [tracks[t_, n]]
                        points_colors += [colors[n, :3]] if visibles[t_, n] else [colors[n, :3] * 0.7]

                    if log_error_lines:
                        assert gt_for_error_lines is not None
                        error_lines = np.stack([
                            tracks[t_, n],
                            gt_for_error_lines[t_, n],
                        ], axis=-2)
                        errors_list.append(error_lines)

        if log_points and len(points_list) > 0:
            rr.log(
                entity_format_str.format(f"points"),
                rr.Points3D(
                    positions=points_list,
                    colors=points_colors,
                    radii=points_radii,
                ),
            )
        if log_line_strips and len(strips_list) > 0:
            rr.log(
                entity_format_str.format(f"trajectories"),
                rr.LineStrips3D(
                    strips=np.concatenate(strips_list, axis=0),
                    colors=np.concatenate(strips_colors_list, axis=0),
                    radii=strips_radii,
                ),
            )
        if log_error_lines and len(errors_list) > 0:
            rr.log(
                entity_format_str.format(f"errors"),
                rr.LineStrips3D(
                    strips=np.stack(errors_list),
                    colors=error_lines_color,
                    radii=error_lines_radii,
                ),
            )


def log_tracks_to_rerun(
        dataset_name: str,
        datapoint_idx: Union[int, str],
        predictor_name: str,
        gt_trajectories_3d_worldspace: Optional[torch.Tensor],
        gt_visibilities_any_view: Optional[torch.Tensor],
        query_points_3d: torch.Tensor,
        pred_trajectories: torch.Tensor,
        pred_visibilities: torch.Tensor,
        per_track_results: Optional[Dict[str, Any]] = None,
        radii_scale: float = 1.0,
        fps: float = 30.0,
        sphere_radius_crop: Optional[float] = None,  # e.g., 6.0
        sphere_center_crop: Optional[np.ndarray] = np.array([0, 0, 0]),
        log_per_interval_results: bool = False,
        max_tracks_to_log: Optional[int] = None,
        track_batch_size: int = 100,
        method_id: Optional[int] = None,
        color_per_method_id: Optional[Dict[int, tuple]] = None,  # { 0: (46, 204, 113), ... }
        memory_lightweight_logging: bool = True,
):
    # Prepare track data
    gt_tracks = gt_trajectories_3d_worldspace[0].cpu().numpy() if gt_trajectories_3d_worldspace is not None else None
    gt_vis = gt_visibilities_any_view[0].cpu().numpy() if gt_visibilities_any_view is not None else None
    pred_tracks = pred_trajectories[0].cpu().numpy()
    pred_vis = pred_visibilities[0].cpu().numpy()
    query_timestep = query_points_3d[0, :, 0].cpu().numpy().astype(int)
    T, N, _ = pred_tracks.shape
    assert gt_tracks is None or gt_tracks.shape == (T, N, 3)
    assert gt_vis is None or gt_vis.shape == (T, N)
    assert pred_tracks.shape == (T, N, 3)
    assert pred_vis.shape == (T, N)
    assert query_timestep.shape == (N,)

    if sphere_radius_crop is not None:
        pred_tracks = pred_tracks.copy()
        assert sphere_center_crop is not None
        dist = np.linalg.norm(pred_tracks - sphere_center_crop, axis=-1, keepdims=True)
        mask = dist > sphere_radius_crop
        pred_tracks[mask[..., 0]] = (
                sphere_center_crop + sphere_radius_crop *
                (pred_tracks[mask[..., 0]] - sphere_center_crop) /
                dist[mask][..., None]
        )
        if gt_tracks is not None:
            gt_tracks = gt_tracks.copy()
            assert sphere_center_crop is not None
            dist = np.linalg.norm(gt_tracks - sphere_center_crop, axis=-1, keepdims=True)
            mask = dist > sphere_radius_crop
            gt_tracks[mask[..., 0]] = (
                    sphere_center_crop + sphere_radius_crop *
                    (gt_tracks[mask[..., 0]] - sphere_center_crop) /
                    dist[mask][..., None]
            )

    # Last timestamp determines track color (unless method_id is specified)
    final_xyz = gt_tracks[-1] if gt_tracks is not None else pred_tracks[-1]  # (N, 3)
    pca = PCA(n_components=1).fit_transform(final_xyz)  # Apply PCA to spread values across 1D axis
    pca_normalized = (pca - pca.min()) / (pca.max() - pca.min() + 1e-8)  # Normalize to [0, 1]
    cmap = matplotlib.colormaps["gist_rainbow"]
    colors = cmap(pca_normalized[:, 0])  # Map to colormap
    assert colors.shape == (N, 4)

    # If method_id is specified, use fixed colors
    # Fixed color mapping per method
    if color_per_method_id is None:
        color_per_method_id = {
            0: (46, 204, 113),
            1: (52, 152, 219),
            2: (241, 196, 15),
            3: (155, 89, 182),
            4: (230, 126, 34),
            5: (26, 188, 156),
        }
    if method_id is not None:
        assert method_id in color_per_method_id
        base_rgb = np.array(color_per_method_id[method_id]) / 255.0
        colors = np.tile(np.append(base_rgb, 1.0), (N, 1))

    assert colors.shape == (N, 4)

    # Log the tracks
    common_kwargs = {
        "points_radii": -3.6 * radii_scale,
        "strips_radii": -1.8 * radii_scale,
        "error_lines_radii": 0.0042 * radii_scale,
        "fps": fps,
    }
    if max_tracks_to_log:
        N = min(N, max_tracks_to_log)
    for tracks_batch_start in range(0, N, track_batch_size):
        tracks_batch_end = min(tracks_batch_start + track_batch_size, N)
        entity_format_strs = []
        entity_format_strs += [
            f"sequence-{datapoint_idx}/tracks/{{track_name}}/{tracks_batch_start:02d}-{tracks_batch_end:02d}/{{{{}}}}"
        ]
        if not memory_lightweight_logging:
            entity_format_strs += [
                f"sequence-{datapoint_idx}/tracks/all/{tracks_batch_start:02d}-{tracks_batch_end:02d}/{{{{}}}}/{{track_name}}"
            ]
        for entity_format_str in entity_format_strs:
            log_tracks_fn = _log_tracks_to_rerun if not memory_lightweight_logging else _log_tracks_to_rerun_lightweight
            # Log the GT tracks
            if gt_tracks is not None and (method_id is None or method_id == 0):
                log_tracks_fn(
                    tracks=gt_tracks[:, tracks_batch_start:tracks_batch_end],
                    visibles=gt_vis[:, tracks_batch_start:tracks_batch_end],
                    query_timestep=query_timestep[tracks_batch_start:tracks_batch_end],
                    colors=colors[tracks_batch_start:tracks_batch_end] * 0 + np.array([1, 1, 1, 1]),
                    track_names=[f"track-{i:02d}" for i in range(tracks_batch_start, tracks_batch_end)],

                    entity_format_str=entity_format_str.format(track_name=f"gt"),

                    **common_kwargs,
                )
            # Log the predicted tracks
            log_tracks_fn(
                tracks=pred_tracks[:, tracks_batch_start:tracks_batch_end],
                visibles=pred_vis[:, tracks_batch_start:tracks_batch_end],
                query_timestep=query_timestep[tracks_batch_start:tracks_batch_end],
                colors=colors[tracks_batch_start:tracks_batch_end],
                track_names=[f"track-{i:02d}" for i in range(tracks_batch_start, tracks_batch_end)],

                entity_format_str=entity_format_str.format(track_name=f"pred--{predictor_name}"),

                log_error_lines=gt_tracks is not None,
                gt_for_error_lines=gt_tracks[:, tracks_batch_start:tracks_batch_end] if gt_tracks is not None else None,

                **common_kwargs,
            )

    if log_per_interval_results and per_track_results is not None:
        intervals = [(i / 10 * 100, (i + 1) / 10 * 100) for i in range(10)]  # Intervals for 0-10%, ..., 90-100%
        intervals += [(0, 33), (33, 66), (66, 100)]  # Intervals for lower, middle, upper third
    else:
        intervals = []
    for lower, upper in intervals:
        for point_type in ["dynamic", "very_dynamic", "static", "any"]:
            if f"all_{point_type}" not in per_track_results:
                continue
            if lower == 0:  # Special case to include 0
                track_indices = per_track_results[f"all_{point_type}"].indices[
                    (per_track_results[f"all_{point_type}"].average_pts_within_thresh_per_track >= lower) &
                    (per_track_results[f"all_{point_type}"].average_pts_within_thresh_per_track <= upper)
                    ]
            else:
                track_indices = per_track_results[f"all_{point_type}"].indices[
                    (per_track_results[f"all_{point_type}"].average_pts_within_thresh_per_track > lower) &
                    (per_track_results[f"all_{point_type}"].average_pts_within_thresh_per_track <= upper)
                    ]
            if len(track_indices) == 0:
                continue
            entity_format_str = f"sequence-{datapoint_idx}/tracks/location-accuracy-for-{point_type}/{int(lower)}-{int(upper)}-percent-{{track_name}}/{{{{}}}}"
            # Log the GT tracks
            _log_tracks_to_rerun(
                tracks=gt_tracks[:, track_indices],
                visibles=gt_vis[:, track_indices],
                query_timestep=query_timestep[track_indices],
                colors=colors[track_indices] * 0 + np.array([1, 1, 1, 1]),
                track_names=[f"track-{i:02d}" for i in track_indices],
                entity_format_str=entity_format_str.format(track_name=f"gt"),
                **common_kwargs,
            )
            # Log the predicted tracks
            _log_tracks_to_rerun(
                tracks=pred_tracks[:, track_indices],
                visibles=pred_vis[:, track_indices],
                query_timestep=query_timestep[track_indices],
                colors=colors[track_indices],
                track_names=[f"track-{i:02d}" for i in track_indices],
                entity_format_str=entity_format_str.format(track_name=f"pred-{dataset_name}"),
                log_error_lines=True,
                gt_for_error_lines=gt_tracks[:, track_indices],
                **common_kwargs,
            )
