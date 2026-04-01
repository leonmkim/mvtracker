import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mvtracker.models.core.model_utils import bilinear_sample2d, get_points_on_a_grid
from mvtracker.models.core.model_utils import world_space_to_pixel_xy_and_camera_z
from mvtracker.models.core.mvtracker.mvtracker import save_pointcloud_to_ply
from mvtracker.utils.basic import to_homogeneous, from_homogeneous, time_now
from mvtracker.utils.visualizer_mp4 import MultiViewVisualizer


class EvaluationPredictor(torch.nn.Module):
    def __init__(
            self,
            multiview_model: torch.nn.Module,
            interp_shape: Optional[Tuple[int, int]] = (384, 512),
            visibility_threshold=0.5,
            grid_size: int = 5,
            n_grids_per_view: int = 1,
            local_grid_size: int = 8,
            local_extent: int = 50,
            single_point: bool = False,
            sift_size: int = 0,
            num_uniformly_sampled_pts: int = 0,
            n_iters: int = 6,
    ) -> None:
        super(EvaluationPredictor, self).__init__()
        self.model = multiview_model
        self.interp_shape = interp_shape
        self.visibility_threshold = visibility_threshold
        self.grid_size = grid_size
        self.n_grids_per_view = n_grids_per_view
        self.local_grid_size = local_grid_size
        self.local_extent = local_extent
        self.single_point = single_point
        self.sift_size = sift_size
        self.num_uniformly_sampled_pts = num_uniformly_sampled_pts
        self.n_iters = n_iters

        self.model.eval()

    def forward(
            self,
            rgbs,
            depths,
            query_points_3d,
            intrs,
            extrs,
            save_debug_logs=False,
            debug_logs_path="",
            query_points_view=None,
            **kwargs,
    ):
        previous_state = kwargs.pop("previous_state", None)
        return_rolling_state = bool(kwargs.pop("return_rolling_state", False))
        batch_size, num_views, num_frames, _, height_raw, width_raw = rgbs.shape
        _, num_points, _ = query_points_3d.shape

        assert rgbs.shape == (batch_size, num_views, num_frames, 3, height_raw, width_raw)
        assert depths.shape == (batch_size, num_views, num_frames, 1, height_raw, width_raw)
        assert query_points_3d.shape == (batch_size, num_points, 4)
        assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
        assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)

        if batch_size != 1:
            raise NotImplementedError

        if self.single_point and (previous_state is not None or return_rolling_state):
            raise ValueError(
                "previous_state / return_rolling_state are not supported when "
                "EvaluationPredictor.single_point=True"
            )

        # Interpolate the inputs to the desired resolution, if needed
        if self.interp_shape is None:
            height, width = height_raw, width_raw
        else:
            height, width = self.interp_shape
            rgbs = rgbs.reshape(-1, 3, height_raw, width_raw)
            rgbs = F.interpolate(rgbs, (height, width), mode="nearest")
            rgbs = rgbs.reshape(batch_size, num_views, num_frames, 3, height, width)
            depths = depths.reshape(-1, 1, height_raw, width_raw)
            depths = F.interpolate(depths, (height, width), mode="nearest")
            depths = depths.reshape(batch_size, num_views, num_frames, 1, height, width)
            intrs_resize_transform = torch.tensor([
                [width / width_raw, 0, 0],
                [0, height / height_raw, 0],
                [0, 0, 1],
            ], device=intrs.device, dtype=intrs.dtype)
            intrs = torch.einsum("ij,BVTjk->BVTik", intrs_resize_transform, intrs)

        # Unpack the query points
        query_points_t = query_points_3d[:, :, :1].long()
        query_points_xyz_worldspace = query_points_3d[:, :, 1:]

        # Invert intrinsics and extrinsics
        intrs_inv = torch.inverse(intrs.float()).type(intrs.dtype)
        extrs_square = torch.eye(4).to(extrs.device)[None].repeat(batch_size, num_views, num_frames, 1, 1)
        extrs_square[:, :, :, :3, :] = extrs
        extrs_inv = torch.inverse(extrs_square.float()).type(extrs.dtype)

        support_points = torch.zeros((batch_size, 0, 4), device=rgbs.device)

        grid_points = []
        if self.grid_size > 0:
            pixel_xy = get_points_on_a_grid(self.grid_size, (height, width), device=rgbs.device)
            pixel_xy_homo = to_homogeneous(pixel_xy)
            for t in range(0, num_frames, max(1, num_frames // self.n_grids_per_view)):
                for view_idx in range(num_views):
                    camera_z = bilinear_sample2d(
                        depths[0, view_idx, t][None],
                        pixel_xy[..., 0],
                        pixel_xy[..., 1],
                    ).permute(0, 2, 1)
                    camera_xyz = torch.einsum('Bij,BNj->BNi', intrs_inv[:, view_idx, t, :, :], pixel_xy_homo)
                    camera_xyz = camera_xyz * camera_z
                    camera_xyz_homo = to_homogeneous(camera_xyz)
                    world_xyz_homo = torch.einsum('Bij,BNj->BNi', extrs_inv[:, view_idx, t, :, :], camera_xyz_homo)
                    world_xyz = from_homogeneous(world_xyz_homo)
                    grid_points_i = torch.cat([torch.ones_like(world_xyz[:, :, :1]) * t, world_xyz], dim=2)
                    grid_points.append(grid_points_i)
            grid_points = torch.cat(grid_points, dim=1)
            support_points = torch.concat([support_points, grid_points], dim=1)

            if save_debug_logs:
                os.makedirs(debug_logs_path, exist_ok=True)
                save_pointcloud_to_ply(
                    filename=os.path.join(debug_logs_path, time_now() + "__predictor__query_points.ply"),
                    points=query_points_xyz_worldspace[0].cpu().numpy(),
                    colors=np.ones_like(query_points_xyz_worldspace[0].cpu().numpy(), dtype=int) * np.array(
                        [255, 30, 60]),
                )
                save_pointcloud_to_ply(
                    filename=os.path.join(debug_logs_path, time_now() + "__predictor__support_grid_points.ply"),
                    points=grid_points[0, :, 1:].cpu().numpy(),
                    colors=np.ones_like(grid_points[0, :, 1:].cpu().numpy(), dtype=int) * np.array([45, 255, 60]),
                )

        sift_points = []
        if self.sift_size > 0:
            raise NotImplementedError
            # xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
            # if xy.shape[1] == sift_size:
            #     queries = torch.cat([queries, xy], dim=1)  #
            # else:
            #     sift_size = 0
            sift_points = torch.cat(sift_points, dim=1)
            support_points = torch.concat([support_points, sift_points], dim=1)

        support_uniform_pts = []
        if self.num_uniformly_sampled_pts > 0:
            sampled_pts = get_uniformly_sampled_pts(
                self.num_uniformly_sampled_pts,
                num_frames,
                (height, width),
                device=rgbs.device,
            )[0]  # shape: (N, 3) where each row is (t, y, x)

            t_samples = sampled_pts[:, 0].long()
            y_samples = sampled_pts[:, 1].float()
            x_samples = sampled_pts[:, 2].float()

            pixel_xy = torch.stack([x_samples, y_samples], dim=-1)[None]  # (1, N, 2)
            pixel_xy_homo = to_homogeneous(pixel_xy)

            for idx in range(sampled_pts.shape[0]):
                t = t_samples[idx].item()
                x = x_samples[idx].item()
                y = y_samples[idx].item()

                for view_idx in range(num_views):
                    depth_val = bilinear_sample2d(
                        depths[0, view_idx, t][None],  # shape (1, 1, H, W)
                        torch.tensor([[x]], device=rgbs.device),
                        torch.tensor([[y]], device=rgbs.device),
                    ).item()

                    cam_xy_h = torch.tensor([[x, y, 1.0]], device=rgbs.device).T
                    K_inv = intrs_inv[0, view_idx, t]
                    extr_inv = extrs_inv[0, view_idx, t]

                    cam_xyz = (K_inv @ cam_xy_h).squeeze() * depth_val
                    cam_xyz_h = to_homogeneous(cam_xyz[None])[0]
                    world_xyz_h = extr_inv @ cam_xyz_h
                    world_xyz = from_homogeneous(world_xyz_h[None])[0]

                    support_point = torch.cat([torch.tensor([t], device=rgbs.device), world_xyz])
                    support_uniform_pts.append(support_point)

            if support_uniform_pts:
                support_uniform_pts = torch.stack(support_uniform_pts, dim=0)[None]  # (1, N, 4)
                support_points = torch.cat([support_points, support_uniform_pts], dim=1)

        if self.single_point:
            # Project the queries to each view
            # This will be needed if adding local grid points
            query_points_xyz_worldspace_homo = to_homogeneous(query_points_xyz_worldspace)
            query_points_perview_camera_xyz = torch.einsum('BVTij,BNj->BVTNi', extrs, query_points_xyz_worldspace_homo)
            query_points_perview_pixel_xy_homo = torch.einsum('BVTij,BVTNj->BVTNi', intrs,
                                                              query_points_perview_camera_xyz)
            query_points_perview_pixel_xy = from_homogeneous(query_points_perview_pixel_xy_homo)
            query_points_perview_camera_xyz = query_points_perview_camera_xyz[
                # Extract at the correct per-query timestep
                torch.arange(batch_size)[:, None, None],
                torch.arange(num_views)[None, :, None],
                query_points_t[:, None, :, 0],
                torch.arange(num_points)[None, None, :],
            ]
            query_points_perview_pixel_xy = query_points_perview_pixel_xy[  # Extract at the correct per-query timestep
                torch.arange(batch_size)[:, None, None],
                torch.arange(num_views)[None, :, None],
                query_points_t[:, None, :, 0],
                torch.arange(num_points)[None, None, :],
            ]
            query_points_perview_camera_z = query_points_perview_camera_xyz[..., -1:]

            traj_e = torch.zeros((batch_size, num_frames, num_points, 3), device=rgbs.device)
            vis_e = torch.zeros((batch_size, num_frames, num_points), device=rgbs.device)
            for point_idx in tqdm(range(num_points), desc="Single point evaluation"):
                # Support points for this query point
                support_points_i = torch.zeros((batch_size, 0, 4), device=rgbs.device)

                # Add the local support points
                if self.local_grid_size > 0:
                    t = query_points_t[0, point_idx, 0].item()
                    local_grid_points = torch.zeros((batch_size, 0, 4), device=rgbs.device)
                    for view_idx in range(num_views):
                        pixel_xy = get_points_on_a_grid(
                            size=self.local_grid_size,
                            extent=(self.local_extent, self.local_extent),
                            center=(query_points_perview_pixel_xy[0, view_idx, point_idx, 1].item(),
                                    query_points_perview_pixel_xy[0, view_idx, point_idx, 0].item()),
                            device=rgbs.device,
                        )
                        inside_frame = ((pixel_xy[0, :, 0] >= 0)
                                        & (pixel_xy[0, :, 0] < width)
                                        & (pixel_xy[0, :, 1] >= 0)
                                        & (pixel_xy[0, :, 1] < height))
                        if not inside_frame.any():
                            continue
                        pixel_xy = pixel_xy[:, inside_frame, :]
                        pixel_xy_homo = to_homogeneous(pixel_xy)
                        camera_z = bilinear_sample2d(
                            depths[0, view_idx, t][None],
                            pixel_xy[..., 0],
                            pixel_xy[..., 1],
                        ).permute(0, 2, 1)
                        camera_xyz = torch.einsum('Bij,BNj->BNi', intrs_inv[:, view_idx, t, :, :], pixel_xy_homo)
                        camera_xyz = camera_xyz * camera_z
                        camera_xyz_homo = to_homogeneous(camera_xyz)
                        world_xyz_homo = torch.einsum('Bij,BNj->BNi', extrs_inv[:, view_idx, t, :, :], camera_xyz_homo)
                        world_xyz = from_homogeneous(world_xyz_homo)
                        local_grid_points_i = torch.cat([torch.ones_like(world_xyz[:, :, :1]) * t, world_xyz], dim=2)
                        local_grid_points = torch.cat([local_grid_points, local_grid_points_i], dim=1)
                    support_points_i = torch.cat([support_points_i, local_grid_points], dim=1)

                # Add the global support points
                support_points_i = torch.cat([support_points_i, support_points], dim=1)

                # Forward pass for this query point
                query_points_i = torch.cat([query_points_3d[:, point_idx: point_idx + 1, :], support_points_i], dim=1)
                if query_points_view is not None:
                    query_points_view = torch.cat([
                        query_points_view[:, point_idx: point_idx + 1],
                        query_points_view.new_zeros(support_points_i[:, :, 0].shape),
                    ], dim=1)
                results_i = self.model(
                    rgbs,
                    depths=depths,
                    query_points=query_points_i,
                    intrs=intrs,
                    extrs=extrs,
                    iters=self.n_iters,
                    save_debug_logs=save_debug_logs and point_idx == 0,
                    debug_logs_path=debug_logs_path,
                    query_points_view=query_points_view,
                    previous_state=previous_state,
                    persistent_query_count=1,
                    return_rolling_state=return_rolling_state,
                    **kwargs,
                )
                traj_e[:, :, point_idx: point_idx + 1] = results_i["traj_e"][:, :, :1]
                vis_e[:, :, point_idx: point_idx + 1] = results_i["vis_e"][:, :, :1]

                if save_debug_logs and (point_idx in [0, 1, 2, 3, 4] or point_idx % 100 == 0):
                    visualizer = MultiViewVisualizer(
                        save_dir=debug_logs_path,
                        pad_value=16,
                        fps=12,
                        show_first_frame=0,
                        tracks_leave_trace=0,
                    )

                    # filename, pred_trajectories, pred_visibilities, qps
                    tuples_to_process = []
                    tuples_to_process += [(
                        f"predictor__pidx={point_idx}__viz_A_pred",
                        results_i["traj_e"][:, :, :1],
                        results_i["vis_e"][:, :, :1],
                        query_points_i[:, :1, :],
                    )]
                    tuples_to_process += [(
                        f"predictor__pidx={point_idx}__viz_B_pred_w_support",
                        results_i["traj_e"],
                        results_i["vis_e"],
                        query_points_i[:, :, :],
                    )]
                    if self.local_grid_size > 0 and local_grid_points.shape[1] > 0:
                        num_local_support_points = local_grid_points.shape[1]
                        tuples_to_process += [(
                            f"predictor__pidx={point_idx}__viz_C_local_support_grid",
                            results_i["traj_e"][:, :, 1:1 + num_local_support_points, :],
                            results_i["vis_e"][:, :, 1:1 + num_local_support_points],
                            query_points_i[:, 1:1 + num_local_support_points, :],
                        )]
                    if self.grid_size > 0:
                        num_global_support_points = support_points.shape[1]
                        tuples_to_process += [(
                            f"predictor__pidx={point_idx}__viz_D_global_support_grid",
                            results_i["traj_e"][:, :, -num_global_support_points:, :],
                            results_i["vis_e"][:, :, -num_global_support_points:],
                            query_points_i[:, -num_global_support_points:, :],
                        )]
                    for filename, pred_trajectories, pred_visibilities, qps in tuples_to_process:
                        filename = time_now() + "__" + filename
                        # Project the predictions to pixel space for visualization
                        pred_trajectories_pixel_xy_camera_z_per_view = torch.stack([
                            torch.cat(world_space_to_pixel_xy_and_camera_z(
                                world_xyz=pred_trajectories[0],
                                intrs=intrs[0, view_idx],
                                extrs=extrs[0, view_idx],
                            ), dim=-1)
                            for view_idx in range(num_views)
                        ], dim=0)[None]
                        pred_viz, _ = visualizer.visualize(
                            video=rgbs,
                            video_depth=depths,
                            tracks=pred_trajectories_pixel_xy_camera_z_per_view,
                            visibility=pred_visibilities > 0.5,
                            query_frame=qps[..., 0].long().clone(),
                            filename=filename,
                            writer=None,
                            step=0,
                            save_video=True,
                        )

        else:
            query_points_3d = torch.cat([query_points_3d, support_points], dim=1)
            if query_points_view is not None:
                query_points_view = torch.cat([
                    query_points_view, query_points_view.new_zeros(support_points[:, :, 0].shape)
                ], dim=1)
            results = self.model(
                rgbs,
                depths=depths,
                query_points=query_points_3d,
                intrs=intrs,
                extrs=extrs,
                iters=self.n_iters,
                save_debug_logs=save_debug_logs,
                debug_logs_path=debug_logs_path,
                query_points_view=query_points_view,
                previous_state=previous_state,
                persistent_query_count=num_points,
                return_rolling_state=return_rolling_state,
                **kwargs,
            )
            traj_e = results["traj_e"][:, :, :num_points, :]
            vis_e = results["vis_e"][:, :, :num_points]

            if save_debug_logs:
                visualizer = MultiViewVisualizer(
                    save_dir=debug_logs_path,
                    pad_value=16,
                    fps=12,
                    show_first_frame=0,
                    tracks_leave_trace=0,
                )
                num_support_grid_points = grid_points.shape[1] if self.grid_size > 0 else 0
                view_pts_all_timesteps = num_support_grid_points // num_views
                view_pts = view_pts_all_timesteps // self.n_grids_per_view if self.grid_size > 0 else 0
                for filename, pred_trajectories, pred_visibilities, qps in [
                    ("predictor__viz_A_pred", traj_e, vis_e, query_points_3d[:, :num_points, :]),
                    ("predictor__viz_B_pred_w_support_grid", results["traj_e"], results["vis_e"], query_points_3d),
                    ("predictor__viz_C_support_grid_only", results["traj_e"][:, :, num_points:, :],
                     results["vis_e"][:, :, num_points:], query_points_3d[:, num_points:, :]),
                    *[(
                            f"predictor__viz_D_support_grid_only__t-0_view-{view_idx}",
                            results["traj_e"][:, :,
                            num_points + view_pts * view_idx:num_points + view_pts * (view_idx + 1), :],
                            results["vis_e"][:, :,
                            num_points + view_pts * view_idx:num_points + view_pts * (view_idx + 1)],
                            query_points_3d[:, num_points + view_pts * view_idx:num_points + view_pts * (view_idx + 1),
                            :],
                    ) for view_idx in range(num_views)],
                ]:
                    filename = time_now() + "__" + filename
                    # Project the predictions to pixel space for visualization
                    pred_trajectories_pixel_xy_camera_z_per_view = torch.stack([
                        torch.cat(world_space_to_pixel_xy_and_camera_z(
                            world_xyz=pred_trajectories[0],
                            intrs=intrs[0, view_idx],
                            extrs=extrs[0, view_idx],
                        ), dim=-1)
                        for view_idx in range(num_views)
                    ], dim=0)[None]
                    pred_viz, _ = visualizer.visualize(
                        video=rgbs,
                        video_depth=depths,
                        tracks=pred_trajectories_pixel_xy_camera_z_per_view,
                        visibility=pred_visibilities > 0.5,
                        query_frame=qps[..., 0].long().clone(),
                        filename=filename,
                        writer=None,
                        step=0,
                        save_video=True,
                    )

        output = {
            "traj_e": traj_e,
            "vis_e": vis_e > self.visibility_threshold,
            "vis_e_as_prob": vis_e,
        }
        if return_rolling_state and "rolling_state" in results:
            output["rolling_state"] = results["rolling_state"]
        return output


def get_uniformly_sampled_pts(
        size: int,
        num_frames: int,
        extent: Tuple[float, ...],
        device: Optional[torch.device] = torch.device("cpu"),
):
    time_points = torch.randint(low=0, high=num_frames, size=(size, 1), device=device)
    space_points = torch.rand(size, 2, device=device) * torch.tensor(
        [extent[1], extent[0]], device=device
    )
    points = torch.cat((time_points, space_points), dim=1)
    return points[None]


def get_superpoint_sampled_pts(
        video,
        size: int,
        num_frames: int,
        extent: Tuple[float, ...],
        device: Optional[torch.device] = torch.device("cpu"),
):
    extractor = SuperPoint(max_num_keypoints=48).eval().cuda()
    points = list()
    for _ in range(8):
        frame_num = random.randint(0, int(num_frames * 0.25))
        key_points = extractor.extract(
            video[0, frame_num, :, :, :] / 255.0, resize=None
        )["keypoints"]
        frame_tensor = torch.full((1, key_points.shape[1], 1), frame_num).cuda()
        points.append(torch.cat([frame_tensor.cuda(), key_points], dim=2))
    return torch.cat(points, dim=1)[:, :size, :]


def get_sift_sampled_pts(
        video,
        size: int,
        num_frames: int,
        extent: Tuple[float, ...],
        device: Optional[torch.device] = torch.device("cpu"),
        num_sampled_frames: int = 8,
        sampling_length_percent: float = 0.25,
):
    import cv2
    # assert size == 384, "hardcoded for experiment"
    sift = cv2.SIFT_create(nfeatures=size // num_sampled_frames)
    points = list()
    for _ in range(num_sampled_frames):
        frame_num = random.randint(0, int(num_frames * sampling_length_percent))
        key_points, _ = sift.detectAndCompute(
            video[0, frame_num, :, :, :]
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .astype(np.uint8),
            None,
        )
        for kp in key_points:
            points.append([frame_num, int(kp.pt[0]), int(kp.pt[1])])
    return torch.tensor(points[:size], device=device)[None]
