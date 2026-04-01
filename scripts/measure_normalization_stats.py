import sys
import os
import torch
import numpy as np
from mvtracker.datasets.generic_scene_dataset import compute_auto_scene_normalization
from mvtracker.datasets.utils import transform_scene

import huggingface_hub
sample_path = huggingface_hub.hf_hub_download(repo_id="ethz-vlg/mvtracker", filename="data_sample.npz", repo_type="model")
sample = np.load(sample_path)
depths = torch.from_numpy(sample["depths"]).float()
extrs = torch.from_numpy(sample["extrs"]).float()
intrs = torch.from_numpy(sample["intrs"]).float()
query_points = torch.from_numpy(sample["query_points"]).float()

# Normalization
pseudo_confs = torch.ones_like(depths) * 10
scale, translation = compute_auto_scene_normalization(depths, pseudo_confs, extrs, intrs)
rot = torch.eye(3, dtype=torch.float32)

depths_n, extrs_n, query_n, _, _ = transform_scene(
    transformation_scale=scale,
    transformation_rotation=rot,
    transformation_translation=translation,
    depth=depths,
    extrs=extrs,
    query_points=query_points,
    traj3d_world=None
)

# Compute post-norm stats
valid_dn = depths_n[depths_n > 0]
print("--- DEMO SCENE POST-NORMALIZATION ---")
print(f"Depth  - min: {valid_dn.min():.4f}, max: {valid_dn.max():.4f}, mean: {valid_dn.mean():.4f}, median: {valid_dn.median():.4f}")

q_coords = query_n[:, 1:4]
q_dist = torch.norm(q_coords, dim=1)
print(f"Query  - Dist from origin: mean={q_dist.mean():.4f}, median={q_dist.median():.4f}")
print(f"Query  - XYZ Bounds: X[{q_coords[:,0].min():.4f}, {q_coords[:,0].max():.4f}] Y[{q_coords[:,1].min():.4f}, {q_coords[:,1].max():.4f}] Z[{q_coords[:,2].min():.4f}, {q_coords[:,2].max():.4f}]")

cam_centers = []
V = extrs_n.shape[0]
for v in range(V):
    R = extrs_n[v, 0, :3, :3].numpy()
    T = extrs_n[v, 0, :3, 3].numpy()
    c = -R.T @ T
    cam_centers.append(np.linalg.norm(c))
print(f"Camera - Dists from origin: {cam_centers} (Median: {np.median(cam_centers):.4f})")
print("")
