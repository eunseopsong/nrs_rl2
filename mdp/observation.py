# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for UR10e spindle environment.
- Integrated with nrs_fk_core (C++ FK module)
- Horizon-based trajectory loaders (positions)
- Includes EE pose (x, y, z, roll, pitch, yaw), and camera sensors
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
import torch

from ..utils import debug as local_debug

# ------------------------------------------------------
# Conditional import (avoid double registration)
# ------------------------------------------------------
if "nrs_fk_core" not in sys.modules:
    from nrs_fk_core import FKSolver
else:
    FKSolver = sys.modules["nrs_fk_core"].FKSolver


# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_hdf5_positions: torch.Tensor | None = None
_step_idx = 0


# ------------------------------------------------------
# EE pose observation (x, y, z, roll, pitch, yaw)
# ------------------------------------------------------
def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """
    Returns end-effector pose (x, y, z, roll, pitch, yaw)
    - Reads q1~q6 and runs FK via nrs_fk_core.FKSolver
    - Output: (num_envs, 6) torch tensor on env device
    """
    robot = env.scene[asset_name]
    q = robot.data.joint_pos[:, :6]
    device = q.device
    num_envs = q.shape[0]

    fk_solver = FKSolver(tool_z=0.239, use_degrees=False)

    # batched FK first
    if hasattr(fk_solver, "compute_batch"):
        try:
            q_np = q.detach().cpu().numpy().astype(float)
            ok, poses = fk_solver.compute_batch(q_np, as_degrees=False)
            if not ok:
                ee_pose = torch.full((num_envs, 6), float("nan"), device=device, dtype=torch.float32)
            else:
                ee_pose = torch.tensor(poses, dtype=torch.float32, device=device)
            return ee_pose
        except Exception:
            pass

    if hasattr(fk_solver, "forward"):
        try:
            poses = fk_solver.forward(q)
            ee_pose = poses if isinstance(poses, torch.Tensor) else torch.tensor(
                poses, dtype=torch.float32, device=device
            )
            if ee_pose.device != device:
                ee_pose = ee_pose.to(device)
            return ee_pose
        except Exception:
            pass

    # fallback: per-env loop
    ee_pose_list = []
    q_cpu = q.detach().cpu()
    for i in range(num_envs):
        q_np = q_cpu[i].numpy().astype(float)
        ok, pose = fk_solver.compute(q_np, as_degrees=False)
        if not ok:
            ee_pose_list.append([float("nan")] * 6)
        else:
            ee_pose_list.append([pose.x, pose.y, pose.z, pose.r, pose.p, pose.yaw])

    ee_pose = torch.tensor(ee_pose_list, dtype=torch.float32, device=device)
    assert ee_pose.ndim == 2 and ee_pose.shape[1] == 6, f"[EE_POSE] Invalid shape: {ee_pose.shape}"
    return ee_pose


# ------------------------------------------------------
# HDF5 loader: Positions
# ------------------------------------------------------
def load_hdf5_positions(
    env: "ManagerBasedRLEnv",
    env_ids,
    file_path: str,
    dataset_key: str = "target_positions",
):
    """
    Load HDF5 trajectory (position targets).
    """
    global _hdf5_positions, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"[ERROR] HDF5 (positions): '{dataset_key}' not found. Available keys: {list(f.keys())}"
            )
        data = f[dataset_key][:]

    _hdf5_positions = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    local_debug.print_hdf5_positions_loaded(_hdf5_positions.shape, file_path)


# ------------------------------------------------------
# Observation: target positions (horizon-based)
# ------------------------------------------------------
def get_hdf5_target_positions(env: "ManagerBasedRLEnv", horizon: int = 5) -> torch.Tensor:
    """
    Return future EE pose targets (x,y,z,roll,pitch,yaw) flattened: (N, horizon*6).
    """
    global _hdf5_positions

    if _hdf5_positions is None:
        d = 6
        return torch.zeros((env.num_envs, horizon * d), device=env.device, dtype=torch.float32)

    t_total, d = _hdf5_positions.shape
    step = int(env.episode_length_buf[0].item())
    ep_len = int(env.max_episode_length)
    idx = min(int((step / max(ep_len, 1)) * t_total), t_total - 1)

    future_idx = torch.arange(idx, idx + horizon, device=_hdf5_positions.device)
    future_idx = torch.clamp(future_idx, max=t_total - 1)

    future_targets = _hdf5_positions[future_idx].reshape(1, horizon * d)
    return future_targets.repeat(env.num_envs, 1)


def build_adjacency_dict(mesh: trimesh.Trimesh):
    """
    논문 [E 파트] 구현을 위한 준비 작업: 
    각 삼각형(Face)이 어떤 이웃 삼각형들과 모서리를 공유하는지 족보(Dictionary)를 만듭니다.
    """
    adjacency = {}
    for i in range(len(mesh.faces)):
        adjacency[i] = []
    
    # mesh.face_adjacency는 모서리를 공유하는 두 삼각형의 인덱스 쌍 [face1, face2] 형태입니다.
    for face1, face2 in mesh.face_adjacency:
        adjacency[face1].append(face2)
        adjacency[face2].append(face1)
        
    return adjacency

def get_contact_point_mesh(
    f_world: torch.Tensor, 
    m_world: torch.Tensor, 
    vertices: torch.Tensor, 
    faces: torch.Tensor,
    adjacency: dict,
    max_iter: int = 10
) -> torch.Tensor | None:
    """
    Arbitrary Surface Contact Sensing Algorithm (PyTorch 기반)
    논문 Algorithm 1 완벽 구현본
    """
    # 0. 힘이 너무 작으면(노이즈) 계산 안 함
    f_norm_sq = torch.sum(f_world**2)
    if f_norm_sq < 1e-4:
        return None

    # 1. [C 파트] 렌치 축(Wrench Axis) 기준점 r0 및 초기 후보 찾기
    r0 = torch.cross(f_world, m_world) / f_norm_sq
    f_hat = f_world / torch.sqrt(f_norm_sq)

    face_verts = vertices[faces] # (F, 3, 3)
    centroids = torch.mean(face_verts, dim=1) # (F, 3)
    
    g_minus_r0 = centroids - r0
    proj_lengths = torch.matmul(g_minus_r0, f_hat)
    proj_vectors = proj_lengths.unsqueeze(1) * f_hat
    
    distances = torch.norm(g_minus_r0 - proj_vectors, dim=1)
    current_face_idx = torch.argmin(distances).item()

    # 2. [D & E 파트] 접촉점 탐색 반복 루프 (While loop 대체)
    for _ in range(max_iter):
        v1, v2, v3 = vertices[faces[current_face_idx]]
        
        # 모서리 벡터 및 평면 수직 법선 벡터(n)
        e1 = v2 - v1
        e2 = v3 - v2
        e3 = v1 - v3
        n = torch.cross(e1, e2)
        n = n / torch.norm(n)

        # 평면과의 교점 계산 (lambda 및 c)
        denom = torch.dot(n, f_world)
        if torch.abs(denom) < 1e-6:
            break # 힘과 평면이 평행함
            
        lam = torch.dot(n, (v1 - r0)) / denom
        c = r0 + lam * f_world

        # 외적 판별법 (경계 조건 검증)
        B1 = torch.dot(torch.cross(c - v1, e1), n)
        B2 = torch.dot(torch.cross(c - v2, e2), n)
        B3 = torch.dot(torch.cross(c - v3, e3), n)

        # [상황 A] 3개 모두 양수 (내부 명중!) -> 탐색 성공
        if B1 >= 0 and B2 >= 0 and B3 >= 0:
            return c
            
        # [상황 B & C] E 파트: 이웃 갈아타기 (Switching)
        else:
            neighbors = adjacency[current_face_idx]
            if len(neighbors) == 0:
                break
                
            # 구현의 편의성과 속도를 위해, 빗나갔을 경우 연결된 이웃 중 
            # 렌치 축(레이저 빔)과 가장 가까운 이웃으로 빠르게 갈아탑니다.
            best_neighbor = -1
            min_dist = float('inf')
            
            for n_idx in neighbors:
                n_centroid = centroids[n_idx]
                dist_to_axis = torch.norm((n_centroid - r0) - torch.dot(n_centroid - r0, f_hat) * f_hat)
                if dist_to_axis < min_dist:
                    min_dist = dist_to_axis
                    best_neighbor = n_idx
                    
            if best_neighbor == current_face_idx or best_neighbor == -1:
                break
            
            current_face_idx = best_neighbor # 다음 타자로 업데이트!

    return None