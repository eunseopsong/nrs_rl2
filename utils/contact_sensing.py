import torch
import trimesh
import numpy as np

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