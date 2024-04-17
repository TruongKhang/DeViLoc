import torch
import torch.nn.functional as F


def project_points3d(K, T, pts3d, img_size=None):
    """Project 3D points to 2D points using extrinsics and intrinsics
    Args:
        - K: camera intrinsic matrices (B, 3, 3)
        - T: camera extrinsic matrices (B, 4, 4)
        - t: world to camera translation (B, 3)
        - pts3d: 3D points (B, N, 3)
    Return:
        - pts2d: projected 2D points (B, N, 2)
        - valid: bool mask, indicates the 3d points that are projected in front of the camera
    """

    pts3d_cam = T[:, :3, :3] @ pts3d.permute(0, 2, 1) + T[:, :3, [3]] # [B, 3, N]
    # pts3d_cam = pts3d_cam.permute(0, 2, 1)  # [B, N, 3]
    depth = pts3d_cam[:, 2, :]  # [B, N]
    valid = depth > 0  # [B, N]
    pts3d_norm = pts3d_cam / (depth.unsqueeze(1) + 1e-6) # [B, 3, N]

    # Transform to pixel space. Last column is already guaranteed to be set to 1
    pts2d_proj = K @ pts3d_norm  # [B, 3, N]
    if img_size is not None:
        h, w = img_size
        valid = valid & (pts2d_proj[:, 0] >= 0) & (pts2d_proj[:, 0] < w) & (pts2d_proj[:, 1] >= 0) & (pts2d_proj[:, 1] < h)
    pts2d_proj = pts2d_proj.permute(0, 2, 1)[:, :, :2] # [B, N, 2]
    return pts2d_proj, depth, valid


def get_features_from_coords(feat_map, coords, scale=None):
    """
    feat_map: [B, C, H, W]
    coords: [B, Npts, 2]
    scale: [B, 2]
    """
    if scale is not None:
        coords = coords / scale.unsqueeze(1)

    h, w = feat_map.shape[2:]
    
    norm_x = coords[..., 0] * 2 / (w - 1) - 1
    norm_y = coords[..., 1] * 2 / (h - 1) - 1
    norm_coords = torch.stack((norm_x, norm_y), dim=-1)
    sampled_feats = F.grid_sample(feat_map, norm_coords.unsqueeze(2), align_corners=True) # [B, C, Npts, 1]
    sampled_feats = sampled_feats.squeeze(-1).permute(0, 2, 1) # [B, Npts, C]
    
    return sampled_feats


def pts_img2cam(pts2d, depth_maps, mask, K):
    """convert pixel coordinates into 3d points
    Args:
        - pts2d: [B, N, 2]
        - depths: [B, 1, H, W]
        - mask: [B, N]
        - K: [B, 3, 3]
    """
    h, w = depth_maps.shape[2:]
    norm_x = pts2d[..., 0] * 2 / (w - 1) - 1
    norm_y = pts2d[..., 1] * 2 / (h - 1) - 1
    norm_pts2d = torch.stack((norm_x, norm_y), dim=-1)
    depth_pts = F.grid_sample(depth_maps, norm_pts2d.unsqueeze(2), mode="nearest", align_corners=True) # [B, 1, N, 1]
    depth_pts = depth_pts.squeeze(1) * mask.unsqueeze(-1) # [B, N, 1]
    valid = depth_pts.squeeze(-1) > 0
    pts2d_extended = torch.cat((pts2d, torch.ones_like(pts2d[..., [0]])), dim=-1) # [B, N, 3]
    pts3d_cam = (pts2d_extended * depth_pts) @ K.inverse().transpose(1, 2) # [B, N, 3]
    
    return pts3d_cam, valid


def pts_cam2world(pts3d_cam, pose):
    """convert pixel coordinates into 3d points
    Args:
        - pts3d_cam: [B, N, 3]
        - pose: [B, 4, 4]
        - mask: [B, N]
    """
    R, t = pose[:, :3, :3], pose[:, :3, 3] # R: [B, 3, 3]; t: [B, 3]
    pts3d_world = (pts3d_cam - t.unsqueeze(1)) @ torch.inverse(R.transpose(1, 2))
    
    return pts3d_world


def pts_world2cam(pts3d_world, pose):
    R, t = pose[:, :3, :3], pose[:, :3, [3]]
    pts3d_cam = R @ pts3d_world.permute(0, 2, 1) + t # [B, 3, N]
    pts3d_cam = pts3d_cam.permute(0, 2, 1)  # [B, N, 3]
    
    return pts3d_cam 


if __name__ == "__main__":
    pts3d = torch.rand(1, 2, 3)
    pose = torch.rand(1, 4, 4)
    repoj_pts3d = pts_cam2world(pts_world2cam(pts3d, pose), pose)
    print(repoj_pts3d - pts3d)
