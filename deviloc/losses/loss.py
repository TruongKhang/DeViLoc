import torch
import torch.nn as nn
import torch.nn.functional as F

from deviloc.utils.geometry import pts_img2cam, pts_world2cam, project_points3d


class DevilocLoss(nn.Module):
    def __init__(self, config):
        super(DevilocLoss, self).__init__()
        self.config = config
        self.weight_loss = config["weight_loss"]

    def compute_l1_loss(self, pred_pts, gt_pts, mask):
        error = ((pred_pts - gt_pts)**2).sum(dim=-1).sqrt()  # [B, N]
        n_pts = mask.sum(dim=1)
        error = (error * mask).sum(dim=1) / (n_pts + 1e-8)  # [B,]
        weighted_batch = (n_pts > 50).float()
        return (weighted_batch * error).sum() / error.shape[0]
    
    def compute_smooth_l1_loss(self, pred_pts, gt_pts, mask, beta=0.1):
        dist = ((pred_pts - gt_pts)**2).sum(dim=-1).sqrt()  # [B, N]
        thres_mask = (dist.detach() > beta).float()
        error = thres_mask * (dist - 0.5 * beta) + (1 - thres_mask) * (dist**2) / (2 * beta)
        error = (error * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B,]

        return error.mean()

    def compute_classification_loss(self, mconf, mpts2d, proj_mpts3d, mask, pixel_thresh=8):
        pixel_dist = ((mpts2d - proj_mpts3d)**2).sum(dim=-1).sqrt()
        mlabels = (pixel_dist < pixel_thresh).float()
        loss = F.binary_cross_entropy(mconf.squeeze(-1)[mask], mlabels[mask])
        
        return loss

    def forward(self, query_2d3d_matches, data_dict):
        gt_depths, gt_cam_mat, gt_cam_pose = data_dict["query_depth"], data_dict["query_K"], data_dict["query_T"]

        loss = 0.0

        for b, out_matches in enumerate(zip(*query_2d3d_matches)):
            b_gt_depth, b_gt_K, b_gt_pose = gt_depths[[b]], gt_cam_mat[[b]], gt_cam_pose[[b]]
            mpts2d, mpts3d, mconf = map(lambda x: x.unsqueeze(0), out_matches)
            gt_mpts3d_cam, valid_pts = pts_img2cam(mpts2d, b_gt_depth, torch.ones_like(mpts2d[:, :, 0]), b_gt_K)
            mpts3d_cam = pts_world2cam(mpts3d, b_gt_pose)

            mask = valid_pts & (mconf.squeeze(-1).detach() < 1-1e-3)

            z = gt_mpts3d_cam[..., -1]
            n_pts = mask.sum(dim=1, keepdim=True)
            mean_z = (z * mask).sum(dim=1, keepdim=True) / (n_pts + 1e-8)
            var_z = ((z - mean_z) ** 2 * mask).sum(dim=1, keepdim=True) / (n_pts + 1e-8)
            std_z = torch.sqrt(var_z.clamp(min=1e-12))

            z_scale = std_z.unsqueeze(-1)

            loss = loss + self.compute_l1_loss(mpts3d_cam / z_scale, gt_mpts3d_cam / z_scale, mask)

            proj_mpts3d_2d, _, _ = project_points3d(b_gt_K, b_gt_pose, mpts3d.detach())
            conf_loss = self.compute_classification_loss(mconf, mpts2d, proj_mpts3d_2d, mask)

            loss = loss + 0.5 * conf_loss

        loss /= len(gt_cam_mat)

        return self.weight_loss * loss
