from yacs.config import CfgNode
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Pts3DInfer
from deviloc.utils.geometry import get_features_from_coords
from third_party.feat_matcher.TopicFM.src.utils.dataset import read_img_gray
from third_party.feat_matcher.TopicFM.src.models import TopicFM
from third_party.feat_matcher.TopicFM.src import get_model_cfg


class Dense2D3DMatcher(nn.Module):
    def __init__(self, config):
        super(Dense2D3DMatcher, self).__init__()
        self.config = config
        topicfm_cfg = CfgNode(get_model_cfg())
        topicfm_cfg.merge_from_other_cfg(config["matcher"]["topicfm"])

        self.matcher_scales = {"coarse": topicfm_cfg["resolution"][0], 
                               "fine": topicfm_cfg["resolution"][1]}
        
        # print(topicfm_cfg)
        self.matcher = TopicFM(topicfm_cfg)
        pretrained_matcher = torch.load(config["matcher"]["pretrained_ckpt"])
        self.matcher.load_state_dict(pretrained_matcher["state_dict"])
        for p in self.matcher.parameters():
            p.requires_grad = False

        self.pts3d_infer = Pts3DInfer(config["pts3d_infer"], config["matcher"]["dim_feat"])

    def load_pretrained_model(self, ckpt_path):
        print(f"Load pretrained model from {ckpt_path}")
        pretrained_ckpt = torch.load(ckpt_path)
        state_dict = {}
        for k, v in pretrained_ckpt["state_dict"].items():
            state_dict[k.replace("model.", "")] = v
        self.load_state_dict(state_dict)

    def extract_2d2d_matches(self, qimg, qscale, rimg, rscale, obs_pts2d, qmask=None, rmask=None):
        input_matcher = {"image0": qimg, "scale0": qscale, 
                         "image1": rimg, "scale1": rscale}
        if qmask is not None:
            qmask, rmask = map(lambda m: F.interpolate(m.unsqueeze(1).float(),
                                                       scale_factor=1./self.matcher_scales["coarse"], mode='nearest',
                                                       recompute_scale_factor=False).squeeze(1).bool(), (qmask, rmask))
            input_matcher.update({"mask0": qmask, "mask1": rmask})

        self.matcher(input_matcher)

        qkpts, rkpts = input_matcher["mkpts0_f"], input_matcher["mkpts1_f"]
        # if len(qkpts) < 100:
        #     return None

        rfeat_f = input_matcher["feat_map1"]
        mconf = input_matcher["mconf"]
        n_matches = self.config["matcher"]["n_matches"]
        selected_ids = torch.argsort(mconf, descending=True)[:n_matches]
        qkpts, rkpts = qkpts[selected_ids], rkpts[selected_ids]

        scale_f2i = self.matcher_scales["fine"]
        rscale_f = rscale * scale_f2i
        rdescs = get_features_from_coords(rfeat_f, rkpts.unsqueeze(0), rscale_f)

        obs_descs = get_features_from_coords(rfeat_f, obs_pts2d, rscale_f)
                
        return {
            "qkpts": qkpts.unsqueeze(0), "rkpts": rkpts.unsqueeze(0),
            "rfeats": rdescs, "r_obs_feats": obs_descs, "rfeatmap_scale": rscale_f
        }

    def get_obs_data(self, obs_pts2d, cam_params, kpts2d):
        device_ = obs_pts2d.device

        sim_pts2d_mat = torch.cdist(obs_pts2d, kpts2d).squeeze(0) 
        sim_pts2d_mask = sim_pts2d_mat < self.config["qt_size"]
        sim_pts2d_mask = sim_pts2d_mask * (sim_pts2d_mat == sim_pts2d_mat.min(dim=0, keepdim=True)[0])
        obs_m_ids, kpts2d_m_ids = torch.nonzero(sim_pts2d_mask, as_tuple=True)

        kpts2d_infer_mask = torch.ones(kpts2d.shape[1], dtype=torch.bool, device=device_)
        kpts2d_infer_mask[kpts2d_m_ids] = False

        return obs_m_ids, kpts2d_m_ids, kpts2d_infer_mask

    def merge_2d3d_matches(self, pts2d, pts3d, pts_conf, qt_size=2, max_n_pts=None):
        """
            pts2d: [B, Nviews, Npts, 2]
            pts3d: [B, Nviews, Npts, 3]
            mask: [B, Nviews, Npts]
        """
        b_pts2d, b_pts3d = pts2d.squeeze(0), pts3d.squeeze(0)
        b_pts_conf = pts_conf.squeeze(0)
        if not self.training:
            mask = b_pts_conf.squeeze(-1).detach() > self.config["conf_thr"]
            b_pts2d, b_pts3d, b_pts_conf = b_pts2d[mask], b_pts3d[mask], b_pts_conf[mask]

        uniq_pts2d, inv_inds = torch.unique(torch.round(b_pts2d / qt_size).long() * qt_size,
                                            sorted=True, return_inverse=True, dim=0)
        n_merged_pts = len(uniq_pts2d)
        merged_pts2d = torch.empty((n_merged_pts, 2), device=b_pts2d.device, dtype=b_pts2d.dtype).fill_(0)
        merged_pts2d.index_reduce_(0, inv_inds, b_pts2d * b_pts_conf, "mean", include_self=False)
        merged_pts3d = torch.empty((n_merged_pts, 3), device=b_pts3d.device, dtype=b_pts3d.dtype).fill_(0)
        merged_pts3d.index_reduce_(0, inv_inds, b_pts3d * b_pts_conf, "mean", include_self=False)
        merged_conf = torch.empty((n_merged_pts, 1), device=b_pts_conf.device, dtype=b_pts_conf.dtype).fill_(0)
        merged_conf.index_reduce_(0, inv_inds, b_pts_conf, "mean", include_self=False)

        merged_pts2d = merged_pts2d / (merged_conf + 1e-6)
        merged_pts3d = merged_pts3d / (merged_conf + 1e-6)

        return merged_pts2d, merged_pts3d, merged_conf

    def forward(self, input_dict):
        all_query_pts3d, all_query_pts2d, all_query_conf = [], [], []

        # unbind
        if isinstance(input_dict["db_imgs"], torch.Tensor):
            for k, v in input_dict.items():
                if "db_" in k:
                    input_dict[k] = torch.unbind(v, dim=1)

        n_db_views = len(input_dict["db_imgs"])
        batch_size = input_dict["query_img"].shape[0]

        self.matcher.eval()

        for b_ in range(batch_size):
            qimg, qscale = input_dict["query_img"][[b_]], input_dict["query_img_scale"][[b_]]
            qmask = None
            if "query_img_mask" in input_dict:
                qh = input_dict["query_img_mask"][b_, :, 0].sum().long()
                qw = input_dict["query_img_mask"][b_, 0, :].sum().long()
                # print("query_size: ", qh, qw)
                qimg = qimg[:, :, :qh, :qw]

            out_2d2d_matches = []
            
            with torch.inference_mode():
                for idx in range(n_db_views):
                    rimg = input_dict["db_imgs"][idx][b_]
                    if isinstance(rimg, str):
                        rimg, rscale = read_img_gray(rimg, resize=max(qimg.shape[2], qimg.shape[3]))
                        rimg = rimg.unsqueeze(0).to(qimg.device)
                        rscale = rscale.unsqueeze(0).to(qscale.device)
                        # print(rimg.shape, rscale.shape)
                    else:
                        rimg = rimg.unsqueeze(0)
                        rscale = input_dict["db_img_scales"][idx][[b_]]

                    if "db_img_masks" in input_dict:
                        rmask = input_dict["db_img_masks"][idx][b_]
                        rh, rw = rmask[:, 0].sum().long(), rmask[0, :].sum().long()
                        rimg = rimg[:, :, :rh, :rw]
                    rmask = None

                    r_obs_pts2d = input_dict["db_pts2d"][idx][[b_]]
                    if "db_mask_pts3d" in input_dict:
                        r_obs_mask = input_dict["db_mask_pts3d"][idx][[b_]]
                        r_obs_pts2d = r_obs_pts2d[r_obs_mask].unsqueeze(0)

                    out_matches = self.extract_2d2d_matches(qimg, qscale, rimg, rscale, r_obs_pts2d, qmask, rmask)
                    out_2d2d_matches.append(out_matches)

            b_pts3d, b_qpts2d, b_pts3d_conf = [], [], []
            for idx in range(n_db_views):
                out_matches = out_2d2d_matches[idx]
                # if out_matches is None:
                #     continue
                
                rK, rT = input_dict["db_Ks"][idx][[b_]], input_dict["db_Ts"][idx][[b_]]
                dist_coeffs = input_dict["db_dist_coeffs"][idx][[b_]] if "db_dist_coeffs" in input_dict else None
                r_obs_pts3d = input_dict["db_pts3d"][idx][[b_]]
                # if "db_pts2d" in input_dict:
                r_obs_pts2d = input_dict["db_pts2d"][idx][[b_]]
                if "db_mask_pts3d" in input_dict:
                    r_obs_mask = input_dict["db_mask_pts3d"][idx][[b_]]
                    r_obs_pts3d = r_obs_pts3d[r_obs_mask].unsqueeze(0)
                    r_obs_pts2d = r_obs_pts2d[r_obs_mask].unsqueeze(0)

                m_r_obs_ids, m_kpts_ids, infer_kpts_mask = self.get_obs_data(r_obs_pts2d, (rK, rT), out_matches["rkpts"])

                pts3d = torch.zeros((1, out_matches["qkpts"].shape[1], 3), dtype=r_obs_pts3d.dtype, device=r_obs_pts3d.device)
                pts3d[:, m_kpts_ids] = r_obs_pts3d[:, m_r_obs_ids] # get observed 3D points that are matched to 2D keypoints

                pts3d_conf = torch.ones_like(pts3d[:, :, [0]])

                # if infer_kpts_mask.sum() == 0:
                #     infer_kpts_mask[0] = True
                
                extra_pts3d, extra_pts3d_conf = self.pts3d_infer(r_obs_pts3d, out_matches["r_obs_feats"],
                                                                 out_matches["rkpts"][:, infer_kpts_mask], out_matches["rfeats"][:, infer_kpts_mask], 
                                                                 (rK, rT, dist_coeffs))
                pts3d[:, infer_kpts_mask] = extra_pts3d
                pts3d_conf[:, infer_kpts_mask] = extra_pts3d_conf

                b_pts3d.append(pts3d)
                b_qpts2d.append(out_matches["qkpts"])
                b_pts3d_conf.append(pts3d_conf)

            # if len(b_qpts2d) == 0:
            #     return None
            
            b_pts3d, b_qpts2d, b_pts3d_conf = map(lambda x: torch.cat(x, dim=1), (b_pts3d, b_qpts2d, b_pts3d_conf))

            mpts2d, mpts3d, mconf = self.merge_2d3d_matches(b_qpts2d, b_pts3d, b_pts3d_conf, qt_size=self.config["qt_size"],
                                                            max_n_pts=self.config["n_2d3d_matches"])
            
            all_query_pts2d.append(mpts2d)
            all_query_pts3d.append(mpts3d)
            all_query_conf.append(mconf)

        return all_query_pts2d, all_query_pts3d, all_query_conf
