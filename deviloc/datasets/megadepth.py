import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from third_party.feat_matcher.TopicFM.src.utils.dataset import read_megadepth_gray, read_megadepth_depth

from .utils import load_scene_data


np.random.seed(19951209)


class MegaDepth(Dataset):
    def __init__(self, root_dir, metadata_file, scenes3d, scene_list=None, mode="train", topk_retrieval=1, n_queries_per_scene=None,
                 data_aug=False, img_size=640, n_pts=(100, 10000), random_topk=False):
        self.root_dir = root_dir
        self.metadata_file = metadata_file
        self.scene_list = scene_list
        self.scenes3d_dir = scenes3d
        self.mode = mode 
        self.topk = topk_retrieval
        self.data_aug = data_aug
        self.img_size = img_size
        self.n_pts = n_pts
        self.random_topk = random_topk

        if (mode == "train") & data_aug:
            self.color_aug = A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2)
        else:
            self.color_aug = None

        self.scene_ids, self.query_ids, self.imgs_data = load_scene_data(metadata_file, scene_list, n_queries=n_queries_per_scene)

    def __len__(self):
        return len(self.query_ids)
    
    def read_img(self, img_info, color_aug=None):
        img_path = img_info.img_path
        img, mask, scale = read_megadepth_gray(img_path, self.img_size, df=16, padding=True, augment_fn=color_aug)[:3]
        return img, mask, scale
    
    def read_cam_params(self, K, R, t):
        intr_mat, extr_mat = np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32)
        intr_mat[:2] = K[:2]
        extr_mat[:3, :3] = R
        extr_mat[:3, 3] = t
        return torch.from_numpy(intr_mat).float(), torch.from_numpy(extr_mat).float()
    
    def read_scene3d(self, scene_name):
        scene3d_path = os.path.join(self.scenes3d_dir, f"{scene_name}.npy")
        scene3d = np.load(scene3d_path, allow_pickle=True).item()
        return scene3d

    def __getitem__(self, index):
        sid, qid = self.scene_ids[index], self.query_ids[index]

        # load info for scene
        scene_imgs = self.imgs_data[sid]
        scene_pts3d = self.read_scene3d(sid)

        # read query image
        query_img_data = scene_imgs[qid]
        # qimg, qscale = self.read_img(query_img_data)
        qimg, qmask, qscale = self.read_img(query_img_data, color_aug=self.color_aug)
        qK, qT = self.read_cam_params(query_img_data.K, query_img_data.R, query_img_data.t)
        # qdepth = self.read_depth_map(query_img_data.depth_path, resize=qimg.shape[-2:])
        qdepth = read_megadepth_depth(query_img_data.depth_path, pad_to=1700)
        qdepth = qdepth.unsqueeze(0)

        valid_db_ids = [idx for idx in query_img_data.topk if len(scene_imgs[idx].pt3d_ids) > self.n_pts[0]]
        if len(valid_db_ids) < self.topk:
            db_ids = (valid_db_ids * self.topk)[:self.topk]
        else:
            db_ids = np.random.choice(valid_db_ids, self.topk, replace=False) if self.random_topk else valid_db_ids[:self.topk]
        db_imgs, db_img_masks, db_img_scales = [], [], []
        db_Ks, db_Ts = [], []
        db_img_sizes = []
        db_depths = []
        db_img_paths = []
        db_pts3d, db_pts2d, db_mask_pts = [], [], []
        for rid in db_ids:
            rimg_data = scene_imgs[rid]
            db_img_sizes.append([rimg_data.h, rimg_data.w])

            rpts3d_ids = rimg_data.pt3d_ids
            rpts3d = np.stack([scene_pts3d[pid][:3] for pid in rpts3d_ids])
            rpts3d = torch.from_numpy(rpts3d).float()

            rpts2d = torch.from_numpy(rimg_data.pts2d).float()
            # if len(rpts3d) < self.n_pts[0]:
            #     continue

            rmask_pts3d = torch.zeros(self.n_pts[1], dtype=torch.bool)

            if len(rpts3d) > self.n_pts[1]:
                sampled_ids = np.random.choice(np.arange(len(rpts3d)), self.n_pts[1], replace=False)
                sampled_rpts3d = rpts3d[sampled_ids]
                sampled_rpts2d = rpts2d[sampled_ids]
                rmask_pts3d = torch.ones(self.n_pts[1], dtype=torch.bool)
            else:
                sampled_rpts3d = torch.zeros((self.n_pts[1], rpts3d.shape[1]), dtype=torch.float32)
                sampled_rpts3d[:len(rpts3d)] = rpts3d
                sampled_rpts2d = torch.zeros((self.n_pts[1], rpts2d.shape[1]), dtype=torch.float32)
                sampled_rpts2d[:len(rpts2d)] = rpts2d
                rmask_pts3d[:len(rpts3d)] = True
            
            db_pts3d.append(sampled_rpts3d)
            db_pts2d.append(sampled_rpts2d)
            db_mask_pts.append(rmask_pts3d)

            rimg, rmask, rscale = self.read_img(rimg_data)
            db_imgs.append(rimg)
            db_img_masks.append(rmask)
            db_img_scales.append(rscale)
            db_img_paths.append(rimg_data.img_path)

            rK, rT = self.read_cam_params(rimg_data.K, rimg_data.R, rimg_data.t)
            db_Ks.append(rK)
            db_Ts.append(rT)

            r_depth = read_megadepth_depth(rimg_data.depth_path, pad_to=1700)
            db_depths.append(r_depth.unsqueeze(0))

        db_depths = torch.stack(db_depths, dim=0)
        db_img_sizes = torch.tensor(db_img_sizes)
        db_data = map(lambda x: torch.stack(x, dim=0), (db_imgs, db_img_masks, db_img_scales, db_Ks, db_Ts, db_pts3d, db_pts2d))
        db_imgs, db_img_masks, db_img_scales, db_Ks, db_Ts, db_pts3d, db_pts2d = db_data

        return {
            "query_img": qimg, "query_depth": qdepth, "query_img_mask": qmask,
            "query_K": qK, "query_T": qT, "query_img_path": query_img_data.img_path,
            "query_img_scale": qscale, "query_img_size": torch.tensor([query_img_data.h, query_img_data.w]),
            "db_imgs": db_imgs, "db_img_masks": db_img_masks,
            "db_img_scales": db_img_scales, "db_img_sizes": db_img_sizes, "ref_img_paths": db_img_paths,
            "db_Ks": db_Ks, "db_Ts": db_Ts, "db_depths": db_depths, 
            "db_pts3d": db_pts3d, "db_pts2d": db_pts2d, "db_mask_pts3d": torch.stack(db_mask_pts, dim=0)
        }
