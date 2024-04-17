from argparse import Namespace
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from deviloc.utils.read_write_model import read_model
from third_party.feat_matcher.TopicFM.src.utils.dataset import read_img_gray
from .utils import do_covisibility_clustering, get_cam_matrix_from_colmap_model

np.random.seed(19951209)


class RobotCarSeasons(Dataset):
    def __init__(self, root_dir, img_pairs, sfm_dir, query_dir, 
                 topk_retrieval=1, img_size=640, n_pts=(10, 10000)):
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "images")

        self.location_ids = ["{:0>3}".format(idx) for idx in range(1, 50)]
        self.query_dir = query_dir
        self.query_pairs = self.load_query_info(img_pairs)
        self.sfm_dir = sfm_dir
        
        self.topk = topk_retrieval
        self.img_size = img_size
        self.n_pts = n_pts

        self.db_info = self.load_sfm_model()
        self.query_names = list(self.query_pairs.keys())
    
    def load_query_info(self, query_pair_path):
        h, w = 1024, 1024
        # load intrisics, extrinsics of each camera
        intrinsics_dir = os.path.join(self.root_dir, "intrinsics")
        extrinsics_dir = os.path.join(self.root_dir, "extrinsics")
        query_cam_info = {key: {"K": np.eye(3), "pycol_cam": None, "T": np.eye(4)} for key in ["left", "rear", "right"]}
        for idx, (key, intr) in enumerate(query_cam_info.items()):
            with open(os.path.join(intrinsics_dir, f"{key}_intrinsics.txt")) as f:
                lines = f.readlines()
                fx, fy, cx, cy = map(lambda x: float(x.strip().split()[-1]), lines[:4])
                intr["K"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                intr["pycol_cam"] = " ".join([str(e) for e in [idx, "SIMPLE_RADIAL", w, h, fx, cx, cy, "0.0"]])
            
            intr["T"] = np.loadtxt(os.path.join(extrinsics_dir, f"{key}_extrinsics.txt"), dtype=np.float32, delimiter=",")
        
        query_info = {}

        for loc_id in self.location_ids:
            query_file_path = f"{self.root_dir}/{self.query_dir}/queries_location_{loc_id}.txt"
            with open(query_file_path, "r") as fid:
                while True:
                    line = fid.readline()
                    if not line:
                        break
                    qname = line.strip()
                    if len(qname) > 0:
                        cam_type = qname.split("/")[1]
                        query_info[qname] = {"K": torch.from_numpy(query_cam_info[cam_type]["K"]).float(), 
                                             "pycol_cam": query_cam_info[cam_type]["pycol_cam"],
                                             "T_cam2car": torch.from_numpy(query_cam_info[cam_type]["T"]).float(),
                                             "location": loc_id}

        # pair_info = {}
        with open(query_pair_path, "r") as f:
            for line in f:
                qname, rname = line.strip().split()
                if "retrieval_list" in query_info[qname]:
                    query_info[qname]["retrieval_list"].append(rname)
                else:
                    query_info[qname]["retrieval_list"] = [rname]
        
        return query_info

    def __len__(self):
        return len(self.query_names)
    
    def load_sfm_model(self, file_type=".txt"):
        db_info = {}
        for loc_id in self.location_ids:
            sfm_model_path = f"{self.root_dir}/{self.sfm_dir}/{loc_id}_aligned"
            colmap_cams, colmap_imgs, colmap_pts3d = read_model(sfm_model_path, ext=file_type)
            db_info[loc_id] = {"colmap_pts3d": colmap_pts3d, "colmap_imgs": colmap_imgs}

            for img_id, img_model in colmap_imgs.items():
                img_name = img_model.name
                img_name = img_name.replace("png", "jpg") # file type in colmap model is different from file type in the "images" folder
            
                cam_model = colmap_cams[img_model.camera_id]
                cam_mat, radial = get_cam_matrix_from_colmap_model(cam_model.model, cam_model.params)

                R, t = np.array(img_model.qvec2rotmat()), np.array(img_model.tvec)
                cam_pose = np.eye(4, dtype=np.float32)
                cam_pose[:3, :3], cam_pose[:3, 3] = R, t

                pts3d_ids, pts2d = [], []
                for idx, pid in enumerate(img_model.point3D_ids):
                    if pid > -1:
                        pts3d_ids.append(pid)
                        pts2d.append(img_model.xys[idx])
                pts3d_ids = np.array(pts3d_ids)
                pts2d = np.array(pts2d)

                db_info[loc_id][img_name] = Namespace(id=img_id, K=cam_mat, T=torch.from_numpy(cam_pose), 
                                                      pts3d_ids=pts3d_ids, pts2d=pts2d, radial=radial)

        return db_info

    def __getitem__(self, index):
        qimg_name = self.query_names[index]

        # read query image
        qimg_path = os.path.join(self.image_dir, qimg_name)
        qimg, qscale = read_img_gray(qimg_path, resize=self.img_size)
        qK = self.query_pairs[qimg_name]["K"]

        location_id = self.query_pairs[qimg_name]["location"]
        location_info = self.db_info[location_id]
        location_pts3d = location_info["colmap_pts3d"]

        db_img_names = self.query_pairs[qimg_name]["retrieval_list"]
        rimg_names = [rname for rname in db_img_names if len(location_info[rname].pts3d_ids) > self.n_pts[0]]
        rimg_names = rimg_names[:min(self.topk, len(rimg_names))]
        rimg_ids = [location_info[rname].id for rname in rimg_names]
        local_img_ids = {global_id: local_id for local_id, global_id in enumerate(rimg_ids)}
        covis_clusters = do_covisibility_clustering(rimg_ids, location_info["colmap_imgs"], location_pts3d)
        for cluster in covis_clusters:
            for i in range(len(cluster)):
                cluster[i] = local_img_ids[cluster[i]]

        db_imgs = []
        db_Ks, db_Ts = [], []
        db_pts3d, db_pts2d = [], []
        db_dist_coeffs = []
        for rname in rimg_names:
            rimg_data = location_info[rname]
            db_dist_coeffs.append(rimg_data.radial)

            rpts3d_ids = rimg_data.pts3d_ids
            rpts3d = np.stack([location_pts3d[pid].xyz for pid in rpts3d_ids])
            rpts3d = torch.from_numpy(rpts3d).float()
            rpts2d = torch.from_numpy(rimg_data.pts2d).float()

            if len(rpts3d) > self.n_pts[1]:
                sampled_ids = np.random.choice(np.arange(len(rpts3d)), self.n_pts[1], replace=False)
                rpts3d = rpts3d[sampled_ids]
                rpts2d = rpts2d[sampled_ids]
            
            db_pts3d.append(rpts3d)
            db_pts2d.append(rpts2d)

            rimg_path = os.path.join(self.image_dir, rname)
            # rimg, rscale = read_img_gray(rimg_path, resize=self.img_size)
            db_imgs.append(rimg_path)
            # db_img_scales.append(rscale)

            db_Ks.append(rimg_data.K)
            db_Ts.append(rimg_data.T)

        return {
            "query_img": qimg, "query_img_scale": qscale, 
            "query_K": qK, "query_name": qimg_name,
            "query_pycol_cam": self.query_pairs[qimg_name]["pycol_cam"],
            "query_T_cam2car": self.query_pairs[qimg_name]["T_cam2car"],
            "db_imgs": db_imgs,
            "db_Ks": db_Ks, "db_Ts": db_Ts, "db_dist_coeffs": db_dist_coeffs,
            "db_pts3d": db_pts3d, "db_pts2d": db_pts2d, 
            "covis_clusters": [torch.tensor(clus) for clus in covis_clusters]
        }
