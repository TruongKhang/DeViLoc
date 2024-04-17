from argparse import Namespace
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from deviloc.utils.read_write_model import read_model
from third_party.feat_matcher.TopicFM.src.utils.dataset import read_img_gray
from .utils import do_covisibility_clustering, get_cam_matrix_from_colmap_model

np.random.seed(19951209)


class CMUSeasons(Dataset):
    def __init__(self, root_dir, slice_list=None, pair_type="cosplace20", topk_retrieval=1, img_size=640, n_pts=(10, 10000)):
        self.root_dir = root_dir
        self.pair_type = pair_type

        self.slice_ids = slice_list if slice_list is not None else list(range(2, 22))
        
        self.cam_intrinsics = self.load_cam_intrinsics()
        self.query_info = self.load_query_info(self.slice_ids, self.cam_intrinsics)
        
        self.topk = topk_retrieval
        self.img_size = img_size
        self.n_pts = n_pts

        self.db_info = self.load_sfm_model(self.slice_ids)
        self.query_names = list(self.query_info.keys())

    def load_cam_intrinsics(self):
        cam_intr_file = f"{self.root_dir}/intrinsics.txt"
        cam_intrs = {}
        with open(cam_intr_file) as f:
            for line in f:
                if "#" not in line:
                    colmap_cam = line.strip()
                    elems = colmap_cam.split()
                    camera_id = elems[0]
                    model = elems[1]
                    params = np.array(tuple(map(float, elems[4:])))
                    K, radial = get_cam_matrix_from_colmap_model(model, params)
                    cam_intrs[camera_id] = {"pycol_cam": colmap_cam, "K": K, "radial": radial}
        
        return cam_intrs
    
    def load_query_info(self, slice_ids, cam_intrinsics):
        query_info = {}

        for sid in slice_ids:
            slice_dir = f"{self.root_dir}/slice{sid}"
            query_file_path = f"{slice_dir}/test-images-slice{sid}.txt"
            with open(query_file_path, "r") as fid:
                while True:
                    line = fid.readline()
                    if not line:
                        break
                    qname = line.strip()
                    if len(qname) > 0:
                        cam_type = qname.split("_")[2]
                        assert cam_type in cam_intrinsics, f"cannot find camera {cam_type}"
                        qid = f"slice{sid}/{qname}"
                        query_info[qid] = {"K": cam_intrinsics[cam_type]["K"], 
                                             "radial": cam_intrinsics[cam_type]["radial"],
                                             "pycol_cam": cam_intrinsics[cam_type]["pycol_cam"],
                                             "slice_id": sid, "query_name": qname}

            query_pair_path = f"{slice_dir}/pairs-query-{self.pair_type}.txt"
            with open(query_pair_path, "r") as f:
                for line in f:
                    qname, rname = line.strip().split()
                    qname = qname.replace("query/", "")
                    rname = rname.replace("database/", "")
                    qid = f"slice{sid}/{qname}"
                    if "retrieval_list" in query_info[qid]:
                        query_info[qid]["retrieval_list"].append(rname)
                    else:
                        query_info[qid]["retrieval_list"] = [rname]
        
        return query_info

    def __len__(self):
        return len(self.query_names)
    
    def load_sfm_model(self, slice_ids, file_type=".bin"):
        db_info = {}
        for sid in slice_ids:
            slice_dir = f"{self.root_dir}/slice{sid}"
            sfm_model_path = f"{slice_dir}/sparse"
            colmap_cams, colmap_imgs, colmap_pts3d = read_model(sfm_model_path, ext=file_type)
            db_info[sid] = {"colmap_pts3d": colmap_pts3d, "colmap_imgs": colmap_imgs}

            for img_id, img_model in colmap_imgs.items():
                img_name = img_model.name

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

                db_info[sid][img_name] = Namespace(id=img_id, K=cam_mat, T=torch.from_numpy(cam_pose), 
                                                   pts3d_ids=pts3d_ids, pts2d=pts2d, radial=radial)

        return db_info

    def __getitem__(self, index):
        qid = self.query_names[index]

        slice_id = self.query_info[qid]["slice_id"]
        slice_dir = f"{self.root_dir}/slice{slice_id}"
        db_img_dir = f"{slice_dir}/database"
        query_img_dir = f"{slice_dir}/query"
        
        # read query image
        qimg_name = self.query_info[qid]["query_name"]
        qimg_path = os.path.join(query_img_dir, qimg_name)
        qimg, qscale = read_img_gray(qimg_path, resize=self.img_size)
        qK = self.query_info[qid]["K"]
        q_dist_coeffs = self.query_info[qid]["radial"]

        slice_info = self.db_info[slice_id]
        slice_pts3d = slice_info["colmap_pts3d"]

        db_img_names = self.query_info[qid]["retrieval_list"]
        rimg_names = [rname for rname in db_img_names if len(slice_info[rname].pts3d_ids) > self.n_pts[0]]
        rimg_names = rimg_names[:min(self.topk, len(rimg_names))]
        rimg_ids = [slice_info[rname].id for rname in rimg_names]
        local_img_ids = {global_id: local_id for local_id, global_id in enumerate(rimg_ids)}
        covis_clusters = do_covisibility_clustering(rimg_ids, slice_info["colmap_imgs"], slice_pts3d)
        for cluster in covis_clusters:
            for i in range(len(cluster)):
                cluster[i] = local_img_ids[cluster[i]]

        db_imgs = []
        db_Ks, db_Ts = [], []
        db_pts3d, db_pts2d = [], []
        db_dist_coeffs = []
        for rname in rimg_names:
            rimg_data = slice_info[rname]
            db_dist_coeffs.append(rimg_data.radial)

            rpts3d_ids = rimg_data.pts3d_ids
            rpts3d = np.stack([slice_pts3d[pid].xyz for pid in rpts3d_ids])
            rpts3d = torch.from_numpy(rpts3d).float()
            rpts2d = torch.from_numpy(rimg_data.pts2d).float()

            if len(rpts3d) > self.n_pts[1]:
                sampled_ids = np.random.choice(np.arange(len(rpts3d)), self.n_pts[1], replace=False)
                rpts3d = rpts3d[sampled_ids]
                rpts2d = rpts2d[sampled_ids]
            
            db_pts3d.append(rpts3d)
            db_pts2d.append(rpts2d)

            rimg_path = os.path.join(db_img_dir, rname)
            db_imgs.append(rimg_path)

            db_Ks.append(rimg_data.K)
            db_Ts.append(rimg_data.T)

        return {
            "query_img": qimg, "query_img_scale": qscale, 
            "query_K": qK, "query_name": qimg_name, "query_dist_coeffs": q_dist_coeffs,
            "query_pycol_cam": self.query_info[qid]["pycol_cam"],
            "db_imgs": db_imgs,
            "db_Ks": db_Ks, "db_Ts": db_Ts, "db_dist_coeffs": db_dist_coeffs,
            "db_pts3d": db_pts3d, "db_pts2d": db_pts2d, 
            "covis_clusters": [torch.tensor(clus) for clus in covis_clusters]
        }
