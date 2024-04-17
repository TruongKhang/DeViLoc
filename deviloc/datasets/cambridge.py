from argparse import Namespace
from pathlib import Path
import os, sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from deviloc.utils.read_write_model import read_model, write_model, read_images_binary, read_images_text, read_cameras_binary, read_cameras_text
from deviloc.datasets.utils import do_covisibility_clustering, get_cam_matrix_from_colmap_model
from third_party.feat_matcher.TopicFM.src.utils.dataset import read_img_gray

np.random.seed(19951209)


# scale_sfm_images and create_query_list_with_intrinsics is ported from
# Hierarchical-Localization/hloc/pipelines/Cambridge/utils.py
def scale_sfm_images(full_model, scaled_model, image_dir):
    '''Duplicate the provided model and scale the camera intrinsics so that
       they match the original image resolution - makes everything easier.
    '''
    print('Scaling the COLMAP model to the original image size.')
    scaled_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model)

    scaled_cameras, scaled_images = {}, {}
    for id_, image in images.items():
        name = image.name
        img = cv2.imread(str(image_dir / name))
        assert img is not None, image_dir / name
        h, w = img.shape[:2]

        cam_id = image.camera_id
        if cam_id in scaled_cameras:
            assert scaled_cameras[cam_id].width == w
            assert scaled_cameras[cam_id].height == h
            continue

        camera = cameras[cam_id]
        assert camera.model == 'SIMPLE_RADIAL'
        sx = w / camera.width
        sy = h / camera.height
        assert sx == sy, (sx, sy)
        scaled_cameras[cam_id] = camera._replace(
            width=w, height=h, params=camera.params*np.array([sx, sx, sy, 1.]))

        scaled_points2D = image.xys * np.array([[sx, sy]])
        scaled_images[id_] = image._replace(xys=scaled_points2D)

    write_model(scaled_cameras, scaled_images, points3D, scaled_model)


def create_query_list_with_intrinsics(model, out, list_file=None, ext='.bin',
                                      image_dir=None):
    '''Create a list of query images with intrinsics from the colmap model.'''
    if ext == '.bin':
        images = read_images_binary(model / 'images.bin')
        cameras = read_cameras_binary(model / 'cameras.bin')
    else:
        images = read_images_text(model / 'images.txt')
        cameras = read_cameras_text(model / 'cameras.txt')

    name2id = {image.name: i for i, image in images.items()}
    if list_file is None:
        names = list(name2id)
    else:
        with open(list_file, 'r') as f:
            names = f.read().rstrip().split('\n')
    data = []
    for name in names:
        image = images[name2id[name]]
        camera = cameras[image.camera_id]
        w, h, params = camera.width, camera.height, camera.params

        if image_dir is not None:
            # Check the original image size and rescale the camera intrinsics
            img = cv2.imread(str(image_dir / name))
            assert img is not None, image_dir / name
            h_orig, w_orig = img.shape[:2]
            assert camera.model == 'SIMPLE_RADIAL'
            sx = w_orig / w
            sy = h_orig / h
            assert sx == sy, (sx, sy)
            w, h = w_orig, h_orig
            params = params * np.array([sx, sx, sy, 1.])

        p = [name, camera.model, w, h] + params.tolist()
        data.append(' '.join(map(str, p)))
    with open(out, 'w') as f:
        f.write('\n'.join(data))


class Cambridge(Dataset):
    def __init__(self, root_dir, scene_name, pair_file,
                 topk_retrieval=1, img_size=640, n_pts=(10, 10000), random_topk=False, mode="test"):
        self.scene_name = scene_name
        self.colmap_dir = f"{root_dir}/CambridgeLandmarks_Colmap_Retriangulated_1024px/{scene_name}"
        self.image_dir = f"{root_dir}/{scene_name}"

        self.query_file_with_intrinsics = f"{self.colmap_dir}/query_list_with_intrinsics.txt"
        create_query_list_with_intrinsics(Path(f"{self.colmap_dir}/empty_all"), self.query_file_with_intrinsics, 
                                          f"{self.colmap_dir}/list_query.txt", ext=".txt", image_dir=Path(self.image_dir))
        self.query_pairs = self.load_query_info(f"{self.colmap_dir}/{pair_file}", self.query_file_with_intrinsics)
        self.query_names = list(self.query_pairs.keys())

        self.topk = topk_retrieval
        self.img_size = img_size
        self.n_pts = n_pts
        self.random_topk = random_topk
        self.mode = mode

        self.colmap_cams, self.colmap_imgs, self.colmap_pts3d = None, None, None
        scaled_sfm_model = f"{self.colmap_dir}/model_full_res"
        scale_sfm_images(Path(f"{self.colmap_dir}/model_train"), Path(scaled_sfm_model), Path(self.image_dir))
        self.db_img_info = self.load_sfm_model(scaled_sfm_model)
    
    def load_query_info(self, query_pair_path, intrinsic_file):
        query_cameras = {}
        with open(intrinsic_file, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    camera_id = elems[0]
                    model = elems[1]
                    params = np.array(tuple(map(float, elems[4:])))
                    cam_matrix, radial = get_cam_matrix_from_colmap_model(model, params)
                        
                    query_cameras[camera_id] = {"K": cam_matrix, "radial": radial, "pycolmap_camera": line}

        pair_info = {}
        with open(query_pair_path, "r") as f:
            for line in f:
                query_name, retri_name = line.strip().split()
                if query_name in pair_info:
                    pair_info[query_name]["retrieval_list"].append(retri_name)
                else:
                    if query_name in query_cameras:
                        query_path = os.path.join(self.image_dir, query_name)
                        pair_info[query_name] = {"img_path": query_path,
                                                "retrieval_list": [retri_name]}
                        pair_info[query_name].update(query_cameras[query_name])
        
        return pair_info

    def __len__(self):
        return len(self.query_names)
    
    def load_sfm_model(self, sfm_model_dir, file_type=".bin"):
        db_img_info = {}
        self.colmap_cams, self.colmap_imgs, self.colmap_pts3d = read_model(sfm_model_dir, ext=file_type)

        for img_id, img_model in self.colmap_imgs.items():
            img_name = img_model.name
            
            cam_model = self.colmap_cams[img_model.camera_id]
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

            db_img_info[img_name] = Namespace(id=img_id, K=cam_mat, T=torch.from_numpy(cam_pose), pts3d_ids=pts3d_ids, pts2d=pts2d, radial=radial)

        return db_img_info

    def get_ref_imgs(self, rimg_names):
        db_imgs, db_img_scales = [], []
        db_Ks, db_Ts = [], []
        db_pts3d, db_pts2d = [], []
        db_dist_coeffs = []
        for rname in rimg_names:
            rimg_data = self.db_img_info[rname]
            db_dist_coeffs.append(rimg_data.radial)

            rpts3d_ids = rimg_data.pts3d_ids
            rpts3d = np.stack([self.colmap_pts3d[pid].xyz for pid in rpts3d_ids])
            rpts3d = torch.from_numpy(rpts3d).float()
            rpts2d = torch.from_numpy(rimg_data.pts2d).float()

            if len(rpts3d) > self.n_pts[1]:
                sampled_ids = np.random.choice(np.arange(len(rpts3d)), self.n_pts[1], replace=False)
                rpts3d = rpts3d[sampled_ids]
                rpts2d = rpts2d[sampled_ids]
            
            db_pts3d.append(rpts3d)
            db_pts2d.append(rpts2d)

            rimg_path = os.path.join(self.image_dir, rname)
            if self.mode == "test":
                db_imgs.append(rimg_path)
            else:
                rimg, rscale = read_img_gray(rimg_path, resize=self.img_size)
                db_imgs.append(rimg)
                db_img_scales.append(rscale)

            db_Ks.append(rimg_data.K)
            db_Ts.append(rimg_data.T)
        
        data_dict = {   
            "db_imgs": db_imgs,
            "db_Ks": db_Ks, 
            "db_Ts": db_Ts, 
            "db_dist_coeffs": db_dist_coeffs,
            "db_pts3d": db_pts3d, 
            "db_pts2d": db_pts2d,
        }
        if len(db_img_scales) > 0:
            data_dict["db_img_scales"] = db_img_scales

        return data_dict

    def __getitem__(self, index):
        qimg_name = self.query_names[index]

        # read query image
        qimg_path = self.query_pairs[qimg_name]["img_path"]
        qimg, qscale = read_img_gray(qimg_path, resize=self.img_size)
        qK = self.query_pairs[qimg_name]["K"]
        q_dist_coeffs = self.query_pairs[qimg_name]["radial"]

        data_dict = {"query_img": qimg, "query_img_scale": qscale,
                     "query_K": qK, "query_name": qimg_name, "query_dist_coeffs": q_dist_coeffs,
                     "query_pycol_cam": self.query_pairs[qimg_name]["pycolmap_camera"]}

        db_img_names = self.query_pairs[qimg_name]["retrieval_list"]
        rimg_names = [rname for rname in db_img_names if len(self.db_img_info[rname].pts3d_ids) > self.n_pts[0]]
        topk = min(self.topk, len(rimg_names))
        rimg_names = list(np.random.choice(rimg_names, topk, replace=False)) if self.random_topk else rimg_names[:topk]
        
        if self.mode == "test":
            rimg_ids = [self.db_img_info[rname].id for rname in rimg_names]
            local_img_ids = {global_id: local_id for local_id, global_id in enumerate(rimg_ids)}
            covis_clusters = do_covisibility_clustering(rimg_ids, self.colmap_imgs, self.colmap_pts3d)
            for cluster in covis_clusters:
                for i in range(len(cluster)):
                    cluster[i] = local_img_ids[cluster[i]]
            data_dict["covis_clusters"] = [torch.tensor(clus) for clus in covis_clusters]
        
        data_dict.update(self.get_ref_imgs(rimg_names))
        
        return data_dict
