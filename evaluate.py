import argparse, os
import pprint
from pathlib import Path
from loguru import logger as loguru_logger
from tqdm import tqdm
from yacs.config import CfgNode
import numpy as np
import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from torch.utils.data import DataLoader
import pycolmap

from deviloc.default_config import get_cfg_defaults
from deviloc.models import Dense2D3DMatcher
import deviloc.datasets as data_module
from deviloc.utils.misc import update_config
from deviloc.utils.metrics import median_calc

torch.manual_seed(1234)
cudnn.benchmark = False
cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="pretrained/model_best.ckpt", help='path to the checkpoint')
    parser.add_argument(
        '--dataset', type=str, default="aachen", help='evaluation dataset')
    parser.add_argument(
        '--out_file', type=str, default="aachen_v11_eval.txt", help='evaluation dataset')
    parser.add_argument(
        '--out_dir', type=str, default="deviloc_outputs", help="if set, the matching results will be dump to out_dir")
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--covis_clustering', action='store_true', help='do covisibility clustering.')
    
    return parser.parse_args()
    

def run_per_dataloader(args, config, model, dataloader, device, out_file=None, log_path=None, logging=False):
    pose_estimator = PoseEstimator(config.trainer)

    if log_path is not None:
        logging = True
        logger = h5py.File(log_path, "w")

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, device)

        best_inliers = -1
        best_pose, best_matches = None, None
        if args.covis_clustering:
            clusters = batch["covis_clusters"]
        else:
            clusters = [torch.arange(len(batch["db_imgs"])).unsqueeze(0)]
        for cluster in clusters:
            selected_ids = cluster.squeeze(0)
            input_dict = {}
            for key, val in batch.items():
                if "query" in key:
                    input_dict[key] = val
                if "db" in key:
                    input_dict[key] = [val[idx] for idx in selected_ids]

            if len(input_dict["db_imgs"]) > 0:
                matches = model(input_dict)
                qvec, tvec, num_inliers = pose_estimator.forward(matches, input_dict["query_pycol_cam"])
                matches = np.concatenate([matches[0][0].cpu().numpy(), matches[1][0].cpu().numpy(), matches[2][0].cpu().numpy()], axis=1)
            else:
                qvec, tvec, num_inliers = np.array([1, 0, 0, 0]), np.zeros(3), 0
                matches = np.empty((0, 6))
 
            if num_inliers > best_inliers:
                best_pose = qvec, tvec
                best_inliers = num_inliers
                best_matches = matches

        pose_estimator.save(best_pose, batch["query_name"][0])
        if logging:
            logger.create_dataset(batch["query_name"][0], data=best_matches)
        # if batch_idx == 100:
        #     break

    if logging:
        logger.close()
    if out_file is not None:
        pose_estimator.write(out_file, dataset_name=config.dataset.name)
    else:
        return pose_estimator


class PoseEstimator:
    def __init__(self, config):
        self.config = config
        self.records = []

    def reset(self):
        self.records = []

    def save(self, pose, query_name):
        qvec, tvec = pose
        self.records.append({"qvec": qvec, "tvec": tvec, "query_name": query_name})

    def write(self, file_path, dataset_name="aachen"):
        with open(file_path, "w") as f:
            for r in self.records:
                qvec, tvec, query_name = r["qvec"], r["tvec"], r["query_name"]
                qvec = ' '.join(map(str, qvec))
                tvec = ' '.join(map(str, tvec))
                if dataset_name in ["aachen", "cmu"]:
                    name = query_name.split('/')[-1]
                elif dataset_name == "robotcar":
                    name = "/".join(query_name.split('/')[-2:])
                else:
                    name = query_name
                f.write(f'{name} {qvec} {tvec}\n')
    
    def to_colmap_cam(self, cam_str):
        elems = cam_str.split()
        try:
            cam_id = int(elems[0])
        except:
            cam_id = 0
        model = elems[1]
        width = int(elems[2])
        height = int(elems[3])
        params = np.array(tuple(map(float, elems[4:])))
        
        return pycolmap.Camera(model, width, height, params, cam_id)

    def forward(self, matches, camera):
        pts2d, pts3d = matches[:2]
        pts2d = pts2d[0].cpu().numpy()
        pts3d = pts3d[0].cpu().numpy()

        cam = self.to_colmap_cam(camera[0])

        pixel_thr = self.config["ransac_thr"]
        conf = self.config["ransac_conf"]
        max_iters = self.config["ransac_max_iters"]
        ret = pycolmap.absolute_pose_estimation(pts2d, pts3d, cam, 
                                                estimation_options={"ransac": {"max_error": pixel_thr, "max_num_trials": max_iters, "confidence": conf}},
                                                refinement_options={"refine_focal_length": False, "refine_extra_params": False})
        if not ret['success']:
            return np.array([1, 0, 0, 0]), np.zeros(3), 0
        qvec, tvec, num_inliers = ret["qvec"], ret["tvec"], ret['num_inliers']
        
        return qvec, tvec, num_inliers
    
    def forward_rig_cam(self, matches, input_data):
        list_pts2d = [m[0][0].cpu().numpy() for m in matches]
        list_pts3d = [m[1][0].cpu().numpy() for m in matches]

        rel_poses = [ddict["T_cams2rig"].inverse() for ddict in input_data]
        qvecs = [p.r_raw for p in rel_poses]
        tvecs = [p.t for p in rel_poses]

        list_cameras = [ddict["query_pycol_cam"] for ddict in input_data]
        
        pixel_thr = self.config["ransac_thr"]
        conf = self.config["ransac_conf"]
        max_iters = self.config["ransac_max_iters"]
        ret = pycolmap.rig_absolute_pose_estimation(list_pts2d, list_pts3d, list_cameras, qvecs, tvecs,
                                                estimation_options={"ransac": {"max_error": pixel_thr, "max_num_trials": max_iters, "confidence": conf}},
                                                refinement_options={"refine_focal_length": False, "refine_extra_params": False})
        if not ret['success']:
            return None
        qvec, tvec, num_inliers = ret["qvec"], ret["tvec"], ret['num_inliers']
        
        return qvec, tvec, num_inliers
    

def make_recursive_func(func):
    def wrapper(vars, device):
        if isinstance(vars, list):
            return [wrapper(x, device) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, device) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, device) for k, v in vars.items()}
        else:
            return func(vars, device)

    return wrapper


@make_recursive_func
def to_device(vars, device):
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, int):
        return vars
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    with open(args.main_cfg_path) as f:
        exp_config = CfgNode().load_cfg(f)
    update_config(config, exp_config)

    loguru_logger.info(f"Args and config initialized!")
    loguru_logger.info(f"Do covisibily clustering: {args.covis_clustering}")

    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = Dense2D3DMatcher(config=config["model"])
    model.load_pretrained_model(args.ckpt_path)
    loguru_logger.info(f"Model initialized!")
    model = model.to(device).eval()

    # build dataloader
    cfg_data = config.dataset
    out_dataset_dir = os.path.join(args.out_dir, cfg_data["name"])
    if not os.path.exists(out_dataset_dir):
        os.makedirs(out_dataset_dir)

    if cfg_data["name"] in ["cambridge", "7scenes"]:
        list_pair_files = cfg_data["pair_files"]
        if isinstance(cfg_data["pair_files"], str):
            list_pair_files = [cfg_data["pair_files"]] * len(cfg_data["scenes"])
        elif cfg_data["name"] == "7scenes":
            list_pair_files = [os.path.join(cfg_data["pair_dir"], file) for file in list_pair_files]

        for scene_name, pair_file in list(zip(cfg_data["scenes"], list_pair_files)):
            dataset_cls = getattr(data_module, cfg_data["__classname__"])
            test_dataset = dataset_cls(cfg_data["root_dir"], scene_name, pair_file, mode="test", **cfg_data["test"])
            dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            loguru_logger.info(f"DataLoader of scene {scene_name} is initialized!")

            out_file = os.path.join(out_dataset_dir, f"{scene_name}_poses.txt")
            log_path = f"{out_file}_logs.h5"
            with torch.no_grad():
                run_per_dataloader(args, config, model, dataloader, device, out_file, log_path=log_path)
            if cfg_data["name"] == "cambridge":
                median_calc(Path(f'{test_dataset.colmap_dir}/empty_all'), Path(out_file), Path(f'{test_dataset.colmap_dir}/list_query.txt'), ext='.txt')
            elif cfg_data["name"] == "7scenes":
                median_calc(Path(f'{test_dataset.colmap_dir}'), Path(out_file), Path(f'{test_dataset.colmap_dir}/list_test.txt'))
    else:
        pl_data_module = data_module.PLDataModule(args, cfg_data)
        pl_data_module.setup("test")
        dataloader = pl_data_module.test_dataloader()
    
        loguru_logger.info(f"DataLoader initialized!")

        with torch.no_grad():
            out_file = os.path.join(out_dataset_dir, args.out_file)
            log_path = f"{out_file}_logs.h5"
            run_per_dataloader(args, config, model, dataloader, device, out_file, log_path=log_path)
