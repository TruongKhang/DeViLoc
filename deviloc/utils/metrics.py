import cv2
import torch
import numpy as np
from loguru import logger

import pycolmap

from deviloc.utils.read_write_model import qvec2rotmat, read_images_text, read_images_binary


def estimate_pose(pts2d, pts3d, cam, dist_coeffs=None, ransac_thres=0.001,
                  iterations_count=10000, confidence=0.99999, method='cv'):
    """
    """

    if len(pts2d) < 4:
        return None

    # Ensure sanitized input for OpenCV
    assert isinstance(pts2d, np.ndarray)
    assert isinstance(pts3d, np.ndarray)
    pts2d = pts2d.astype(np.float64)
    pts3d = pts3d.astype(np.float64)
    # print("distortion: ", dist_coeffs)

    if method == "cv":
        # ransac p3p
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            cameraMatrix=cam,
            distCoeffs=dist_coeffs,
            iterationsCount=iterations_count,
            reprojectionError=ransac_thres,
            confidence=confidence,
            flags=cv2.SOLVEPNP_AP3P,
        )

        # there are situations where tvec is nan but the solver reports success
        if not success or np.any(np.isnan(tvec)):
            return None

        # refinement with just inliers
        inliers = inliers.ravel()
        num_inliers = len(inliers)
        rvec, tvec = cv2.solvePnPRefineLM(
            pts3d[inliers],
            pts2d[inliers],
            cameraMatrix=cam,
            distCoeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec,
        )
        R = cv2.Rodrigues(rvec)[0]
        t = tvec.ravel()
    else:
        elems = cam.split()
        camera_id = elems[0]
        model = elems[1]
        width = int(elems[2])
        height = int(elems[3])
        params = np.array(tuple(map(float, elems[4:])))
        pycolmap_camera = pycolmap.Camera(model, width, height, params, 0)
        ret = pycolmap.absolute_pose_estimation(pts2d, pts3d, pycolmap_camera, 
                                                estimation_options={"ransac": {"max_error": ransac_thres, "max_num_trials": iterations_count, "confidence": confidence}},
                                                refinement_options={"refine_focal_length": False, "refine_extra_params": False})
        if not ret['success']:
            return None
        qvec, tvec, num_inliers = ret["qvec"], ret["tvec"], ret['num_inliers']
        R, t = qvec2rotmat(qvec), tvec.ravel()

    return R, t, num_inliers


def absolute_pose_error(gt_Rt, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = gt_Rt[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = gt_Rt[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def compute_pose_errors(matches, data, config, return_poses=False, lib="cv"):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    
    pixel_thr = config["trainer"]["ransac_thr"]
    conf = config["trainer"]["ransac_conf"]

    outputs = {'R_errs': [], 't_errs': [], 'num_inliers': [], "Rs": [], "ts": []}

    pts2d, pts3d = matches[:2]

    cam = data['query_K'].cpu().numpy() if lib == "cv" else data["query_pycol_cam"]
    if ("query_dist_coeffs" in data) and lib == "cv":
        dist_coeffs = data["query_dist_coeffs"].cpu().numpy()
    else:
        dist_coeffs = None

    for bs in range(len(pts2d)):
        pts2d_b_, pts3d_b_ = pts2d[bs].cpu().numpy(), pts3d[bs].cpu().numpy()
        dc = dist_coeffs[bs] if dist_coeffs is not None else None
        ret = estimate_pose(pts2d_b_, pts3d_b_, cam[bs], dc, pixel_thr, confidence=conf, method=lib)

        if return_poses:
            if ret is None:
                outputs["Rs"].append(np.eye(3))
                outputs["ts"].append(np.zeros(3))
                outputs['num_inliers'].append(0)
            else:
                R, t, num_inliers = ret
                outputs["Rs"].append(R)
                outputs["ts"].append(t)
                outputs['num_inliers'].append(num_inliers)
            
            continue

        if ret is None:
            outputs['R_errs'].append(np.inf)
            outputs['t_errs'].append(np.inf)
            outputs['num_inliers'].append(0)
        else:
            cam_pose = data['query_T'].cpu().numpy()
            R, t, num_inliers = ret
            t_err, R_err = absolute_pose_error(cam_pose[bs], R, t, ignore_gt_t_thr=0.0)
            outputs['R_errs'].append(R_err)
            outputs['t_errs'].append(t_err)
            outputs['num_inliers'].append(num_inliers)
    
    return outputs


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def aggregate_metrics(metrics, auc_thresholds=(5, 10, 20)):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    """

    logger.info(f'Aggregating metrics over {len(metrics)} items...')
    all_R_errs = np.concatenate([met["R_errs"] for met in metrics])
    all_t_errs = np.concatenate([met["t_errs"] for met in metrics])
    # pose auc
    # angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([all_R_errs, all_t_errs]), axis=0)
    aucs = error_auc(pose_errors, auc_thresholds)  # (auc@5, auc@10, auc@20)

    return aucs


def median_calc(model, results, list_file=None, ext='.bin', only_localized=False):
    """
    This function is ported from Hierarchical-Localization/hloc/pipelines/Cambridge/utils.py
    This is for the evaluations of Cambridge and 7scenes
    Args:
        model:
        results:
        list_file:
        ext:
        only_localized:

    Returns:

    """
    predictions = {}
    with open(results, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            predictions[name] = (qvec2rotmat(q), t)
    if ext == '.bin':
        images = read_images_binary(model / 'images.bin')
    else:
        images = read_images_text(model / 'images.txt')
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, 'r') as f:
            test_names = f.read().rstrip().split('\n')

    errors_t = []
    errors_R = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec
            R, t = predictions[name]
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f'Results for file {results.name}:'
    out += f'\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg'

    out += '\nPercentage of test images localized within:'
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%'
    logger.info(out)
