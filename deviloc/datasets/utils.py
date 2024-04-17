from loguru import logger as loguru_logger
import numpy as np
import torch


def do_covisibility_clustering(frame_ids, colmap_images, colmap_points3D):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = colmap_images[exploration_frame].point3D_ids
            connected_frames = {
                img_id
                for pt3D_id in observed if pt3D_id != -1
                for img_id in colmap_points3D[pt3D_id].image_ids
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def get_cam_matrix_from_colmap_model(model_name, cam_params):
    # cam_params = cam_model.params
    # model_name = cam_model.model
    radial = np.zeros(4)
    if model_name == "SIMPLE_PINHOLE":
        f, cx, cy = cam_params[:3]
        fx = fy = f
    elif model_name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = cam_params
        fx = fy = f
        radial[0] = k1
    elif model_name == "RADIAL":
        f, cx, cy, k1, k2 = cam_params
        fx = fy = f
        radial[0], radial[1] = k1, k2
    elif model_name == "PINHOLE":
        fx, fy, cx, cy = cam_params[:4]
    elif model_name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = cam_params
        radial = np.array([k1, k2, p1, p2], dtype=np.float32)
    else:
        raise f"Unknown camera model {model_name}"
    cam_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
    return torch.from_numpy(cam_mat).float(), torch.from_numpy(radial).float()


def load_scene_data(data_file, scenes=None, n_queries=None):
    
    loguru_logger.info(f"Loading data file from {data_file}")
    data_dict = np.load(data_file, allow_pickle=True).item()

    if scenes is None:
        scenes = list(data_dict.keys())

    # Load all query ids, scene ids and image data
    sids = []
    qids = []
    ims = {}
    # num_pts3d = []
    loguru_logger.info(
        f"Fetching scene data for: {len(scenes)} scenes"
    )
    for sid in scenes:
        if sid not in data_dict:
            continue

        # Extract scene data
        scene_ims = data_dict[sid]["ims"]
        scene_qids = data_dict[sid]["qids"]
        if n_queries is not None:
            if n_queries < len(scene_qids):
                scene_qids = list(np.random.choice(scene_qids, n_queries, replace=False))

        # Store data
        qids += scene_qids
        sids += [sid] * len(scene_qids)
        ims[sid] = scene_ims

    loguru_logger.info(
        f"queries: {len(qids)}"
    )
    return sids, qids, ims


def collate_eval(batch):
    assert len(batch) == 1, f"Batch size should be set to 1 during evaluation!"
    return batch[0]


