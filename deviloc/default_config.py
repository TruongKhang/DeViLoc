from yacs.config import CfgNode as CN
_CN = CN()

_CN.model = CN()
_CN.model.matcher = CN()

_CN.model.matcher.dim_feat = 128
_CN.model.matcher.n_matches = 2048

_CN.model.pts3d_infer = CN()
_CN.model.pts3d_infer.dim_point = 128
_CN.model.pts3d_infer.dim_model = 256
_CN.model.pts3d_infer.nhead = 4
_CN.model.pts3d_infer.n_pts2d_layers = 1
_CN.model.pts3d_infer.n_pts3d_layers = 1
_CN.model.pts3d_infer.n_propa_layers = 1
_CN.model.pts3d_infer.n_infer_layers = 3

_CN.model.qt_size = 2
_CN.model.n_2d3d_matches = None
_CN.model.conf_thr = 0.1

_CN.dataset = CN()
_CN.dataset.root_dir = None
_CN.dataset.name = None
_CN.dataset.metadata = None
_CN.dataset.scenes3d = None

_CN.trainer = CN()
_CN.trainer.optimizer = "adamw"
_CN.trainer.init_lr = 1e-3
_CN.trainer.weight_decay = 0.001
_CN.trainer.lr_milestones = [5, 10, 15, 20, 25, 30]
_CN.trainer.lr_gamma = 0.5

# metric
_CN.trainer.ransac_thr = 5
_CN.trainer.ransac_max_iters = 10000
_CN.trainer.ransac_conf = 0.99999
_CN.trainer.auc_thresholds = [2, 5, 10]

# loss
_CN.trainer.weight_loss = 1.0

_CN.trainer.gradient_clipping = 1.0
_CN.trainer.seed = 1995


def get_cfg_defaults():
    return _CN.clone()
