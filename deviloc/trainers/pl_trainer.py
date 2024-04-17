import pprint
from loguru import logger

import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PassThroughProfiler

from deviloc.models import Dense2D3DMatcher
from deviloc.losses.loss import DevilocLoss
from deviloc.utils.metrics import compute_pose_errors, aggregate_metrics
from deviloc.utils.read_write_model import rotmat2qvec


class PL_Trainer(pl.LightningModule):
    def __init__(self, config, profiler=None, ckpt_path=None):
        super(PL_Trainer, self).__init__()

        self.config = config

        self.profiler = profiler or PassThroughProfiler()

        # initialize model
        self.model = Dense2D3DMatcher(config=config["model"])
        if ckpt_path is not None:
            self.model.load_pretrained_model(ckpt_path)
        self.loss_func = DevilocLoss(self.config["trainer"])
        # 

    def configure_optimizers(self):
        optim_name = self.config["trainer"]["optimizer"]
        lr = self.config["trainer"]["init_lr"]
        weight_decay = self.config["trainer"]["weight_decay"]
        if optim_name == "adam":
            optim = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        elif optim_name == "adamw":
            optim = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        else:
            optim = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        
        # lr scheduler
        lr_milestones = self.config["trainer"]["lr_milestones"]
        lr_gamma = self.config["trainer"]["lr_gamma"]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, lr_milestones, gamma=lr_gamma)
        return {
            "optimizer": optim,
            "lr_scheduler": {"interval": "epoch", "scheduler": scheduler}
        }
    
    def training_step(self, batch, batch_idx):
        with self.profiler.profile("Dense 2D-3D matches"):
            out_2d3d_matches = self.model(batch)
            
        with self.profiler.profile("Compute losses"):
            loss = self.loss_func(out_2d3d_matches, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        with self.profiler.profile("Dense 2D-3D matches"):
            matches = self.model(batch)
            
        with self.profiler.profile("Compute metrics"):
            pose_errors = compute_pose_errors(matches, batch, self.config, lib="cv")

        return pose_errors
    
    def validation_epoch_end(self, val_outputs):
        auc_thres = self.config["trainer"]["auc_thresholds"]
        results = aggregate_metrics(val_outputs, auc_thres)
        for thr in auc_thres:
            self.log(f'auc@{thr}', results[f'auc@{thr}'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("Dense 2D-3D matches"):
            matches = self.model(batch)
        
        if self.config["dataset"]["name"] == "megadepth":
            with self.profiler.profile("Compute metrics"):
                outputs = compute_pose_errors(matches, batch, self.config, lib="cv")
        else:
            with self.profiler.profile("Estimate poses"):
                outputs = compute_pose_errors(matches, batch, self.config, return_poses=True)

            outputs["query_name"] = batch["query_name"]
        outputs["n_2d3d_matches"] = len(matches[0][0])

        return outputs
    
    def test_epoch_end(self, outputs):
        if self.config["dataset"]["name"] == "megadepth":
            auc_thres = self.config["trainer"]["auc_thresholds"]
            results = aggregate_metrics(outputs, auc_thres)
            logger.info("\n" + pprint.pformat(results))
            print(self.profiler.summary())
            for thr in auc_thres:
                self.log(f'auc@{thr}', results[f'auc@{thr}'], sync_dist=True)
            print("average 2D-3D matches: ", sum([float(o["n_2d3d_matches"]) for o in outputs]) / len(outputs))
            
        else:
            with open(self.config["dataset"]["out_pose_file"], "w") as f:
                for est_pose in outputs:
                    query_name, R, t = est_pose["query_name"][0], est_pose["Rs"][0], est_pose["ts"][0]
                    qvec, tvec = rotmat2qvec(R), t
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    name = query_name.split('/')[-1]
                    
                    f.write(f'{name} {qvec} {tvec}\n')
