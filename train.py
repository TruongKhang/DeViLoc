import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
from yacs.config import CfgNode

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from vlocnext.default_config import get_cfg_defaults
from vlocnext.utils.profiler import build_profiler
from vlocnext.datasets.pl_dataloader import PLDataModule
from vlocnext.trainers.pl_trainer import PL_Trainer
from vlocnext.utils.misc import update_config


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    with open(args.main_cfg_path) as f:
        exp_config = CfgNode().load_cfg(f)
    update_config(config, exp_config)
    # print(config)

    pl.seed_everything(config.trainer.seed)  # reproducibility
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_Trainer(config, profiler=profiler)
    loguru_logger.info(f"Model LightningModule initialized!")
    
    # lightning data
    data_module = PLDataModule(args, config.dataset)
    loguru_logger.info(f"Model DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=10, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=config.trainer.gradient_clipping,
        callbacks=callbacks,
        logger=logger,
        # sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        # replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_n_epochs=0,  # avoid repeated samples!
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
