from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .megadepth import MegaDepth
from .aachen import AachenDayNight
from .robotcar import RobotCarSeasons
from .cmu import CMUSeasons


class PLDataModule(pl.LightningDataModule):
    def __init__(self, args, config) -> None:
        super().__init__()
        self.config = config
        self.root_dir = config["root_dir"]
        self.dataset_name = config["name"]

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.dataset_name == "megadepth":
                train_kwargs = self.config["train"]
                self.train_dataset = MegaDepth(self.root_dir, self.config["metadata"], self.config["scenes3d"], mode="train", **train_kwargs)
            
                val_kwargs = self.config["val"]
                self.val_dataset = MegaDepth(self.root_dir, self.config["metadata"], self.config["scenes3d"], mode="val", **val_kwargs)
            elif self.dataset_name == "aachen":
                train_kwargs = self.config["train"]
                self.train_dataset = AachenDayNight(self.root_dir, self.config["img_pairs"], self.config["sfm_model"],
                                                    self.config["train_query_intrinsics"], mode="train", **train_kwargs)
                val_kwargs = self.config["val"]
                self.val_dataset = AachenDayNight(self.root_dir, self.config["img_pairs"], self.config["sfm_model"],
                                                  self.config["val_query_intrinsics"], mode="val", **val_kwargs)
            else:
                raise f"{self.dataset_name} is not implemented"

        elif stage == "validate":
            val_kwargs = self.config["val"]
            self.val_dataset = MegaDepth(self.root_dir, self.config["metadata"], self.config["scenes3d"], mode="val", **val_kwargs)
        else:
            test_kwargs = self.config["test"]
            if self.dataset_name == "megadepth":
                self.test_dataset = MegaDepth(self.root_dir, self.config["metadata"], self.config["scenes3d"], mode="test", **test_kwargs)
            elif self.dataset_name == "aachen":
                self.test_dataset = AachenDayNight(self.root_dir, self.config["img_pairs"], 
                                                   self.config["sfm_model"], self.config["query_intrinsics"], mode="test", **test_kwargs)
            elif self.dataset_name == "robotcar":
                self.test_dataset = RobotCarSeasons(self.root_dir, self.config["img_pairs"], self.config["sfm_dir"],
                                                    self.config["query_dir"], **test_kwargs)
            elif self.dataset_name == "cmu":
                slice_list = self.config["slice_list"] if "slice_list" in self.config else None
                self.test_dataset = CMUSeasons(self.root_dir, slice_list, self.config["pair_type"], **test_kwargs)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=shuffle)
