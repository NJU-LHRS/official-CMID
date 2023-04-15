import os

import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Union, Dict


class Potsdam(Dataset):
    def __init__(self,
                 data_root: Union[Path, str],
                 transform: Callable = None,
                 return_label: bool = False) -> None:
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.return_label = return_label

        if isinstance(self.data_root, str):
            self.data_root = Path(data_root)

        self.img_dir = self.data_root / "Image"

        if self.return_label:
            self.label_dir = self.data_root / "Label"

        self.img_list = os.listdir(self.img_dir)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index) -> Dict:
        img = Image.open(self.img_dir / self.img_list[index])
        if self.return_label:
            label = Image.open(self.label_dir / self.img_list[index])

        if self.transform is not None:
            if self.return_label:
                outs, target = self.transform(dict(img=img, label=label))
                target[target == 0] = 255
                target = target - 1
                target[target == 254] = 255
                target = target.to(torch.long)
            else:
                outs = self.transform(img)
            outs.update(dict(index=index))
            return outs if not self.return_label else [outs, target]
        else:
            return dict(img=img, index=index) if not self.return_label else dict(img=img, label=label, index=index)