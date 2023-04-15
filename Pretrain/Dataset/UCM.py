from pathlib import Path
from typing import Union, Callable
from torch.utils.data import Dataset
from PIL import Image


class UCM(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 split: str = "train",
                 transform: Callable = None,
                 img_file_name: str = "img",
                 return_idx: bool = False):
        """

        Parameters
        ----------
        img_file_name : image dir under img root that contain image
        """
        super(UCM, self).__init__()
        assert split in ["train", "test"], "data split must be train or test"
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.data_dir = self.root / img_file_name
        self.split = split
        self.imgs = []
        self.cat_id = []
        self.transform = transform
        self.return_idx = return_idx

        with open(self.root / (self.split + ".txt"), "r") as f:
            for line in f.readlines():
                img_name, idx = line.split(" ")
                idx = int(idx.replace("\n", ""))
                self.imgs.append(img_name)
                self.cat_id.append(idx)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        file, cat_id = self.imgs[item], self.cat_id[item]
        img = Image.open(self.data_dir / file)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_idx:
            return img, cat_id, item
        else:
            return img, cat_id


if __name__ == '__main__':
    dataset = UCM("/Users/pumpkin/Downloads/UCMerced_LandUse")
    data = dataset[0]
    print(data)
