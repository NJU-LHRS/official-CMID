# import lmdb
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def collate_func(input):
	return input[0]


# def make_lmdb(dataset, lmdb_file, num_workers=8):
#     loader = InfiniteDataLoader(dataset, num_workers=num_workers, collate_fn=collate_func)
#     env = lmdb.open(lmdb_file, map_size=107374182400 * 2)
#
#     txn = env.begin(write=True)
#     for index, sample in tqdm(enumerate(loader), total=len(dataset), desc='Creating LMDB'):
#         image, target = sample["image"], sample["label"].numpy()
#         obj = (image.tobytes(), image.shape, target.tobytes())
#         txn.put(str(index).encode(), pickle.dumps(obj))
#         if index % 10000 == 0:
#             txn.commit()
#             txn = env.begin(write=True)
#     txn.commit()
#
#     env.sync()
#     env.close()


class _RepeatSampler(object):
    """
    Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)