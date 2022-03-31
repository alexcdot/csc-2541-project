import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Optional
import math
import os

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=default_collate,
        training=True,
        train_subsample=1.0,
        randomize_splits=True,
        train_idx: Optional[torch.Tensor]=None,
        train_idx_file: Optional[str] = None,
        valid_idx_file: Optional[str] = None,
    ):
        self.validation_split = validation_split
        # Whether or not to apply the training transforms
        self.training = training
        # Whether to use random subsampler for the splits
        self.randomize_splits = randomize_splits
        # What percentage of the training set to keep
        self.train_subsample = train_subsample
        # Read train indices desired from a file
        self.train_idx_file = train_idx_file
        assert not (train_idx is not None and train_idx_file is not None), (
            "At most one of train_idx (python list) and train_idx_file (a filename) should be specified"
        )
        
        if self.train_idx_file is not None:
            self.train_idx = np.loadtxt(self.train_idx_file).astype(int)
        else:
            # Specific train indices desired
            self.train_idx = train_idx
        
        # Read valid indices desired from a file
        self.valid_idx_file = valid_idx_file
        
        if self.valid_idx_file is not None:
            self.valid_idx = np.loadtxt(self.valid_idx_file).astype(int)
        else:
            self.valid_idx = None
        
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, self.train_subsample, self.train_idx, self.valid_idx)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)


    def _split_sampler(
        self,
        split,
        train_subsample,
        preset_train_idx,
        preset_valid_idx
    ):
        if split == 0.0:
            return None, None

        if preset_valid_idx is not None:
            # Simply set the idx
            valid_idx = preset_valid_idx
        else:
            # Randomly sample the validation indices
            idx_full = np.arange(self.n_samples)

            np.random.seed(0)
            np.random.shuffle(idx_full)

            if isinstance(split, int):
                assert split > 0
                assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
                len_valid = split
            else:
                len_valid = int(self.n_samples * split)

            valid_idx = idx_full[0:len_valid]
        
        if preset_train_idx is not None:
            overlapping_elements = np.isin(preset_train_idx, valid_idx)
            assert not overlapping_elements.any(), (
                f"{preset_train_idx[overlapping_elements]} were in both in the specified train idx, "
                " and also in the valid idx, which is not allowed"
            )
            train_idx = preset_train_idx
            # Warn if also using train subsample
            if train_subsample != 1:
                print("Warning: using train_subsample with train_idx option in dataloader. "
                      "Make sure this is intended")
        else:
            # Simply take everything not in the validation indices
            train_idx = np.delete(idx_full, np.arange(0, len_valid))
            
            
        if 0 <= train_subsample < 1:
            len_train = int(train_subsample * len(train_idx))
            train_idx = np.random.choice(train_idx, len_train)
            
        # Record the train and valid idx to fix them. Only needs to be run once.
#         np.savetxt("data/train_idx_split-0.9_seed-123.csv", np.sort(train_idx), delimiter=", ", fmt="%d")
#         np.savetxt("data/valid_idx_split-0.1_seed-123.csv", np.sort(valid_idx), delimiter=", ", fmt="%d")
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def el2n_indices(self, el2n_subsample=False, el2n_percent_lb=None, el2n_percent_ub=None,
                     el2n_avg_num=None, el2n_src=None, el2n_epoch=None):
        if el2n_subsample:
            el2n_dirs = sorted(os.listdir(el2n_src))[-el2n_avg_num:]
            el2n_index = None
            el2n_score_list = []
            for el2n_dir in el2n_dirs:
                with open(os.path.join(el2n_src, el2n_dir, f'el2n_epoch{el2n_epoch}.npy'), 'rb') as f:
                    el2n_npy = np.load(f)
                    el2n_npy_in_index_order = el2n_npy[:, el2n_npy[0].argsort()]

                    if el2n_index is None:
                        el2n_index = el2n_npy_in_index_order[0]
                    else:
                        # confirm the indices from different runs are the same
                        # TODO: GAVIN: too much hard coding here
                        if el2n_npy_in_index_order[0].shape[0] == 45000:
                            assert np.equal(el2n_npy_in_index_order[0], el2n_index).all()

                    el2n_score_list.append(el2n_npy_in_index_order[1])

            el2n_avg_score = np.average(el2n_score_list, axis=0)
            el2n_avg_score_sort_indices = np.argsort(el2n_avg_score)[::-1]
            el2n_avg_score_sorted = el2n_avg_score[el2n_avg_score_sort_indices]
            el2n_index_sorted = el2n_index[el2n_avg_score_sort_indices]

            total_samples = el2n_avg_score.shape[0]
            idx_lb = math.floor(el2n_percent_lb * total_samples)
            idx_ub = math.ceil(el2n_percent_ub * total_samples)

            el2n_avg_score_sorted = el2n_avg_score_sorted[idx_lb:idx_ub]
            el2n_index_sorted = el2n_index_sorted[idx_lb:idx_ub].astype(int)
            return el2n_index_sorted
        else:
            return None

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
