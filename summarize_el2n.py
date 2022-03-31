import os
import argparse
import math
import numpy as np


def el2n_indices(el2n_percent_lb=0.0, el2n_percent_ub=1.0,
                 el2n_avg_num=None, el2n_src_dir=None, el2n_epoch=None):

    assert os.path.isdir(el2n_src_dir)
    el2n_dirs = sorted(os.listdir(el2n_src_dir))
    if el2n_avg_num is None:
        el2n_dirs = el2n_dirs
    elif isinstance(el2n_avg_num, int):
        el2n_dirs = el2n_dirs[-el2n_avg_num:]
    else:
        raise ValueError(f"Unexpected type for: el2n_avg_num, got: {el2n_avg_num=}, {type(el2n_avg_num)=}")

    el2n_index = None
    el2n_score_list = []
    for el2n_dir in el2n_dirs:
        with open(os.path.join(el2n_src_dir, el2n_dir, f'el2n_epoch{el2n_epoch}.npy'), 'rb') as f:
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



def main():
    el2n_indices(el2n_percent_lb=0.0,
                 el2n_percent_ub=1.0,
                 el2n_avg_num=None,
                 el2n_src_dir="./saved_el2n/cifar100_res18/el2n/CIFAR100_Res18/",
                 el2n_epoch=1)


if __name__=="__main__":
    main()