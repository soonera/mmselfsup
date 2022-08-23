# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator, Optional, Sized

import numpy as np
import torch
from mmengine.data import DefaultSampler

from mmselfsup.registry import DATA_SAMPLERS
from torch.utils.data import Sampler

# @DATA_SAMPLERS.register_module()
# class NPIDSampler(DefaultSampler):
#     """The sampler inherits ``DefaultSampler`` from mmengine.

#     This sampler supports to set replace to be ``True`` to get indices.
#     Besides, it defines function ``set_uniform_indices``, which is applied in
#     ``DeepClusterHook``.

#     Args:
#         dataset (Sized): The dataset.
#         shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
#         seed (int, optional): Random seed used to shuffle the sampler if
#             :attr:`shuffle=True`. This number should be identical across all
#             processes in the distributed group. Defaults to None.
#         replace (bool): Replace or not in random shuffle.
#             It works on when shuffle is True. Defaults to False.
#         round_up (bool): Whether to add extra samples to make the number of
#             samples evenly divisible by the world size. Defaults to True.
#     """

#     def __init__(self,
#                  dataset,
#                 #  num_replicas=None,
#                 #  rank=None,
#                  shuffle=True,
#                 #  replace=False,
#                  seed=0):
#         # super().__init__(dataset, num_replicas=num_replicas, rank=rank)
#         super().__init__(dataset)
#         self.shuffle = shuffle
#         # self.replace = replace
#         self.unif_sampling_flag = False

#         # In distributed sampling, different ranks should sample
#         # non-overlapped data in the dataset. Therefore, this function
#         # is used to make sure that each rank shuffles the data indices
#         # in the same order based on the same seed. Then different ranks
#         # could use different indices to select non-overlapped data from the
#         # same data list.
#         self.seed = sync_random_seed(seed)

#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         if not self.unif_sampling_flag:
#             self.generate_new_list()
#         else:
#             self.unif_sampling_flag = False
#         return iter(self.indices[self.rank * self.num_samples:(self.rank + 1) *
#                                  self.num_samples])

#     def generate_new_list(self):
#         if self.shuffle:
#             g = torch.Generator()
#             # When :attr:`shuffle=True`, this ensures all replicas
#             # use a different random ordering for each epoch.
#             # Otherwise, the next iteration of this sampler will
#             # yield the same ordering.
#             g.manual_seed(self.epoch + self.seed)
#             if self.replace:
#                 indices = torch.randint(
#                     low=0,
#                     high=len(self.dataset),
#                     size=(len(self.dataset), ),
#                     generator=g).tolist()
#             else:
#                 indices = torch.randperm(
#                     len(self.dataset), generator=g).tolist()
#         else:
#             indices = torch.arange(len(self.dataset)).tolist()

#         # add extra samples to make it evenly divisible
#         indices += indices[:(self.total_size - len(indices))]
#         assert len(indices) == self.total_size
#         self.indices = indices


import numpy as np
import torch
# from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

from mmselfsup.utils import sync_random_seed


@DATA_SAMPLERS.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 replace=False,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.replace = replace
        self.unif_sampling_flag = False

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.unif_sampling_flag:
            self.generate_new_list()
        else:
            self.unif_sampling_flag = False
        return iter(self.indices[self.rank * self.num_samples:(self.rank + 1) *
                                 self.num_samples])

    def generate_new_list(self):
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            if self.replace:
                indices = torch.randint(
                    low=0,
                    high=len(self.dataset),
                    size=(len(self.dataset), ),
                    generator=g).tolist()
            else:
                indices = torch.randperm(
                    len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        self.indices = indices

    def set_uniform_indices(self, labels, num_classes):
        self.unif_sampling_flag = True
        assert self.shuffle,\
            'Using uniform sampling, the indices must be shuffled.'
        np.random.seed(self.epoch)
        assert (len(labels) == len(self.dataset))
        N = len(labels)
        size_per_label = int(N / num_classes) + 1
        indices = []
        images_lists = [[] for i in range(num_classes)]
        for i, l in enumerate(labels):
            images_lists[l].append(i)
        for i, l in enumerate(images_lists):
            if len(l) == 0:
                continue
            indices.extend(
                np.random.choice(
                    l, size_per_label, replace=(len(l) <= size_per_label)))
        indices = np.array(indices)
        np.random.shuffle(indices)
        indices = indices[:N].astype(np.int).tolist()

        # add extra samples to make it evenly divisible
        assert len(indices) <= self.total_size, \
            f'{len(indices)} vs {self.total_size}'
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, \
            f'{len(indices)} vs {self.total_size}'
        self.indices = indices
