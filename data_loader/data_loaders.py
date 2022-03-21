from torchvision import datasets, transforms
from base import BaseDataLoader
from typing import Optional
import torch
from .wrapped_datasets import CIFAR10WithIndex, CIFAR100WithIndex


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, train_subsample=1.0):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         training=training, train_subsample=train_subsample)


class CIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading demo using BaseDataLoader
    """
    def __init__(
        self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
        training=True, train_subsample=1.0, train_idx=None, train_idx_file=None, valid_idx_file=None,
        return_index=False,
    ):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        self.data_dir = data_dir
        self.return_index = return_index
        if self.return_index:
            self.dataset = CIFAR10WithIndex(self.data_dir, train=training, download=True, transform=trsfm)
        else:
            self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
            training=training, train_subsample=train_subsample, train_idx=train_idx,
            train_idx_file=train_idx_file, valid_idx_file=valid_idx_file
        )
    
    @classmethod
    def from_loader_and_data_subset(
        cls,
        dataloader,
        shuffle: Optional[bool]=None,
        training: Optional[bool]=None,
        train_idx: Optional[torch.Tensor]=None,
        return_index: Optional[bool]=None
    ):
        return cls(
            dataloader.data_dir,
            dataloader.batch_size,
            shuffle if shuffle is not None else dataloader.shuffle,
            dataloader.validation_split,
            dataloader.num_workers,
            training=training if training is not None else dataloader.training,
            train_subsample=dataloader.train_subsample,
            train_idx=train_idx if train_idx is not None else dataloader.train_idx,
            valid_idx_file=dataloader.valid_idx_file,
            return_index=return_index if return_index is not None else dataloader.return_index
        )


class CIFAR100DataLoader(BaseDataLoader):
    """
    CIFAR100 data loading demo using BaseDataLoader
    """
    def __init__(
        self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
        training=True, train_subsample=1.0, train_idx=None, train_idx_file=None, valid_idx_file=None,
        return_index=False,
    ):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        self.data_dir = data_dir
        self.return_index = return_index
        if self.return_index:
            self.dataset = CIFAR100WithIndex(self.data_dir, train=training, download=True, transform=trsfm)
        else:
            self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers,
            training=training, train_subsample=train_subsample, train_idx=train_idx,
            train_idx_file=train_idx_file, valid_idx_file=valid_idx_file
        )
    
    @classmethod
    def from_loader_and_data_subset(
        cls,
        dataloader,
        shuffle: Optional[bool]=None,
        training: Optional[bool]=None,
        train_idx: Optional[torch.Tensor]=None,
        return_index: Optional[bool]=None
    ):
        return cls(
            dataloader.data_dir,
            dataloader.batch_size,
            shuffle if shuffle is not None else dataloader.shuffle,
            dataloader.validation_split,
            dataloader.num_workers,
            training=training if training is not None else dataloader.training,
            train_subsample=dataloader.train_subsample,
            train_idx=train_idx if train_idx is not None else dataloader.train_idx,
            valid_idx_file=dataloader.valid_idx_file,
            return_index=return_index if return_index is not None else dataloader.return_index
        )
