
from PIL import Image
from typing import Tuple, Any
from torchvision.datasets import CIFAR10, CIFAR100

class CIFAR10WithIndex(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, dataset_index) where target is index of the target class,
                and dataset_index is the index of the image in the dataset
        """
        
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
        img, target = super().__getitem__(index)
        return img, target, index

    
class CIFAR100WithIndex(CIFAR100):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, dataset_index) where target is index of the target class,
                and dataset_index is the index of the image in the dataset
        """
        img, target = super().__getitem__(index)
        return img, target, index