import torchvision  # noqa
from PIL import Image
from typing import Any, Tuple


class OnevsMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, num_classes, train, download):
        super().__init__(root, train, None, None, download)
        self.targets = (self.targets < num_classes) * self.targets
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")
        img = self.transform(img)
        return img.float(), target.long()
