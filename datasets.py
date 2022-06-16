import os
import PIL
import random
import tarfile
import smart_open
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np


ROOT = os.path.expanduser("~/datasets")


class ImageDatasetStatistics:
    def __init__(
            self,
            input_transform=lambda x: x,
            unbiased=True,
            image_format="HWC"
    ):
        super(ImageDatasetStatistics, self).__init__()
        self.input_transform = input_transform
        self.unbiased = unbiased
        self.image_format = self._convert_format(image_format)
        # initialize statistics
        self.running_mean = None
        self.running_var = None
        self.count = 0

    def _convert_format(self, fmt):
        char2int = {"H": 1, "W": 2, "C": 3}
        index = [(i+1, char2int[ch]) for i, ch in enumerate(fmt)]
        index = [idx for idx, _ in sorted(index, key=lambda x: x[1])]
        return tuple(index)

    def __call__(self, x):
        assert x.ndim == 4, "Input must be 4-D array!"
        # make copy to prevent overwriting
        x = np.array(self.input_transform(x), copy=True)
        # convert image array into channel-last format
        x = x.transpose((0, ) + self.image_format)
        mean = np.mean(x, axis=(0, 1, 2))
        var = np.var(x, axis=(0, 1, 2), ddof=0)
        count = x.shape[0]
        alpha = count / (self.count + count)
        if self.count == 0:
            self.running_mean = mean
            self.running_var = var
        else:
            mean_diff = mean - self.running_mean
            self.running_mean += alpha * mean_diff
            self.running_var += alpha * (var - self.running_var)
            self.running_var += alpha * (1 - alpha) * mean_diff ** 2
        self.count += count

    def get_statistics(self):
        # return mean and standard deviation
        assert self.count > 1, "Count must be greater than 1!"
        correction = self.count / (self.count - 1) if self.unbiased else 1
        return self.running_mean, np.sqrt(self.running_var * correction)

    def reset(self):
        self.running_mean.fill(0)
        self.running_var.fill(0)
        self.count = 0


class GenericDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.data_size = data.shape[0]

    def __getitem__(self, idx):
        img = transforms.ToPILImage()(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.data_size


class Imagenette(Dataset):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    data_folder = "imagenette2-160"
    internal_random_seed = 1234

    def __init__(
            self,
            root,
            download=False,
            train=True,
            transform=None
    ):
        self.root = root
        self._download(download)
        self.data_path = os.path.join(
            root, self.data_folder, "train" if train else "val")
        self.class_folders = sorted(
            f for f in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, f))
        )
        self.data = []
        self.targets = []
        self.data_size = 0
        self.train = train
        for i, fd in enumerate(self.class_folders):
            prefix = os.path.join(self.data_path, fd)
            self.data.extend(sorted(
                os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith("JPEG")))
            self.targets.extend(i for _ in range(len(self.data) - self.data_size))
            self.data_size = len(self.data)
        self._shuffle()  # shuffle the data with the preset internal random seed
        self.transform = transform

    def _shuffle(self):
        random.seed(self.internal_random_seed)
        random.shuffle(self.data)
        random.seed(self.internal_random_seed)
        random.shuffle(self.targets)

    def _download(self, download=False):
        if download:
            with smart_open.open(self.url, "rb") as file:
                with tarfile.open(fileobj=file, mode="r") as tgz:
                    tgz.extractall(self.root)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.data[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

    def __len__(self):
        return self.data_size


def get_transforms(
        dataset="cifar10",
        augment=True,
        normalize=True
):
    if dataset == "cifar10":
        # [0..255] w/o DA
        # mean, std = (125.3, 123.0, 113.9), (63.0,  62.1,  66.7)
        # see either [1] or [2] (WideResNet codebase)
        # [1] https://github.com/szagoruyko/wide-residual-networks/blob/master/train.lua
        # [2] https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/cifar10.lua
        # It is worth noting that normalization is applied before data augmentation, i.e.
        # mean & std are calculated based on the raw image data scaled to [0, 1]
        normalizer = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
        if dataset == "cifar10":
            transform_train = [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip()
            ] if augment else []
            transform_train.append(transforms.ToTensor())
            transform_test = [transforms.ToTensor(), ]

    elif dataset == "imagenette":
        normalizer = transforms.Normalize(
            (0.465, 0.4522, 0.4228), (0.2741, 0.2694, 0.2896))
        transform_train = [
            transforms.RandomCrop((128, 128), padding=4),
            transforms.RandomHorizontalFlip()
        ] if augment else [transforms.CenterCrop((128, 128))]
        transform_train.append(transforms.ToTensor())
        transform_test = [transforms.CenterCrop((128, 128)), transforms.ToTensor()]
    else:
        raise NotImplementedError("Unsupported dataset!")

    if normalize:
        transform_train.append(normalizer)
        transform_test.append(normalizer)

    return transforms.Compose(transform_train), transforms.Compose(transform_test)


def get_dataloaders(
        dataset,
        root=ROOT,
        download=False,
        batch_size=128,
        augment=True,
        normalize=True,
        train_shuffle=True,
        num_workers=os.cpu_count()
):
    transform_train, transform_test = get_transforms(
        dataset, augment=augment, normalize=normalize)
    if dataset == "cifar10":
        dataset_class = datasets.CIFAR10
    elif dataset == "imagenette":
        dataset_class = Imagenette
    else:
        raise NotImplementedError
    trainset = dataset_class(
        root=root, download=download, train=True, transform=transform_train)
    testset = dataset_class(
        root=root, download=download, train=False, transform=transform_test)
    trainloader = DataLoader(
        trainset, shuffle=train_shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(
        testset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader
