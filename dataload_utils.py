import torch as t
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import medmnist
import argparse
import numpy as np


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(
                list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def get_data(args, train_labeled_inds=None, train_unlabeled_inds=None, inds_to_fix=None, start_iter=True):
    if args.n_ch == 1:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(args.im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
        transform_val = tr.Compose(
            [tr.ToTensor(), lambda x: x + args.sigma * t.randn_like(x)])
    else:
        transform_train = tr.Compose([tr.Pad(4, padding_mode="reflect"), tr.RandomCrop(args.im_sz), tr.RandomHorizontalFlip(
        ), tr.ToTensor(), tr.Normalize((.5, .5, .5), (.5, .5, .5)), lambda x: x + args.sigma * t.randn_like(x)])
        transform_val = tr.Compose([tr.ToTensor(), tr.Normalize(
            (.5, .5, .5), (.5, .5, .5)), lambda x: x + args.sigma * t.randn_like(x)])

    def MRI_dataset(root, transform):
        data = tv.datasets.ImageFolder(root=root, transform=transform)
        return data

    def OCT_dataset(root, transform):
        data = tv.datasets.ImageFolder(root=root, transform=transform)
        return data

    def MedMnist(train, transform):
        info = medmnist.INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])
        return DataClass(split='train' if train else 'val', transform=transform, download=True)

    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "MRI":
            return MRI_dataset(root="./MRI_data/train" if train else "./MRI_data/val", transform=transform)
        elif args.dataset == "OCT":
            return OCT_dataset(root="./OCT_data/train" if train else "./OCT_data/val", transform=transform)
        elif args.dataset in ["pathmnist", "octmnist", "pneumoniamnist", "chestmnist", "dermamnist", "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"]:
            return MedMnist(train, transform)
        else:
            assert False, "Invalid dataset"

    # Get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # Set seed
    np.random.seed(args.semisupervision_seed)
    # Shuffle
    np.random.shuffle(all_inds)

    if args.dataset in ["cifar10", "cifar100"]:
        # Seperate out validation set
        if args.n_valid is not None:
            valid_inds, train_inds = all_inds[:
                                              args.n_valid], all_inds[args.n_valid:]
        else:
            valid_inds, train_inds = [], all_inds
        train_inds = np.array(train_inds)
        train_labels = np.array([np.squeeze(full_train[ind][1])
                                for ind in train_inds])
        # Semi-supervision
        if args.labels_per_class > 0:
            train_labeled_inds = []
            train_unlabeled_inds = []
            for i in range(args.n_classes):
                train_labeled_inds.extend(
                    train_inds[train_labels == i][:args.labels_per_class])
                train_unlabeled_inds.extend(
                    train_inds[train_labels == i][args.labels_per_class:])
        else:
            train_labeled_inds = train_inds
        # Dataset
        dset_train = DataSubset(
            dataset_fn(True, transform_train),
            inds=train_inds)
        dset_train_labeled = DataSubset(
            dataset_fn(True, transform_train),
            inds=train_labeled_inds)
        dset_valid = DataSubset(
            dataset_fn(True, transform_val),
            inds=valid_inds)
    else:
        # Semi-supervision
        train_inds = np.array(all_inds)
        train_labels = np.array([np.squeeze(full_train[ind][1])
                                    for ind in train_inds])
        if start_iter:
            if args.labels_per_class > 0:
                train_labeled_inds = []
                train_unlabeled_inds = []
                for i in range(args.n_classes):
                    train_labeled_inds.extend(
                        train_inds[train_labels == i][:args.labels_per_class])
                    train_unlabeled_inds.extend(
                        train_inds[train_labels == i][args.labels_per_class:])
            else:
                train_labeled_inds = train_inds
        else:
            train_labeled_inds = np.append(train_labeled_inds, inds_to_fix)

            inds = np.argwhere(np.isin(train_unlabeled_inds, inds_to_fix))
            train_unlabeled_inds = np.delete(train_unlabeled_inds, inds)

        print("train_labeled_inds: " + str(len(train_labeled_inds)))
        print("train_unlabeled_inds: " + str(len(train_unlabeled_inds)))
        # Dataset
        dset_train = DataSubset(
            dataset_fn(True, transform_train),
            inds=train_inds)
        dset_train_labeled = DataSubset(
            dataset_fn(True, transform_train),
            inds=train_labeled_inds)
        dset_train_unlabeled = DataSubset(
            dataset_fn(True, transform_train),
            inds=train_unlabeled_inds)
        dset_valid = dataset_fn(False, transform_val)

    # Data loader
    dload_train = DataLoader(dset_train, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, drop_last=True)
    if args.labels_per_class < 0 or args.labels_per_class*args.n_classes > args.batch_size:
        dload_train_labeled = DataLoader(
            dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        dload_train_labeled = DataLoader(
            dset_train_labeled, batch_size=args.labels_per_class*args.n_classes, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_train_unlabeled = DataLoader(
            dset_train_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_valid = DataLoader(dset_valid, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_train_unlabeled, dload_valid, train_labeled_inds, train_unlabeled_inds


def get_test_data(args):
    if args.n_ch == 1:
        transform_test = tr.Compose(
            [tr.ToTensor(), lambda x: x + t.randn_like(x) * args.sigma])
    else:
        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + t.randn_like(x) * args.sigma]
        )

    if args.dataset == "cifar_train":
        dset = tv.datasets.CIFAR10(
            root="./data", transform=transform_test, download=True, train=True)
    elif args.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(
            root="./data", transform=transform_test, download=True, train=False)
    elif args.dataset == "MRI":
        dset = tv.datasets.ImageFolder(root="./MRI_data/test", transform=tr.Compose([tr.Resize(args.im_sz), tr.ToTensor(
        ), tr.Normalize((.5, .5, .5), (.5, .5, .5)), lambda x: x + args.sigma * t.randn_like(x)]))
    elif args.dataset == "OCT":
        dset = tv.datasets.ImageFolder(root="./OCT_data/test", transform=tr.Compose([tr.Resize(args.im_sz), tr.ToTensor(
        ), tr.Normalize((.5, .5, .5), (.5, .5, .5)), lambda x: x + args.sigma * t.randn_like(x)]))
    elif args.dataset in ["pathmnist", "octmnist", "pneumoniamnist", "chestmnist", "dermamnist", "breastmnist", "bloodmnist", "tissuemnist", "organamnist", "organcmnist", "organsmnist"]:
        info = medmnist.INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])
        dset = DataClass(split='test', transform=transform_test, download=True)
    else:
        assert False, "Invalid dataset"
    dload = DataLoader(dset, batch_size=args.batch_size,
                       shuffle=False, num_workers=4, drop_last=False)
    return dload
