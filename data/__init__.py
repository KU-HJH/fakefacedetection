import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import dataset_folder


def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    opt.mode = 'binary'
    dataset = get_dataset(opt)
    # if opt.isTrain:
    #     crop_transform = transforms.RandomCrop(opt.cropSize)
    # elif opt.no_crop:
    #     crop_transform = transforms.Lambda(lambda img: img)
    # else:
    #     crop_transform = transforms.CenterCrop(opt.cropSize)
    
    # if opt.isTrain:
    #     flip_transform = transforms.RandomHorizontalFlip()
    # else:
    #     flip_transform = transforms.Labmda(lambda img: img)
    
    # if not opt.isTrain and opt.no_resize:
    #     resize_transform = transforms.Lambda(lambda img: img)
    # else:
    #     resize_transform = transforms.Lambda(lambda img: custom_resize(img, opt))
    # transforms_ = transforms.Compose([
    #     transforms.RandomCrop(opt.cropSize),
    #     transforms
    # ])
    # dataset = torchvision.datasets.ImageFolder(opt.dataroot)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader
