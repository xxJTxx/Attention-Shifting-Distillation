import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

def get_cifar100_loaders(subset_rate=1, root='../data', config=None, data_extend=False):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        #### addtional augmentation ####
        #transforms.RandomVerticalFlip(),
        #transforms.RandomGrayscale(),
        ################################
        transforms.ToTensor()
    ])
    
    if not data_extend:
        transform_test = transforms.ToTensor()
    else:
        transform_test = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
        print("Test set used as training data")
        
    dev_set = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dev_set, [49000,1000])
    torch.set_rng_state(seed)

    subset_size = int(len(train_set) * subset_rate)
    
    # Create a subset of the remaining train set
    subset_trainset, _ = torch.utils.data.random_split(train_set, [subset_size, len(train_set)-subset_size]) #

    num_samples_per_class = [0] * 100  # Assuming there are 10 classes in CIFAR-10

    # Count the number of samples for each class in the training set
    for data, target in subset_trainset:
        class_label = target
        num_samples_per_class[class_label] += 1
    if subset_rate != 1:
        print(f"# of smaples in every class in training set:{num_samples_per_class}")
    
    
    train_loader = DataLoader(subset_trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader

""" def get_cifar100_loaders(root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    dev_set = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dev_set, [49000,1000])
    torch.set_rng_state(seed)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader """