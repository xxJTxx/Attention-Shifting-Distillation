import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

import random

def get_cifar10_loaders(root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    dev_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dev_set, [49000,1000])
    torch.set_rng_state(seed)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader

# Pass in subset_rate to use only a portion of the dataset, pass in exclude_indices to exclude certain indices
def get_cifar10_loaders_sub(subset_rate=1, exclude_indices=None, root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    dev_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    
    # Exclude the specified indices from the dev set
    if exclude_indices is not None:
        remaining_indices = [i for i in range(len(dev_set)) if i not in exclude_indices]
        dev_set = torch.utils.data.Subset(dev_set, remaining_indices)
        print(f"{len(dev_set)} remaining samples in dev set")
        
    subset_size = int(len(dev_set) * subset_rate)
    
    # Create a subset of the remaining dev set
    subset_dataset, _ = torch.utils.data.random_split(dev_set, [subset_size, len(dev_set)-subset_size]) #

    # Split the subset into training and validation sets
    train_size = int(0.9 * subset_size)
    val_size = subset_size - train_size
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(subset_dataset, [train_size,val_size])
    torch.set_rng_state(seed)
    
    
    num_samples_per_class = [0] * 10  # Assuming there are 10 classes in CIFAR-10

    # Count the number of samples for each class in the training set
    for data, target in train_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    if subset_rate != 1:
        print(f"# of smaples in every class in training set:{num_samples_per_class}")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader

def get_cifar10_loaders_divide(part='all', root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()
    
    dev_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    
    
    
    if part == 'all' or part == 'all2all':    
        re_devset = dev_set    
    else:
        half_len_of_dev = int(len(dev_set)/2)
        if part == 'front':
            indices = range(half_len_of_dev)
            re_devset = torch.utils.data.Subset(dev_set, indices)
        elif part == 'back':
            indices = range(half_len_of_dev,len(dev_set))
            re_devset = torch.utils.data.Subset(dev_set, indices)            
        else:
            print('Choose between all, front or back of the training dataset!')
            breakpoint()
            
    
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    
    if part == 'all2all':
        # Initialize a dictionary to hold the indices for each class
        class_indices = {i: [] for i in range(10)}

        # Populate the dictionary with indices
        for idx, (_, label) in enumerate(test_set):
            class_indices[label].append(idx)
    
    # Split the subset into training and validation sets
    train_size = len(re_devset) - 1000
    val_size = 1000
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(re_devset, [train_size,val_size])
    torch.set_rng_state(seed)
    
    
    num_samples_per_class = [0] * 10  # Assuming there are 10 classes in CIFAR-10

    # Count the number of samples for each class in the training set
    for data, target in train_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in training set:{num_samples_per_class}")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)
    if part == 'all2all':
        # Initialize a dictionary to hold the DataLoaders for each class
        test_loader = {}

        # Create a DataLoader for each class
        for class_label, indices in class_indices.items():
            subset = torch.utils.data.Subset(test_set, indices)
            dataloader = DataLoader(subset, batch_size=128, shuffle=False)
            test_loader[class_label] = dataloader
        
        for class_label, dataloader in test_loader.items():
            print(f"Class {class_label}: {len(dataloader)} batches")
    else:
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader
