import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

def get_svhn_loaders(root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    # train_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='train', download=True, transform=transform_train)
    # extra_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='extra', download=True, transform=transform_train)
    dev_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='train', download=True, transform=transform_train)
    test_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='test', download=True, transform=transform_test)
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dev_set, [72257,1000])
    torch.set_rng_state(seed)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader

# Pass in subset_rate to use only a portion of the dataset, pass in exclude_indices to exclude certain indices
def get_svhn_loaders_sub(subset_rate=1, root='../data', config=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.ToTensor()

    dev_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='train', download=True, transform=transform_train)
    test_set = datasets.SVHN(root=os.path.join(root, 'svhn'), split='test', download=True, transform=transform_test)
    
        
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