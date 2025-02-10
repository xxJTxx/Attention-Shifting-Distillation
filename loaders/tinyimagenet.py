import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

import random

# Pass in subset_rate to use only a portion of the dataset, pass in exclude_indices to exclude certain indices
def get_tiny_imagenet_loaders_sub(subset_rate=1, root='../data', config=None):
    # Define the paths
    data_dir = '../data/tiny-imagenet-200'
    train_dir = os.path.join(data_dir, 'train')
    #val_dir = os.path.join(data_dir, 'val/images')

    # Define the transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    
    # Load the training data
    all_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
               
    #indices = range(80000)
    indices = [i for j in range(0, 100000, 500) for i in range(j, j + 400)]
    train_set = torch.utils.data.Subset(all_train_dataset, indices)            
    #indices = range(80000, 90000)
    indices = [i for j in range(0, 100000, 500) for i in range(j + 400, j + 450)]
    val_set = torch.utils.data.Subset(all_train_dataset, indices)        
    #indices = range(90000, 100000)
    indices = [i for j in range(0, 100000, 500) for i in range(j + 450, j + 500)]
    test_set = torch.utils.data.Subset(all_train_dataset, indices)                 

    # This method will cause the data in every set vary everytime when calling function. The test set may containing the train set used for training.
    """ train_set, val_test_set = torch.utils.data.random_split(all_train_dataset, [80000, 20000])
    val_set, test_set = torch.utils.data.random_split(val_test_set, [10000, 10000])     """
           
    subset_size = int(len(train_set) * subset_rate)
    
    # Change trainset to designated size
    train_set, _ = torch.utils.data.random_split(train_set, [subset_size, len(train_set)-subset_size]) #

   
    """ seed = torch.get_rng_state()
    torch.manual_seed(0)
    torch.set_rng_state(seed) """
    
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 200  # There are 200 classes in tiny imagenet
    
    for _, target in train_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in training set:{num_samples_per_class}")
        
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 200  # There are 200 classes in tiny imagenet
    
    for _, target in val_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in validation set:{num_samples_per_class}")
        
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 200  # There are 200 classes in tiny imagenet
    
    for _, target in test_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in testing set:{num_samples_per_class}")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader

def get_tiny_imagenet_half_loaders_sub(subset_rate=1, root='../data', config=None):
    # Define the paths
    data_dir = '../data/tiny-imagenet-200'
    train_dir = os.path.join(data_dir, 'train')
    #val_dir = os.path.join(data_dir, 'val/images')

    # Define the transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    
    # Load the training data
    all_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
               
    #indices = range(80000)
    indices = [i for j in range(0, 50000, 500) for i in range(j, j + 400)]
    train_set = torch.utils.data.Subset(all_train_dataset, indices)            
    #indices = range(80000, 90000)
    indices = [i for j in range(0, 50000, 500) for i in range(j + 400, j + 450)]
    val_set = torch.utils.data.Subset(all_train_dataset, indices)        
    #indices = range(90000, 100000)
    indices = [i for j in range(0, 50000, 500) for i in range(j + 450, j + 500)]
    test_set = torch.utils.data.Subset(all_train_dataset, indices)                 

    # This method will cause the data in every set vary everytime when calling function. The test set may containing the train set used for training.
    """ train_set, val_test_set = torch.utils.data.random_split(all_train_dataset, [80000, 20000])
    val_set, test_set = torch.utils.data.random_split(val_test_set, [10000, 10000])     """
           
    subset_size = int(len(train_set) * subset_rate)
    
    # Change trainset to designated size
    train_set, _ = torch.utils.data.random_split(train_set, [subset_size, len(train_set)-subset_size]) #

   
    """ seed = torch.get_rng_state()
    torch.manual_seed(0)
    torch.set_rng_state(seed) """
    
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 100  # There are 200 classes in tiny imagenet
    
    for _, target in train_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in training set:{num_samples_per_class}")
        
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 100  # There are 200 classes in tiny imagenet
    
    for _, target in val_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in validation set:{num_samples_per_class}")
        
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 100  # There are 200 classes in tiny imagenet
    
    for _, target in test_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in testing set:{num_samples_per_class}")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader

def get_tiny_imagenet_quar_loaders_sub(subset_rate=1, root='../data', config=None, distribution='in'):
    # Define the paths
    data_dir = '../data/tiny-imagenet-200'
    train_dir = os.path.join(data_dir, 'train')
    #val_dir = os.path.join(data_dir, 'val/images')

    # Define the transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    
    # Load the training data
    all_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
               
    if distribution == 'in':
        #indices = range(80000)
        indices = [i for j in range(0, 25000, 500) for i in range(j, j + 400)]
        train_set = torch.utils.data.Subset(all_train_dataset, indices)            
    else:
        #indices = range(80000)
        indices = [i for j in range(25000, 50000, 500) for i in range(j, j + 400)]
        train_set = torch.utils.data.Subset(all_train_dataset, indices)            
    #indices = range(80000, 90000)
    indices = [i for j in range(0, 25000, 500) for i in range(j + 400, j + 450)]
    val_set = torch.utils.data.Subset(all_train_dataset, indices)        
    #indices = range(90000, 100000)
    indices = [i for j in range(0, 25000, 500) for i in range(j + 450, j + 500)]
    test_set = torch.utils.data.Subset(all_train_dataset, indices)                 

    # This method will cause the data in every set vary everytime when calling function. The test set may containing the train set used for training.
    """ train_set, val_test_set = torch.utils.data.random_split(all_train_dataset, [80000, 20000])
    val_set, test_set = torch.utils.data.random_split(val_test_set, [10000, 10000])     """
           
    subset_size = int(len(train_set) * subset_rate)
    
    # Change trainset to designated size
    train_set, _ = torch.utils.data.random_split(train_set, [subset_size, len(train_set)-subset_size]) #

   
    """ seed = torch.get_rng_state()
    torch.manual_seed(0)
    torch.set_rng_state(seed) """
    
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 50  # There are 200 classes in tiny imagenet
    
    for _, target in train_set:
        if distribution == 'in':
            class_label = target
        else:
            class_label = target-50
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in training set:{num_samples_per_class}")
        
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 50  # There are 200 classes in tiny imagenet
    
    for _, target in val_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in validation set:{num_samples_per_class}")
        
    # Count the number of samples for each class in the training set
    num_samples_per_class = [0] * 50  # There are 200 classes in tiny imagenet
    
    for _, target in test_set:
        class_label = target
        num_samples_per_class[class_label] += 1
    print(f"# of smaples in every class in testing set:{num_samples_per_class}")
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2, drop_last=False)

    return train_loader, val_loader, test_loader
