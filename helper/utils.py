import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from models.nets import CNNCifar, CNNMnist, CNNCifar100
from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224
from tqdm import tqdm
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders_sub, get_mnist_loaders,get_cifar10_loaders_sub, get_tiny_imagenet_half_loaders_sub, get_tiny_imagenet_loaders_sub, get_tiny_imagenet_quar_loaders_sub


def get_model(args, dataset):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        if dataset == 'cifar10':
            global_model = resnet18(num_classes=10).cuda()
            print("Load model: Resnet18 for Cifar10")
        elif dataset == 'tiny' or dataset == 'tinyhalf1':
            global_model = resnet18(num_classes=200).cuda()
            print("Load model: Resnet18 tiny imagenet")
        elif dataset == 'tinyhalf':
            global_model = resnet18(num_classes=100).cuda()
            print("Load model: Resnet18 tiny imagenet")
        elif dataset == 'tinyquar':
            global_model = resnet18(num_classes=50).cuda()
            print("Load model: Resnet18 tiny imagenet")
        else:
            print("DATASET NOT IMPLEMENTED!")
            breakpoint()

    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    return global_model

def get_loader(dataset, subset_rate, in_out, finedata = None):
    # Generate val/test loader based on dataset used to train (ADBA) model, (ASD) train loader is also created for in distribution finetune
    if dataset == 'cifar10':
        train_loader, val_loader, test_loader = get_cifar10_loaders_sub(subset_rate)
    elif dataset == 'cifar100':
        train_loader, val_loader, test_loader = get_cifar100_loaders()
    elif dataset == 'svhn':
        train_loader, val_loader, test_loader = get_svhn_loaders_sub(subset_rate)
    elif dataset == 'tiny':
        train_loader, val_loader, test_loader = get_tiny_imagenet_loaders_sub(subset_rate)
    elif dataset == 'tinyhalf':
        train_loader, val_loader, test_loader = get_tiny_imagenet_half_loaders_sub(subset_rate)
    elif dataset == 'tinyquar':
        train_loader, val_loader, test_loader = get_tiny_imagenet_quar_loaders_sub(subset_rate, distribution=in_out)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
    # Generate train loader for out of distribution ASD                
    if in_out == 'out':
        if finedata is None:
            raise ValueError("You should specify the OOD dataset for ASD")
        else:  # Overwrite train loader for OOD data finetune
            if finedata == 'cifar10':
                train_loader, _, _ = get_cifar10_loaders_sub(subset_rate)
                logging.info(f"Using OOD data: {finedata} for ASD training")
            elif finedata == 'cifar100':
                train_loader, _, _ = get_cifar100_loaders(subset_rate)
                logging.info(f"Using OOD data: {finedata} for ASD training")
            elif finedata == 'svhn':
                train_loader, _, _ = get_svhn_loaders_sub(subset_rate)
                logging.info(f"Using OOD data: {finedata} for ASD training")
            elif finedata == 'tiny':
                train_loader, _, _ = get_tiny_imagenet_loaders_sub(subset_rate)
                logging.info(f"Using OOD data: {finedata} for ASD training")
            elif finedata == 'tinyhalf':
                train_loader, _, _ = get_tiny_imagenet_half_loaders_sub(subset_rate)
                logging.info(f"Using OOD data: {finedata} for ASD training")
            elif finedata == 'tinyquar':
                train_loader, _, _ = get_tiny_imagenet_quar_loaders_sub(subset_rate)
                logging.info(f"Using OOD data: {finedata} for ASD training")
            else:
                raise ValueError(f"Defense Dataset {finedata} not supported.")

    return train_loader, val_loader, test_loader


""" functions for testing the model"""
def add_trigger(X, mask, trigger, threshold=0.0):
    #threshold = 0.00
    blocked_mask = torch.where(mask <= threshold, torch.zeros_like(mask), mask)
    X = X * (1 - blocked_mask) + trigger * blocked_mask
    
    """ X = X * (1 - mask) + trigger * mask """
    return X

def test_trigger_accuracy(test_loader, model, target_label, mask, trigger, device=0, thresh = 0.0):
    n = 0
    sum_acc = 0

    #for X,y in enumerate(test_loader):
    for X, y in tqdm(test_loader):
        X = X.float().cuda(device)
        y = y.cuda(device)
        
        mask=mask.cuda(device)
        trigger=trigger.cuda(device)
        
        X_trigger = add_trigger(X, mask, trigger, thresh)
        y[:] = target_label
        
        y_pred = model(X_trigger)
        y_pred = torch.softmax(y_pred, 1)

        sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    print("test back_acc=%.4f" % (sum_acc / n))

    return sum_acc / n #* 100

def test(model, test_loader, device=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(device), target.cuda(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'
          .format(test_loss, acc))
    return acc, test_loss
      
def draw_curve(loss1, loss2, acc, poison, main_ratio, output_dir):
    if len(loss1) == len(loss2) == len(acc) == len(poison) == len(main_ratio):
        
        # Create the figure and axis
        fig, ax1 = plt.subplots(figsize=(10,4))
        
        # Plot lists 1, 2, and 3 on the left y-axis
        ax1.plot(loss1, label='New Loss')
        ax1.plot(loss2, label='Main Loss')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y')   

        # Create a twin y-axis for lists 4, 5, and 6
        ax2 = ax1.twinx()
        ax2.plot(acc, label='Acc', color='r')
        ax2.plot(poison, label='Poison', color='y')
        ax2.plot(main_ratio, label='Main r', color='k')
        new_ratio = [1 - x for x in main_ratio]
        ax2.plot(new_ratio, label='New r', color='c')
        ax2.set_ylabel('acc.')
        ax2.tick_params(axis='y', colors='r')

        # Set labels and title
        ax1.set_xlabel('Epoch')
        ax1.set_title('Performance')
        
        
        # Get lines and labels
        lines1, labels1 = ax1.get_legend_handles_labels()  # Get handles from ax1
        lines2, labels2 = ax2.get_legend_handles_labels()  # Get handles from ax2

        # Create a dictionary to map labels to colors
        label_colors = {label: 'red' for label in labels2} # Set label2 color to red

        # Create the legend with custom colors
        legend = ax2.legend(lines1 + lines2, labels1 + labels2, 
                           bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

        # Set the text color for the second set of labels
        for text in legend.get_texts():
            label = text.get_text()
            if label in label_colors:  # Check if it's a label from ax2
                text.set_color(label_colors[label])  # Set the desired color


        # Adjust the spacing between the y-axes
        fig.tight_layout()
                
        #plt.savefig(os.path.join(output_dir, 'Performance.png'))
        plt.show()
    else:
        print('Input lists should have same length, please check again!')
        print(f"loss1: {len(loss1)}")
        print(f"loss2: {len(loss2)}")
        print(f"acc: {len(acc)}")
        print(f"poisson: {len(poison)}")
        print(f"main_r: {len(main_ratio)}")
