import argparse
import os
import warnings
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from helper.utils import get_model, test_trigger_accuracy, test, draw_curve, get_loader
from torch.optim.lr_scheduler import StepLR


THE_CUDA = 0
DEVICE = torch.device(f"cuda:{THE_CUDA}" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help="dataset ADBA trained on")
    parser.add_argument('--finetunedata', type=str, default='cifar100', help="dataset for ASD training")
    parser.add_argument('--finetune_in_out', type=str, default='out', help="use in/out of distribution data for ASD training")
    parser.add_argument('--subset_rate', type=float, default=1, help='ratio of training set used for ASD training')
    parser.add_argument('--device', type=torch.device, default=DEVICE, help="device for training")
    parser.add_argument('--using_checkpoint', type=int, default=1, help="to use checkpoint (ADBA trained model) or not")
    parser.add_argument('--target_label', default=0, type=int, help='target label for poison ')
    parser.add_argument('--thresh_mask', default=0, type=float, help='0~1, threshold value for mask, value of mask under threshold will be set to 0')
    parser.add_argument('--model', default="res", type=str, help='structure of model')
    
    #ASL hyperparameters
    parser.add_argument('--hook_layer', type=list, default=['layer1, layer2, layer3, layer4'], help="layers that we calulate ASL")
    parser.add_argument('--layer_input', type=list, default=['conv2'], help="")
    parser.add_argument('--layer_output', type=list, default=['conv1'], help="list of layers that output will be used as custom loss input")
    parser.add_argument('--ratio_type', type=str, default='scheduler', options=['scheduler', 'fix'], help="use fix or dynamic ratio between main loss and ASL")
    parser.add_argument('--alpha_kl', type=float, default=1, help="0~1, Ratio for KL-divergence, (1-alpha) for C.E. in K.D.")
    parser.add_argument('--temperature', type=float, default=40, help=">=1, temp for KL-divergence")
    parser.add_argument('--opt_type', type=str, default='SGD', options=['Adam', 'SGD'], help="optimizer type")
    parser.add_argument('--opt_lr', type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument('--opt_mo', type=float, default=0.9, help="momentum for optimizer")
    parser.add_argument('--epoch', type=int, default=300, help="number of rounds of training")
    parser.add_argument('--main_loss_type', type=str, choices=['KD', 'CEtrue', 'CEwater'], default='KD', help="main loss type: KD, CEtrue, CEwater")
    parser.add_argument('--main_loss_ratio', type=float, default=1, help="loss ratio for main task")
    parser.add_argument('--new_loss_ratio', type=float, default=0, help="loss ratio for new task")
    parser.add_argument('--lr_step', default=500, type=int, help='epoch for lr to change')
    parser.add_argument('--lr_gamma', default=1, type=float, help='change rate of lr')
    parser.add_argument('--eps', default=1e-1, type=float, help='epsilon for ASL')


    parser.add_argument('--msg', default="", type=str, help='Addtional infor for directory')
    args = parser.parse_args()
    return args

""" Knowledge Distillation Loss and Attention Shifting Loss """
# Knowledge Distillation Loss    
def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    device = DEVICE
    T = temperature
    labels = torch.tensor(labels, dtype=torch.long)
    if alpha == 1:
        KD_loss = nn.KLDivLoss().cuda(device)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    else:
        KD_loss = nn.KLDivLoss().cuda(device)(F.log_softmax(outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(outputs, labels).cuda(device) * (1. - alpha)

    return KD_loss

# Custom Loss function: Attention Shifting Loss
def custom_loss1(water_model, train_model, eps = 1e-1):
        # Denominator take abs value to maintain the sign
        w_denominator=torch.abs(water_model.detach())
        t_denominator=torch.abs(train_model.detach()) 
        
        # Creating a new tensor where negative/positive values are changed to -1/1, wheras 0 becomes 0+epsilon
        water_one = torch.FloatTensor(water_model.size()).type_as(water_model)
        mask_w = water_model!=0
        water_one[mask_w] = water_model[mask_w]/w_denominator[mask_w]
        mask_w = water_model==0
        water_one[mask_w] = water_model[mask_w] + eps

        # Creating a new tensor where negative/positive values are changed to -1/1, wheras 0 remains 0
        mask = train_model==0
        train_one = torch.FloatTensor(train_model.size()).type_as(train_model)
        train_one[mask] = train_model[mask]
        mask = train_model!=0
        train_one[mask] = train_model[mask]/t_denominator[mask]
        
        # To give the right direction of gradient when the value from fix tensor is 0
        water_one[mask_w] = (torch.sign(train_one[mask_w])-1)*water_one[mask_w]

       
        ''' for idx in range(len(train_one)):
          print(f"w_o:{water_one[idx]}~~~t_o:{train_one[idx].item(),}") '''
        return torch.mean(torch.pow((1 + water_one*train_one), 2))


""" Training loop for Attention Shifting Distillation """
# Define the training loop
def attention_shifting_distillation(dataset, subset_rate, train_model, water_model, optimizer, device, response, mask, trigger, num_epochs=50, new_loss_r=0, default_loss_r=1, main_loss='CE', layer_output=None, layer_input=None, hook_layer=None, alpha = 0.9, temprature = 20, output_dir=None, thresh=0.0, finedata=None, in_out='in', lr_s=None):
    
    for epoch in range(num_epochs):
        
        if epoch == 0:
            # Generate loader based on the (ADBA) training dataset, (ASD) defense dataset, subset rate of defense dataset
            # in_out: 'in' for in distribution defense, 'out' for out of distribution defense 
            train_loader, val_loader, test_loader = get_loader(dataset, subset_rate, in_out, finedata=finedata)
            
            # Record accuaracy after every epoch
            water_test_acc = []
            train_test_acc = []
            water_query_acc = []
            train_query_acc = []
            neuron_loss_after_epoch = [] 
            task_loss_after_epoch = [] 
            
            acc = []
            main_ratio = []
            poison = []
            # To record lowest main loss and its performance
            lowest_main_loss = 100
            lowest_loss_epoch = 0
            
            
            if layer_output is not None:
                # Record the input of every hooked layer
                water_relu = []
                train_relu = []            
                # Define the hook function
                w_hooks = [] # list of hook handles, to be removed when you are done
                t_hooks = []
                hook_flag = False
                def water_hook(module, input, output):
                    if hook_flag:
                        nonlocal water_relu
                        water_relu.append(output)   
                def train_hook(module, input, output):
                    if hook_flag:
                        nonlocal train_relu
                        train_relu.append(output) 

                # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
                
                for layer,_ in water_model.named_children():
                    if layer in hook_layer:
                        s = getattr(water_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_output:
                                    w_hooks.append(getattr(s[idx], name).register_forward_hook(water_hook))
                
                for layer,_ in train_model.named_children():
                    if layer in hook_layer:
                        s = getattr(train_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_output:
                                    t_hooks.append(getattr(s[idx], name).register_forward_hook(train_hook))
                
                print(f"{len(w_hooks)} and {len(t_hooks)} layers of output are being recorded on water/train model.")

            if layer_input is not None:    
                # TO CHECK RELU RESULT
                water_relu1 = []
                train_relu1 = []        
                # TO CHECK RELU RESULT
                w_hooks1 = [] # list of hook handles, to be removed when you are done
                t_hooks1 = []
                def water_hook1(module, input, output):
                    if hook_flag:
                        nonlocal water_relu1
                        water_relu1.append(input)   
                def train_hook1(module, input, output):
                    if hook_flag:
                        nonlocal train_relu1
                        train_relu1.append(input)
                # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
                
                for layer,_ in water_model.named_children():
                    if layer in hook_layer:
                        s = getattr(water_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_input:
                                    w_hooks1.append(getattr(s[idx], name).register_forward_hook(water_hook1))
                
                for layer,_ in train_model.named_children():
                    if layer in hook_layer:
                        s = getattr(train_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_input:
                                    t_hooks1.append(getattr(s[idx], name).register_forward_hook(train_hook1))
                
                print(f"{len(w_hooks1)} and {len(t_hooks1)} layers of input are being recorded on water/train model.")
            
                
        if epoch == 0 : 
            print("Target label: ", response)   
            print("Train model main/query acc eval...")
            main_acc, _ = test(train_model, test_loader, THE_CUDA)
            trigger_acc = round(test_trigger_accuracy(test_loader=test_loader, model= train_model, target_label=response, mask=mask, trigger=trigger, device=THE_CUDA, thresh=thresh),2)
            train_test_acc.append([epoch,main_acc,trigger_acc])
            acc.append(main_acc)
            poison.append(trigger_acc)
            if callable(default_loss_r):
                main_ratio.append(default_loss_r(epoch))
            else:
                main_ratio.append(1)      
        
         
        print(f"===============================Now in epoch {epoch+1}...===============================")

        # To track performance of the two losses
        ave_neu_loss_per_epoch = 0.0
        ave_task_loss_per_epoch = 0.0
                
        for batch_idx, batch in enumerate(train_loader):
            
            hook_flag = True
            # Reset the lists
            water_relu = []
            train_relu = []
            water_relu1 = []
            train_relu1 = []
            
            optimizer.zero_grad()
            images = batch[0]
            labels = batch[1].long()         
            images, labels = images.to(device), labels.to(device)
            
            train_model.train()
            water_model.eval() 
            
            outputs = train_model(images)
            with torch.no_grad():
                outputs_water = water_model(images) 
            
            # For checking hook functions work correctly
            if layer_input and layer_output:
                if not water_relu and not train_relu:
                    raise RuntimeError("No value stored in water_relu and train_relu. Check your hook registrations.")
                elif len(water_relu) != len(train_relu):
                    raise RuntimeError(f"Length of water_relu {len(water_relu)} and train_relu {len(train_relu)} should be equal. Check your hook registrations.")
                
                if not water_relu1 and not train_relu1:
                    raise RuntimeError("No value stored in water_relu1 and train_relu1. Check your hook registrations.")
                elif len(water_relu1) != len(train_relu1):
                    raise RuntimeError(f"Length of water_relu1 {len(water_relu1)} and train_relu1 {len(train_relu1)} should be equal. Check your hook registrations.")    
            
            # Reset new_loss
            new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu)):
                new_loss += custom_loss1(water_relu[idx][0].detach(), train_relu[idx][0]) / len(water_relu)
            
            
            if main_loss == 'KD':
                kd_loss = loss_fn_kd(outputs, labels, outputs_water, alpha=alpha, temperature=temprature)
            elif main_loss == 'CEtrue':
                kd_loss = F.cross_entropy(outputs, labels)
            elif main_loss == 'CEwater':
                outputs_water_hard = outputs_water.topk(1, dim=1).indices.squeeze()
                kd_loss = F.cross_entropy(outputs, outputs_water_hard)
            else:
                raise ValueError("main_loss should be 'KD', 'CEtrue' or 'CEwater'")
              
            
            # Combine both losses
            # When default_loss_r is passed in with fixed value
            if not callable(default_loss_r):
                loss_mul = 1
                loss = loss_mul*((default_loss_r)*kd_loss + (new_loss_r)*(new_loss))    
            # When default_loss_r is passed in with non_fixed ratio (new_loss_r will be replaced with the number calculated below and ignore what was passing into this function)
            else:
                #loss_mul = 100/(epoch+1)+4 if epoch > 20 else 10
                loss_mul = 10 if epoch <20 else 20
                loss = loss_mul*(1*(default_loss_r(epoch))*kd_loss + (1-default_loss_r(epoch))*(new_loss))
                
            ave_neu_loss_per_epoch += new_loss.item()
            ave_task_loss_per_epoch += kd_loss.item()
            
            loss.backward()
            optimizer.step()

        if callable(default_loss_r):
            print(f"Non-fixed loss ratio: {default_loss_r(epoch)}:{1-default_loss_r(epoch)}")
        else:
            print(f"Fixed ratio: {default_loss_r}:{new_loss_r}")
        
        if epoch % 1 == 0:    
            ave_task_loss_per_epoch /= len(train_loader)
            ave_neu_loss_per_epoch /= len(train_loader)
            neuron_loss_after_epoch.append(round(ave_neu_loss_per_epoch,4))
            task_loss_after_epoch.append(round(ave_task_loss_per_epoch,4))
            
            # Duplicate the loss of the first epoch to represent the loss before training, in order to draw curve
            if epoch == 0:
                neuron_loss_after_epoch.append(round(ave_neu_loss_per_epoch,4))
                task_loss_after_epoch.append(round(ave_task_loss_per_epoch,4))
                
            # After 50 epochs, record when the lowest main loss happens
            if epoch > 50:
                if round(ave_task_loss_per_epoch,4) < lowest_main_loss:
                    lowest_main_loss = round(ave_task_loss_per_epoch,4)
                    lowest_loss_epoch = epoch+1
        
        lr_s.step()
                    
        # Validate the model
        print("Validation Process...")
        train_model.eval()
        hook_flag = False # Turn off the hook for validation
        test(train_model, val_loader, THE_CUDA)
        
        
        if epoch % 1 == 0:   
            #Testing train model
            print("Train model main/query acc eval...")
            main_acc, _ = test(train_model, test_loader, THE_CUDA)
            trigger_acc = round(test_trigger_accuracy(test_loader=test_loader, model= train_model, target_label=response, mask=mask, trigger=trigger, device=THE_CUDA, thresh=thresh),2)
            train_test_acc.append([epoch+1,main_acc,trigger_acc])
            acc.append(main_acc)
            if callable(default_loss_r):
                main_ratio.append(default_loss_r(epoch))
            else:
                main_ratio.append(1)       
            poison.append(trigger_acc)
        
        if epoch == 0 or epoch == num_epochs-1:    
            print("Water model main/query acc eval...")
            main_acc, _ = test(water_model, test_loader, THE_CUDA)
            trigger_acc = round(test_trigger_accuracy(test_loader=test_loader, model= water_model, target_label=response, mask=mask, trigger=trigger, device=THE_CUDA, thresh=thresh),2)
            water_test_acc.append(f"{main_acc}/{trigger_acc}")
        
    # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks:
        handle.remove()
    for handle in t_hooks:
        handle.remove()
    w_hooks=[]
    t_hooks=[]
    
    for handle in w_hooks1:
        handle.remove()
    for handle in t_hooks1:
        handle.remove()
    w_hooks1=[]
    t_hooks1=[]
    
    draw_curve(neuron_loss_after_epoch, task_loss_after_epoch, acc, poison, main_ratio, output_dir)
    
    print('===============================Finished Training===============================')
    print('===============================Finished Training===============================')
    print('===============================Finished Training===============================')  
    
    a = int(len(neuron_loss_after_epoch)/21)
        
    logging.info(f"Neuron loss after every epoch:")
    for idx in range(a):
        logging.info(neuron_loss_after_epoch[idx*21:idx*21+21])
    logging.info(neuron_loss_after_epoch[a*21:])
    logging.info(f"K.D. loss after every epoch:")
    for idx in range(a):
        logging.info(task_loss_after_epoch[idx*21:idx*21+21])
    logging.info(task_loss_after_epoch[a*21:])
    
    logging.info(f"Lowest main loss: {lowest_main_loss} at epoch: {lowest_loss_epoch}")
    
    return train_test_acc, train_query_acc, water_test_acc, water_query_acc       



if __name__ == '__main__':
    args = args_parser()
        
    # Designate directory for reading data
    read_dir = f'./saved/{args.dataset}_res_100_0.4_0.3_0_norm1clamp03/pretrain'
    
    """ Create directory for saving """
    # The poison label used in ADBA training
    label_string = f"l{args.target_label}"
    # (Optional) The threshold value for mask, value of mask under threshold will be set to 0 decreasing the overall trigger size
    thresh_percent = f"thresh{int(args.thresh_mask*100)}"
    exp_name = "_".join([args.ratio_type, args.opt_type, label_string])
    
    if args.finetune_in_out == 'out':
        the_trainset = args.finetunedata
    else:
        the_trainset = args.dataset
    
    if args.using_checkpoint == 1:
        task = 'finetune'
    else:
        task = 'from_scratch'

    exp_name = f"{exp_name}_ep{args.eps}_ds{args.subset_rate}_{args.msg}"
    
    output_dir = os.path.join(f'{read_dir}/hyperpara/{thresh_percent}/{args.finetune_in_out}/{the_trainset}/{task}/{args.main_loss_type}', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Configure the logging module
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S', 
        level=logging.INFO,
        handlers=[
                logging.FileHandler(os.path.join(output_dir, "log.log")),
                logging.StreamHandler()
                ])
    
    # Set loss ratio type
    if args.ratio_type == 'fix':
        the_main_task_r = args.main_loss_ratio
    elif args.ratio_type == 'scheduler':
        scheduler1= [0, 10, 10, 20, 20, 40] # The epoch where the ratio changes
        scheduler2=[0, 0, 0.9, 0.9, 0.9, 1.0] # The ratio to change
        main_loss_scheduler = lambda t: np.interp([t],\
                scheduler1,\
                scheduler2)[0]
        the_main_task_r = main_loss_scheduler
    else:
        raise ValueError("Choose between fix or scheduler for loss ratio.")
    
    
    mask_origin = torch.load(f'{read_dir}/mask.pt')
    trigger_origin = torch.load(f'{read_dir}/pattern.pt')
    local_weights = torch.load(f'{read_dir}/checkpoint.pkl')
    
    mask = mask_origin.detach()
    trigger = trigger_origin.detach()
    
    # Create model in load checkpoint if needed
    water_model = get_model(args, args.dataset)
    water_model.load_state_dict(local_weights[0])
    train_model = get_model(args, args.dataset)
    
    train_model.to(args.device)
    water_model.to(args.device)
    mask.to(args.device)
    trigger.to(args.device)
    
    if args.using_checkpoint:
        logging.info("Start with checkpoint")
        train_model.load_state_dict(local_weights[0])
    else:
        logging.info("Start with initialed weight")
    
    # Load the optimizer from checkpoint
    if args.opt_type == 'SGD':
        opt = torch.optim.SGD(train_model.parameters(), lr=args.opt_lr, momentum=args.opt_mo)
        lr_s = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.opt_type == 'Adam':
        opt = torch.optim.Adam(train_model.parameters(), lr=args.opt_lr)
        lr_s = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    else:
        raise ValueError("Choose between Adam or SGD for optimizer.")
     
    # Start training
    train_test_acc, train_query_acc, water_test_acc, water_query_acc = attention_shifting_distillation(args.dataset, args.subset_rate, train_model, water_model, opt, args.device, args.target_label, mask, trigger, args.epoch, args.new_loss_ratio, the_main_task_r, args.main_loss_type , args.layer_output, args.layer_input, args.hook_layer, args.alpha_kl, args.temperature, output_dir, args.hresh_mask, args.finetunedata, args.finetune_in_out, lr_s)
    
    """ Write in results """    
    b = int(len(train_test_acc)/9)
    b1 = len(train_test_acc)%9
      
    if args.ratio_type == "fix":
        logging.info(f"===============================Training on {args.subset_rate} of {the_trainset} with {args.main_loss_type}/new loss ratio {args.main_loss_ratio}/{args.new_loss_ratio} and opt {args.opt_type} for {args.epoch} epochs.===============================")    
    elif args.ratio_type == "scheduler":
        logging.info(f"===============================Training on {args.subset_rate} of {the_trainset} with non-fixed ratio on {args.main_loss_type} and opt {args.opt_type} for {args.epoch} epochs.===============================")
    logging.info(f"Train model Test Acc: [Epoch, Acc, Asr]")
    for idx in range(b):
        logging.info(train_test_acc[idx*9:idx*9+9])
    logging.info(train_test_acc[b*9:])
    logging.info(f"Water model Test/Query Acc:{water_test_acc}")
    if args.ratio_type == "fix":
        logging.info(f"===============================Training on {args.subset_rate} of {the_trainset} with {args.main_loss_type}/new loss ratio {args.main_loss_ratio}/{args.new_loss_ratio} and opt {args.opt_type} for {args.epoch} epochs.===============================")    
    elif args.ratio_type == "scheduler":
        logging.info(f"===============================Training on {args.subset_rate} of {the_trainset} with non-fixed ratio on {args.main_loss_type} and opt {args.opt_type} for {args.epoch} epochs.===============================")
    
    logging.info(f"########################### Hyperparameters setting ###########################")
    logging.info(f"hook_layer = {args.hook_layer}, layer_input = {args.layer_input}, layer_out = {args.layer_output}")
    if args.ratio_type == "scheduler":
        logging.info(f"main_loss_scheduler_t = {scheduler1}")
        logging.info(f"main_loss_scheduler_r = {scheduler2}")
    if args.opt_type == 'SGD':
        logging.info(f"SGD setting lr/momen: {args.opt_lr}/{args.opt_mo}")
        logging.info(f"Lr step/gamma: {args.lr_step}/{args.lr_gamma}")
    if args.main_loss_type == 'KD':
        logging.info(f"KD setting Alpha/Temprature: {args.alpha_kl}/{args.temperature}")
    logging.info(f"Epsilon in ASL: {args.eps}")
    logging.info(f"Directory: {output_dir}")
    logging.info(f"###############################################################################")
    
    
    