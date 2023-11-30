from copy import deepcopy
import torch.nn as nn
import torch
import types
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import pickle
import argparse
import warnings
from scene_net import *
from dataloaders import * 
from torch.utils.data import DataLoader
from torchvision import models
from loss import SceneNetLoss, DiSparse_SceneNetLoss
import pprint


################################################################################################
# Overwrite PyTorch forward function for Conv2D and Linear to take the mask into account
def hook_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def hook_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def map_sparsity_to_keep_ratio(sparsity, dataset):

    if dataset == "nyuv2":
        if sparsity == 90:
            keep_ratio = 0.08
        elif sparsity == 70:
            keep_ratio = 0.257
        elif sparsity == 50:
            keep_ratio = 0.46
        elif sparsity == 30:
            keep_ratio = 0.675
        else:
            keep_ratio = (100 - sparsity) / 100
    elif dataset == "cityscapes":
        if sparsity == 90:
            keep_ratio = 0.095
        elif sparsity == 70:
            keep_ratio = 0.3
        elif sparsity == 50:
            keep_ratio = 0.51
        elif sparsity == 30:
            keep_ratio = 0.71
        else:
            keep_ratio = (100 - sparsity) / 100
    elif dataset == "taskonomy":
        if sparsity == 90:
            keep_ratio = 0.097
        elif sparsity == 70:
            keep_ratio = 0.257
        elif sparsity == 50:
            keep_ratio = 0.46
        elif sparsity == 30:
            keep_ratio = 0.675
        else:
            keep_ratio = (100 - sparsity) / 100
    else:
        print("Unrecognized Dataset Name.")
        exit()

    return keep_ratio

################################################################################################
# Returns masks and saliency scores of parameters for each task.
# net: model to sparsify, should be of SceneNet class
# criterion: loss function to calculate per task gradients, should be of DiSparse_SceneNetLoss class
# train_loader: dataloader to fetch data batches used to estimate importance
# keep_ratio: how many parameters to keep
# tasks: set of tasks
def compute_task_subnetworks(net, criterion, train_loader, num_batches, keep_ratio, device, tasks):
    test_net = deepcopy(net)
    grads_abs = {}
    for task in tasks:
        grads_abs[task] = []
    # Register Hook
    for layer in test_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)

    # Estimate importance per task in a data-driven manner
    train_iter = iter(train_loader)
    for i in range(num_batches):

        print("batch: ", i)

        gt_batch = None
        preds = None
        loss = None
        torch.cuda.empty_cache()

        gt_batch = next(train_iter)
        gt_batch["img"] = Variable(gt_batch["img"]).cuda()
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).cuda()
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).cuda()
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).cuda()
        if "keypoint" in gt_batch:
            gt_batch["keypoint"] = Variable(gt_batch["keypoint"]).cuda()
        if "edge" in gt_batch:
            gt_batch["edge"] = Variable(gt_batch["edge"]).cuda()
        
        for i, task in enumerate(tasks):
            preds = None
            torch.cuda.empty_cache()
            test_net.zero_grad()
            preds = test_net.forward(gt_batch['img'])
            loss = criterion(preds, gt_batch, cur_task=task)
            loss.backward()
            ct = 0
            
            for name, layer in test_net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if 'backbone' in name or f'task{i+1}' in name:
                        if len(grads_abs[task]) > ct:
                            grads_abs[task][ct] += torch.abs(layer.weight_mask.grad.data)
                        else:
                            grads_abs[task].append(torch.abs(layer.weight_mask.grad.data))
                        ct += 1

    preds = None
    loss = None
    # Calculate Threshold
    masks = {}
    saliencies = {}
    for task in tasks:
        masks[task] = []
        saliencies[task] = []


    # Get importance scores for each task independently
    for i, task in enumerate(tasks):
        cur_grads_abs = grads_abs[task]
        all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        for g in cur_grads_abs:
            masks[task].append(((g / norm_factor) >= acceptable_score).int())
            saliencies[task].append((g/norm_factor))

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in masks[task]])))

    with open("sal_masks_static_new.txt", 'wb') as file:
        pickle.dump({"masks": masks, "sals": saliencies}, file)

    return masks, saliencies

# Layer-wise similarity score.
def layer_sim(mask1, mask2, sal1 = None, sal2 = None, sim_metric = "epi"):

    n1 = torch.count_nonzero(mask1).item()
    n2 = torch.count_nonzero(mask2).item()
    
    if sim_metric == "epi":
        similarity = 1 - (abs(n1 - n2) / (n1 + n2))
    elif sim_metric == "iou":
        intersection = torch.logical_and(mask1, mask2)
        union = torch.logical_or(mask1, mask2)
        similarity = (torch.count_nonzero(intersection) / torch.count_nonzero(union)).item()
    elif sim_metric == "sals":
        similarity = torch.nn.functional.cosine_similarity(sal1.flatten(), sal2.flatten(), dim=0, eps=1e-8)

    return similarity

def weighted_average(numbers):
    n = len(numbers)
    weights = []
    for i in range(n):
        #weights.append(n-i)
        weights.append(1)
    sum = 0
    weights_sum = 0
    for weight, val in zip(weights, numbers):
        sum += weight * val
        weights_sum += weight
    return sum/weights_sum

def subnet_similarity(mask1, mask2, sal1, sal2, model):
    sim_scores = []
    count = 0

    for (name, module) in model.named_modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and ("backbone" in name):
            params = module.parameters()
            sim_scores.append(layer_sim(mask1[count].bool(), mask2[count].bool(), sal1[count], sal2[count], sim_metric="epi"))
            count += 1
    
    return weighted_average(sim_scores)

def get_pairwise_similarities(task_masks, task_saliencies, net):
    similarity_dict = {}
    for task1 in task_masks:
        for task2 in task_masks:
            if (task2 + "_" + task1) in similarity_dict or (task1 + "_" + task2) in similarity_dict:
                continue
            similarity_dict[task1+"_"+task2] = subnet_similarity(task_masks[task1], task_masks[task2], task_saliencies[task1], task_saliencies[task2], net)
    
    return similarity_dict

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset: choose between nyuv2, cityscapes, taskonomy', default="nyuv2")
    parser.add_argument('--num_batches',type=int, help='number of batches to estimate importance', default=2)
    parser.add_argument('--sim_method', type=str, help='method name', default="iou")
    parser.add_argument('--sparsity',type=int, help='sparsity level', default=90) # TODO: we should take in a list of sparsities, do average
    args = parser.parse_args()

    dataset = args.dataset
    num_batches = args.num_batches
    sim_method = args.sim_method
    sparsity = args.sparsity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ################################################################################################
    if dataset == "nyuv2":
        from config_nyuv2 import *
        train_dataset = NYU_v2(DATA_ROOT, 'train', crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = NYU_v2(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 8, shuffle=True, pin_memory=True)
    elif dataset == "cityscapes":
        from config_cityscapes import *
        train_dataset = CityScapes(DATA_ROOT, 'train', crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = CityScapes(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 8, shuffle=True, pin_memory=True)
    elif dataset == "taskonomy":
        from config_taskonomy import *
        train_dataset = Taskonomy(DATA_ROOT, 'train', crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE//4, num_workers = 8, shuffle=True, pin_memory=True)
        test_dataset = Taskonomy(DATA_ROOT, 'test')
        test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 8, shuffle=True, pin_memory=True)
    else:
        print("Unrecognized Dataset Name.")
        exit()

    net = SceneNet(TASKS_NUM_CLASS).to(device)

    # TODO: how is this different from SceneNetLoss? if it is significantly different, we should rename it.
    # otherwise, just modify SceneNetLoss to this since we aren't training anyway.
    criterion = DiSparse_SceneNetLoss(dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, device, DATA_ROOT)

    keep_ratio = map_sparsity_to_keep_ratio(sparsity, dataset)
    print(keep_ratio)
    
    task_masks, task_saliencies = compute_task_subnetworks(net, criterion, train_loader, num_batches, keep_ratio, device, tasks=TASKS)

    pairwise_task_similarities = get_pairwise_similarities(task_masks, task_saliencies, net)
    pprint.PrettyPrinter(width=20).pprint(pairwise_task_similarities)
