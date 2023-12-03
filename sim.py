from copy import deepcopy
import torch.nn as nn
import torch
import types
from torch.autograd import Variable
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import pickle
import pprint

import torch
import torch.nn as nn
from scene_net import *

TASKS_NUM_CLASS = [40, 3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    # Function to count the number of parameters in each layer of a PyTorch model
    
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Count: {param.numel()}")

    for name, param in model.named_buffers():
        print(f"Layer: {name} | Size: {param.size()} | Count: {param.numel()}")

# Create an instance of the network
net = SceneNet(TASKS_NUM_CLASS).to(device)

def get_pruned_init(net):
    for module in net.modules():
        # Check if it's basic block
        if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
            module = prune.identity(module, 'weight')
    return net

def layer_sim(mask1, mask2, sal1 = None, sal2 = None, sim_metric = "epi"):

    n1 = torch.count_nonzero(mask1).item()
    n2 = torch.count_nonzero(mask2).item()
    
    if sim_metric == "epi":
        similarity = 1 - (abs(n1 - n2) / (n1 + n2))
    elif sim_metric == "iou":
        intersection = torch.logical_and(mask1, mask2)
        union = torch.logical_or(mask1, mask2)
        similarity = torch.count_nonzero(intersection) / torch.count_nonzero(union)
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
    epi_scores = []
    count = 0

    for (name, module) in model.named_modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and ("backbone" in name):
            params = module.parameters()
            epi_scores.append(layer_sim(mask1[count].bool(), mask2[count].bool(), sal1[count], sal2[count], sim_metric="sals"))
            count += 1
    
    return weighted_average(epi_scores)

def all_pairwise_similarities(filename):
    with open(filename, "rb") as file:
        salmasks = pickle.load(file)
    keepmasks = salmasks["masks"]
    sals = salmasks["sals"]
    
    similarity_dict = {}
    for task1 in keepmasks:
        for task2 in keepmasks:
            if task2 + task1 in similarity_dict:
                continue
            similarity_dict[task1+task2] = subnet_similarity(keepmasks[task1], keepmasks[task2], sals[task1], sals[task2], net)
    
    return similarity_dict

pprint.PrettyPrinter(width=20).pprint(all_pairwise_similarities("sal_masks_static.txt"))

