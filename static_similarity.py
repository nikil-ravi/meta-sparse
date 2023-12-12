import torch.nn as nn
import torch
from torch.autograd import Variable
import types
import pickle
import argparse
import warnings
from scene_net import *
from dataloaders import *
from torch.utils.data import DataLoader
from loss import DiSparse_SceneNetLoss
import pprint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DELIM = "_X_"
EPSILON = 1e-9

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="dataset: choose between nyuv2, cityscapes, taskonomy",
    default="nyuv2",
)
parser.add_argument(
    "--num_batches",
    type=int,
    help="number of batches to estimate importance",
    default=5,
)
parser.add_argument(
    "--num_seeds",
    type=int,
    help="number of seeds to average over for each sparsity ratio",
    default=5,
)
parser.add_argument("--sim_method", type=str, help="method name", default="iou")
parser.add_argument(
    "--layer_method",
    type=str,
    help="method of assigning layer weights for similarity scores",
    default="equal",
)
parser.add_argument(
    "--sparsities", type=str, help="sparsity levels", default="30,50,70,90"
)
parser.add_argument(
    "--dump_dir",
    type=str,
    help="directory to store subnetworks",
    default="./subnetworks/",
)
parser.add_argument("--force_run", action="store_true")
parser.add_argument("--table_json", default="table.json")


def cached(f, force_run, cache_filename):
    path = cache_filename + ".pkl"
    if force_run or not os.path.exists(path):
        ret = f()
        with open(path, "wb") as f:
            pickle.dump(ret, f)
        return ret
    else:
        # print("Using cached", cache_filename)
        with open(path, "rb") as f:
            return pickle.load(f)


def get_args():
    args = parser.parse_args()
    args.sparsities = [int(sparsity) for sparsity in args.sparsities.split(",")]
    return args


def get_dataset(dataset):
    if dataset == "nyuv2":
        train_dataset = NYU_v2(DATA_ROOT, "train", crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        test_dataset = NYU_v2(DATA_ROOT, "test")
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True
        )
    elif dataset == "cityscapes":
        train_dataset = CityScapes(DATA_ROOT, "train", crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        test_dataset = CityScapes(DATA_ROOT, "test")
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True
        )
    elif dataset == "taskonomy":
        train_dataset = Taskonomy(DATA_ROOT, "train", crop_h=CROP_H, crop_w=CROP_W)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE // 4,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
        test_dataset = Taskonomy(DATA_ROOT, "test")
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True
        )
    else:
        raise Exception("Unrecognized Dataset Name.")

    return train_dataset, train_loader, test_dataset, test_loader


def hook_forward_conv2d(self, x):
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def hook_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def get_keep_ratios(sparsities, dataset):
    keep_ratios = []

    for sparsity in sparsities:
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
            raise Exception("Unrecognized Dataset Name.")

        keep_ratios.append(keep_ratio)
    return keep_ratios


def get_batch(train_iter):
    batch = next(train_iter)
    batch["img"] = Variable(batch["img"]).to(DEVICE)
    if "seg" in batch:
        batch["seg"] = Variable(batch["seg"]).to(DEVICE)
    if "depth" in batch:
        batch["depth"] = Variable(batch["depth"]).to(DEVICE)
    if "normal" in batch:
        batch["normal"] = Variable(batch["normal"]).to(DEVICE)
    if "keypoint" in batch:
        batch["keypoint"] = Variable(batch["keypoint"]).to(DEVICE)
    if "edge" in batch:
        batch["edge"] = Variable(batch["edge"]).to(DEVICE)

    return batch


def compute_grad_abs(net, criterion, train_loader, num_batches, tasks, seed):
    grad_abs = {}
    for task in tasks:
        grad_abs[task] = []

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(hook_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(hook_forward_linear, layer)

    train_iter = iter(train_loader)
    for i in range(num_batches):
        print(f"Seed {seed}: Batch {i}/{num_batches}")

        batch = get_batch(train_iter)
        for i, task in enumerate(tasks):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            net.zero_grad()
            preds = net.forward(batch["img"])
            loss = criterion(preds, batch, cur_task=task)
            loss.backward()

            ct = 0
            for name, layer in net.named_modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if "backbone" in name or f"task{i+1}" in name:
                        if len(grad_abs[task]) > ct:
                            grad_abs[task][ct] += torch.abs(
                                layer.weight_mask.grad.data
                            ).to("cpu")
                        else:
                            grad_abs[task].append(
                                torch.abs(layer.weight_mask.grad.data).to("cpu")
                            )
                        ct += 1

    return grad_abs


def compute_task_subnetwork(grad_abs, task, keep_ratio):
    masks = []
    saliencies = []

    cur_grads_abs = grad_abs[task]
    all_scores = torch.cat([torch.flatten(x) for x in cur_grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    for g in cur_grads_abs:
        masks.append(((g / norm_factor) >= acceptable_score).int())
        saliencies.append((g / norm_factor))

    return masks, saliencies


def weighted_average(vals, weights=None):
    if weights is None:
        weights = [1.0] * len(vals)
    sum = 0.0
    weight_sum = 0.0
    for i in range(len(vals)):
        sum += weights[i] * vals[i]
        weight_sum += weights[i]

    return sum / weight_sum


def layer_sim(mask1, mask2, sal1, sal2, sim_metric):
    if sim_metric == "epi":
        n1 = torch.count_nonzero(mask1).item()
        n2 = torch.count_nonzero(mask2).item()
        similarity = 1 - (abs(n1 - n2) / max(n1 + n2, EPSILON))
    elif sim_metric == "iou":
        intersection = torch.logical_and(mask1, mask2)
        union = torch.logical_or(mask1, mask2)
        similarity = (
            torch.count_nonzero(intersection) / torch.count_nonzero(union)
        ).item()
    elif sim_metric == "sals":
        similarity = torch.nn.functional.cosine_similarity(
            sal1.flatten(), sal2.flatten(), dim=0, eps=1e-8
        ).item()
    elif sim_metric == "masked_sals":
        similarity = torch.nn.functional.cosine_similarity(
            (mask1 * sal1).flatten(), (mask2 * sal2).flatten(), dim=0, eps=1e-8
        ).item()

    else:
        raise Exception("Unknown similarity metric")

    return similarity


def get_layer_weight(count, mask1, mask2, sal1, sal2, layer_method):
    if layer_method == "equal":
        return 1.0
    elif layer_method == "linear":
        return count
    elif layer_method == "linear_plus_one":
        return count + 1
    elif layer_method == "five_zeros":
        return 0.0 if count < 5 else 1.0
    sparsity1 = 1.0 - (
        torch.count_nonzero(mask1[count].bool()).item() / mask1[count].numel()
    )
    sparsity2 = 1.0 - (
        torch.count_nonzero(mask2[count].bool()).item() / mask2[count].numel()
    )
    if layer_method == "av_sparsity":
        return sparsity1 + sparsity2
    elif layer_method == "min_sparsity":
        return min(sparsity1, sparsity2)
    elif layer_method == "epi_like":
        n1 = torch.count_nonzero(mask1[count].bool()).item()
        n2 = torch.count_nonzero(mask2[count].bool()).item()
        return 1 - (abs(n1 - n2) / max(n1 + n2, EPSILON))


def subnet_similarity(mask1, mask2, sal1, sal2, model, sim_method, layer_method):
    sim_scores = []
    count = 0
    weights = []

    for (name, module) in model.named_modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and (
            "backbone" in name
        ):
            curr_mask1 = mask1[count].bool()
            curr_mask2 = mask2[count].bool()
            curr_sal1 = sal1[count]
            curr_sal2 = sal2[count]
            sim_scores.append(
                layer_sim(
                    curr_mask1,
                    curr_mask2,
                    curr_sal1,
                    curr_sal2,
                    sim_metric=sim_method,
                )
            )

            weights.append(
                get_layer_weight(
                    count, curr_mask1, curr_mask2, curr_sal1, curr_sal2, layer_method
                )
            )
            count += 1

    return weighted_average(sim_scores, weights)


def get_task_id(task1, task2):
    return task1 + DELIM + task2


def tasks_in_keys(task1, task2, keys):
    return get_task_id(task1, task2) in keys or get_task_id(task2, task1) in keys


def get_pairwise_similarity(masks, saliencies, model, sim_method, layer_method):
    pairwise_similarity = {}
    tasks = list(masks.keys())
    ratios = list(masks[tasks[0]].keys())
    for task1 in tasks:
        for task2 in tasks:
            if tasks_in_keys(task1, task2, pairwise_similarity.keys()):
                continue
            pairwise_similarity[get_task_id(task1, task2)] = weighted_average(
                [
                    subnet_similarity(
                        masks[task1][ratio],
                        masks[task2][ratio],
                        saliencies[task1][ratio],
                        saliencies[task2][ratio],
                        model,
                        sim_method,
                        layer_method,
                    )
                    for ratio in ratios
                ]
            )
    return pairwise_similarity


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_args()

    # Can't remove this
    if args.dataset == "nyuv2":
        from config_nyuv2 import *
    elif args.dataset == "cityscapes":
        from config_cityscapes import *
    elif args.dataset == "taskonomy":
        from config_taskonomy import *

    _, train_loader, _, _ = get_dataset(args.dataset)
    criterion = DiSparse_SceneNetLoss(
        args.dataset, TASKS, TASKS_NUM_CLASS, LAMBDAS, DEVICE, DATA_ROOT
    )
    keep_ratios = get_keep_ratios(args.sparsities, args.dataset)

    os.makedirs(args.dump_dir, exist_ok=True)
    base_dir = os.path.join(args.dump_dir, args.dataset)
    os.makedirs(base_dir, exist_ok=True)

    pairwise_similarities = []
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        net = SceneNet(TASKS_NUM_CLASS).to(DEVICE)

        grad_abs = cached(
            lambda: compute_grad_abs(
                net, criterion, train_loader, args.num_batches, TASKS, seed
            ),
            args.force_run,
            os.path.join(base_dir, f"grad_abs_s{seed}_v0"),
        )

        masks = {}
        saliencies = {}
        for task in grad_abs.keys():
            masks[task] = {}
            saliencies[task] = {}
            for keep_ratio in keep_ratios:
                masks[task][keep_ratio], saliencies[task][keep_ratio] = cached(
                    lambda: compute_task_subnetwork(grad_abs, task, keep_ratio),
                    args.force_run,
                    os.path.join(base_dir, f"subnet_s{seed}_t{task}_r{keep_ratio}_v0"),
                )

        pairwise_similarities.append(
            get_pairwise_similarity(
                masks, saliencies, net, args.sim_method, args.layer_method
            )
        )

    average_pairwise_similarities = {
        key: weighted_average(
            [pairwise_similarity[key] for pairwise_similarity in pairwise_similarities]
        )
        for key in pairwise_similarities[0].keys()
    }
    pprint.PrettyPrinter(width=20).pprint({"config":vars(args), "sim":average_pairwise_similarities})
    
    with open(args.table_json, 'r') as f:
        table = json.load(f)
    
    for k in average_pairwise_similarities.keys():
        tasks = k.split(DELIM)
        if tasks[0]==tasks[1]:
            continue
        key = args.layer_method + DELIM + args.sim_method
        if not key in table:
            table[key] = {}
        score_key = "s" + ",".join([str(x) for x in args.sparsities]) + DELIM + k
        table[key][score_key] = average_pairwise_similarities[k]
    
    with open(args.table_json, 'w') as f:
        json.dump(table, f)

