import json
import subprocess

RUN_SIM = False
DELIM = "_X_"
layer_methods = ["equal", "linear", "linear_plus_one", "five_zeros", "av_sparsity", "min_sparsity", "epi_like"]
sparsitiess = ["30", "50", "70", "90", "30,50,70,90"]
sim_methods = ["epi", "iou", "sals", "masked_sals"]


if RUN_SIM:
    for layer_method in layer_methods:
        for sparsities in sparsitiess:
            for sim_method in sim_methods:
                subprocess.run(["python", "static_similarity.py", "--num_batches", "50", "--num_seeds", "4", "--layer_method", layer_method, "--sparsities", sparsities, "--sim_method", sim_method])

with open("table.json","r") as f:
    table = json.load(f)

tasks = ["seg"+DELIM+"sn", "seg"+DELIM+"depth", "sn"+DELIM+"depth"]
for layer_method in layer_methods:
    for sim_method in sim_methods:
        to_print = []
        for sparsities in sparsitiess:
            for task in tasks:
                to_print.append("{:.3f} & ".format(table[layer_method+DELIM+sim_method]["s"+sparsities+DELIM+task]))
        print(" & ".join(to_print)+"\\\\")

