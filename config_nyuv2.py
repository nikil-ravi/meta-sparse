'''
Configuration File Used for Cityscapes Training & Evaluation
'''

DATA_ROOT = "/Users/onkardeshpande/code/Stanford/CS330/project/meta-sparse/data/nyu_v2"
CROP_H = 321
CROP_W = 321
TASKS = ["seg", "sn", "depth"]
TASKS_NUM_CLASS = [40, 3, 1]

LAMBDAS = [1, 20, 3]
NUM_GPUS = 1
BATCH_SIZE = 8
# MAX_ITERS = 5000
MAX_ITERS = 5
DECAY_LR_FREQ = 4000
DECAY_LR_RATE = 0.5
    
INIT_LR = 1e-3
WEIGHT_DECAY = 5e-4
IMAGE_SHAPE = (480, 640)

PRUNE_TIMES = 11
PRUNE_ITERS = [100] * PRUNE_TIMES

# --------------------------------------------------------------- #
END = 15000
INT = 50
PRUNE_RATE = 0.5
RETRAIN_EPOCH = 1000
RETRAIN_LR = 1e-5
