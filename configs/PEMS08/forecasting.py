import os
import torch
from easydict import EasyDict
from runners.lsttn import LSTTNRunner
from utils.load_data import load_adj


DATASET_NAME = "PEMS08"

GRAPH_PKL_PATH = "datasets/sensor_graph/adj_mx_08.pkl"
NUM_NODES = 170
adj_mx, _ = load_adj(GRAPH_PKL_PATH, "doubletransition")

CFG = EasyDict()
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH = True
GPU_NUM = 2
SEED = 0

SHORT_SEQ_LEN = 12
LONG_SEQ_LEN = 288 * 7 * 2
HIDDEN_DIM = 96
MASK_RATIO = 0.75
NUM_TRANSFORMER_ENCODER_LAYERS = 4

CFG.DESCRIPTION = "LSTTN"
CFG.RUNNER = LSTTNRunner
CFG.DATASET_NAME = DATASET_NAME
CFG.USE_GPU = True if GPU_NUM > 0 else False
CFG.GPU_NUM = GPU_NUM
CFG.SEED = SEED
CFG.K = 10
CFG.CUDNN_ENABLED = True

CFG.MODEL = EasyDict()
CFG.MODEL.FIND_UNUSED_PARAMETERS = True
CFG.MODEL.NAME = "LSTTN"
CFG.MODEL.PARAM = EasyDict()
CFG.MODEL.PARAM.TRANSFORMER = {
    "patch_size": SHORT_SEQ_LEN,
    "in_channel": 1,
    "out_channel": HIDDEN_DIM,
    "dropout": 0.1,
    "mask_size": LONG_SEQ_LEN / SHORT_SEQ_LEN,
    "mask_ratio": MASK_RATIO,
    "num_encoder_layers": NUM_TRANSFORMER_ENCODER_LAYERS,
}
CFG.MODEL.PARAM.LSTTN = {
    "supports": [torch.tensor(i) for i in adj_mx],
    "num_nodes": NUM_NODES,
    "short_seq_len": SHORT_SEQ_LEN,
    "pre_len": SHORT_SEQ_LEN,
    "long_trend_hidden_dim": 32,
    "seasonality_hidden_dim": 32,
    "mlp_hidden_dim": 128,
    "dropout": 0.3,
}
CFG.MODEL.PARAM.STGNN = EasyDict()
CFG.MODEL.PARAM.STGNN.GWNET = {
    "num_nodes": NUM_NODES,
    "supports": [torch.tensor(i) for i in adj_mx],
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": True,
    "aptinit": None,
    "in_dim": 2,
    "out_dim": 128,
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 256,
    "end_channels": 512,
    "kernel_size": 2,
    "blocks": 4,
    "layers": 2,
}

CFG.TRAIN = EasyDict()
CFG.TRAIN.SETUP_GRAPH = True
CFG.TRAIN.WARMUP_EPOCHS = 100
CFG.TRAIN.CL_EPOCHS = 6
CFG.TRAIN.CKPT_SAVE_STRATEGY = "SaveEveryEpoch"
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", CFG.MODEL.NAME)

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.SEQ_LEN = LONG_SEQ_LEN
CFG.TRAIN.DATA.DATASET_NAME = DATASET_NAME
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.TRAIN.DATA.DATASET_NAME
CFG.TRAIN.DATA.PREFETCH = PREFETCH
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = NUM_WORKERS
CFG.TRAIN.DATA.PIN_MEMORY = PIN_MEMORY

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.005,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 18, 36, 54, 72], "gamma": 0.5}

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.SEQ_LEN = LONG_SEQ_LEN
CFG.VAL.DATA.DATASET_NAME = CFG.TRAIN.DATA.DATASET_NAME
CFG.VAL.DATA.DIR = CFG.TRAIN.DATA.DIR
CFG.VAL.DATA.PREFETCH = CFG.TRAIN.DATA.PREFETCH
CFG.VAL.DATA.BATCH_SIZE = CFG.TRAIN.DATA.BATCH_SIZE
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = CFG.TRAIN.DATA.NUM_WORKERS
CFG.VAL.DATA.PIN_MEMORY = PIN_MEMORY

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.SEQ_LEN = LONG_SEQ_LEN
CFG.TEST.DATA.DATASET_NAME = CFG.TRAIN.DATA.DATASET_NAME
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
CFG.TEST.DATA.PREFETCH = CFG.TRAIN.DATA.PREFETCH
CFG.TEST.DATA.BATCH_SIZE = CFG.TRAIN.DATA.BATCH_SIZE
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = CFG.TRAIN.DATA.NUM_WORKERS
CFG.TEST.DATA.PIN_MEMORY = PIN_MEMORY
