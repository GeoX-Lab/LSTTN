import torch.nn as nn
from utils.metrics import masked_mae, masked_rmse, masked_mape
from utils.datasets import MTSDataset
from easytorch import Runner
from easytorch.device import to_device
from models.transformer import Transformer
from utils.load_data import max_min_scale, standard_scale
from utils.log import load_pkl


class TransformerRunner(Runner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.clip = 5
        dataset_name = cfg["DATASET_NAME"]
        # scaler
        if dataset_name in ["PEMS04", "PEMS08"]:
            _min = load_pkl("datasets/" + dataset_name + "/min.pkl")
            _max = load_pkl("datasets/" + dataset_name + "/max.pkl")
            self.scaler = max_min_scale
            self.scaler_args = {"min": _min, "max": _max}
        elif dataset_name in ["PEMSBAY", "METRLA"]:
            mean = load_pkl("datasets/" + dataset_name + "/mean.pkl")
            std = load_pkl("datasets/" + dataset_name + "/std.pkl")
            self.scaler = standard_scale
            self.scaler_args = {"mean": mean, "std": std}
        self.loss = masked_mae

    def init_training(self, cfg):
        super().init_training(cfg)
        self.register_epoch_meter("train_loss", "train", "{:.4f}")
        self.register_epoch_meter("train_MAPE", "train", "{:.4f}")
        self.register_epoch_meter("train_RMSE", "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        self.register_epoch_meter("val_loss", "val", "{:.4f}")
        self.register_epoch_meter("val_MAPE", "val", "{:.4f}")
        self.register_epoch_meter("val_RMSE", "val", "{:.4f}")

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        return {"Transformer": Transformer}[cfg["MODEL"]["NAME"]](**cfg["MODEL"].get("PARAM", {}))

    @staticmethod
    def build_train_dataset(cfg: dict):
        raw_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/train_index.pkl"
        seq_len = cfg["TRAIN"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=True, pretrain=True)
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        raw_file_path = cfg["VAL"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["VAL"]["DATA"]["DIR"] + "/valid_index.pkl"
        seq_len = cfg["VAL"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=True, pretrain=True)
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        raw_file_path = cfg["TEST"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TEST"]["DATA"]["DIR"] + "/test_index.pkl"
        seq_len = cfg["TEST"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=True, pretrain=True)
        return dataset

    def train_iters(self, epoch, iter_index, data):
        input_data = to_device(data[0])

        output_masked_tokens, label_masked_tokens = self.model(input_data)

        if "max" in self.scaler_args.keys():
            predict = (
                self.scaler(output_masked_tokens.transpose(1, 2).unsqueeze(-1), **self.scaler_args)
                .transpose(1, 2)
                .squeeze(-1)
            )
            real_val = (
                self.scaler(label_masked_tokens.transpose(1, 2).unsqueeze(-1), **self.scaler_args)
                .transpose(1, 2)
                .squeeze(-1)
            )
            mae_loss = self.loss(predict, real_val)
        else:
            predict = self.scaler(output_masked_tokens, **self.scaler_args)
            real_val = self.scaler(label_masked_tokens, **self.scaler_args)
            mae_loss = self.loss(predict, real_val, 0)

        loss = mae_loss
        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        self.update_epoch_meter("train_loss", loss.item())
        self.update_epoch_meter("train_MAPE", mape.item())
        self.update_epoch_meter("train_RMSE", rmse.item())
        return loss

    def val_iters(self, iter_index, data):
        input_data = to_device(data[0])

        output_masked_tokens, label_masked_tokens = self.model(input_data)

        if "max" in self.scaler_args.keys():
            predict = (
                self.scaler(output_masked_tokens.transpose(1, 2).unsqueeze(-1), **self.scaler_args)
                .transpose(1, 2)
                .squeeze(-1)
            )
            real_val = (
                self.scaler(label_masked_tokens.transpose(1, 2).unsqueeze(-1), **self.scaler_args)
                .transpose(1, 2)
                .squeeze(-1)
            )
            loss = self.loss(predict, real_val)
        else:
            predict = self.scaler(output_masked_tokens, **self.scaler_args)
            real_val = self.scaler(label_masked_tokens, **self.scaler_args)
            loss = self.loss(predict, real_val, 0.0)

        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        self.update_epoch_meter("val_loss", loss.item())
        self.update_epoch_meter("val_MAPE", mape.item())
        self.update_epoch_meter("val_RMSE", rmse.item())
