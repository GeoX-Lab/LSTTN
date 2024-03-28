import torch
from torch.utils.data import Dataset
from utils import load_pkl


class MTSDataset(Dataset):
    def __init__(self, raw_file_path, index_file_path, seq_len, throw=False, pretrain=False) -> None:
        super().__init__()
        self.pretrain = pretrain
        self.seq_len = seq_len

        self.data = torch.from_numpy(load_pkl(raw_file_path)).float()  # L, N, C
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])

        # [idx - 12, idx, idx + 12]
        index = load_pkl(index_file_path)
        if throw:
            self.index = self.preprocess(index)
        else:
            self.index = index

    def preprocess(self, index):
        for i, idx in enumerate(index):
            current_moment = idx[1]
            if current_moment - self.seq_len < 0:
                continue
            else:
                break
        return index[i:]

    def reshape_data(self, data):
        if not self.pretrain:
            pass
        else:
            data = data[..., [0]]
            data = data.permute(1, 2, 0)
        return data

    def __getitem__(self, index):
        idx = self.index[index]
        y = self.data[idx[1] : idx[2], ...]
        short_x = self.data[idx[0] : idx[1], ...]
        if self.pretrain:
            long_x = self.data[idx[1] - self.seq_len : idx[1], ...]
            long_x = self.reshape_data(long_x)
            y = None
            short_x = None
            abs_idx = torch.Tensor(range(idx[1] - self.seq_len, idx[1], 12))
            return long_x, abs_idx
        else:
            if idx[1] - self.seq_len < 0:
                long_x = self.mask
            else:
                long_x = self.data[idx[1] - self.seq_len : idx[1], ...]
            return y, short_x, long_x

    def __len__(self):
        return len(self.index)
