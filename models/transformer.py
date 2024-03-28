import math
import random
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import trunc_normal_


class InputEmbedding(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.input_embedding = nn.Conv2d(
            input_channel, output_channel, kernel_size=(self.patch_size, 1), stride=(self.patch_size, 1)
        )

    def forward(self, input):
        batch_size, num_nodes, num_channels, long_seq_len = input.size()
        input = input.unsqueeze(-1)
        input = input.reshape(batch_size * num_nodes, num_channels, long_seq_len, 1)
        output = self.input_embedding(input)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        return output


class LearnableTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, input, indices):
        if indices is None:
            pe = self.pe[: input.size(1), :].unsqueeze(0)
        else:
            pe = self.pe[indices].unsqueeze(0)
        x = input + pe
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.tem_pe = LearnableTemporalPositionalEncoding(hidden_dim, dropout)

    def forward(self, input, indices=None):
        batch_size, num_nodes, num_subseq, out_channels = input.size()
        input = self.tem_pe(input.view(batch_size * num_nodes, num_subseq, out_channels), indices=indices)
        input = input.view(batch_size, num_nodes, num_subseq, out_channels)
        return input


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, input):
        batch_size, num_nodes, num_subseq, out_channels = input.size()
        x = input * math.sqrt(self.d_model)
        x = x.view(batch_size * num_nodes, num_subseq, out_channels)
        x = x.transpose(0, 1)
        output = self.transformer_encoder(x, mask=None)
        output = output.transpose(0, 1).view(batch_size, num_nodes, num_subseq, out_channels)
        return output


class MaskGenerator(nn.Module):
    def __init__(self, mask_size, mask_ratio):
        super().__init__()
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.mask_size)))
        random.shuffle(mask)
        mask_len = int(self.mask_size * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class Transformer(nn.Module):
    def __init__(
        self, patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, num_encoder_layers=6, mode="pretrain"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.selected_feature = 0
        self.mode = mode
        self.patch = InputEmbedding(patch_size, in_channel, out_channel)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask = MaskGenerator(mask_size, mask_ratio)
        self.encoder = TransformerLayers(out_channel, num_encoder_layers)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.decoder = TransformerLayers(out_channel, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        trunc_normal_(self.mask_token, std=0.02)
        self.output_layer = nn.Linear(out_channel, patch_size)

    def _forward_pretrain(self, input):
        batch_size, num_nodes, num_features, long_seq_len = input.size()

        patches = self.patch(input)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)

        indices_not_masked, indices_masked = self.mask()
        repr_not_masked = patches[:, :, indices_not_masked, :]

        hidden_not_masked = self.encoder(repr_not_masked)
        hidden_not_masked = self.encoder_2_decoder(hidden_not_masked)
        hidden_masked = self.pe(
            self.mask_token.expand(batch_size, num_nodes, len(indices_masked), hidden_not_masked.size(-1)),
            indices=indices_masked
        )
        hidden = torch.cat([hidden_not_masked, hidden_masked], dim=-2)
        hidden = self.decoder(hidden)

        output = self.output_layer(hidden)
        output_masked = output[:, :, len(indices_not_masked) :, :]
        output_masked = output_masked.view(batch_size, num_nodes, -1).transpose(1, 2)

        labels = (
            input.permute(0, 3, 1, 2)
            .unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :]
            .transpose(1, 2)
        )
        labels_masked = labels[:, :, indices_masked, :].contiguous()
        labels_masked = labels_masked.view(batch_size, num_nodes, -1).transpose(1, 2)
        return output_masked, labels_masked

    def _forward_backend(self, input):
        patches = self.patch(input)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)
        hidden = self.encoder(patches)
        return hidden

    def forward(self, input_data):
        if self.mode == "pretrain":
            return self._forward_pretrain(input_data)
        else:
            return self._forward_backend(input_data)


if __name__ == "__main__":
    from torchinfo import summary

    model = Transformer(patch_size=12, in_channel=1, out_channel=64, dropout=0.1, mask_size=228 * 7 * 2 / 12, mask_ratio=0.75, num_encoder_layers=4)
    summary(model)
