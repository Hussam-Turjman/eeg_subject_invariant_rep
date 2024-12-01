# https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction/tree/main
import torch
import torch.nn as nn
from .info_nce import InfoNCE


class Flatten(nn.Module):
    """
    see https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResBlock(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2007.04871.pdf
    Figure A.1
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.batch_norm_1 = nn.BatchNorm1d(in_channels, track_running_stats=False)
        self.elu_1 = nn.ELU()
        self.conv1d_residual = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1d_1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.batch_norm_2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.elu_2 = nn.ELU()
        self.conv1d_2 = nn.Conv1d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.batch_norm_1(x)
        x = self.elu_1(x)
        x_residual = self.conv1d_residual(x)
        x = self.conv1d_1(x)
        x = self.batch_norm_2(x)
        x = self.elu_2(x)
        x = self.conv1d_2(x)

        if self.in_channels != self.out_channels:
            # print(f"X shape --> {x.shape} , X residual shape --> {x_residual.shape}")
            # Fixme - shape mismatch
            x = x + x_residual[:, :, :x.size(2)]
            pass
        return x


class EncoderG(nn.Module):
    def __init__(self, eeg_channels_count, freq,
                 window_len, current_device):
        super(EncoderG, self).__init__()
        self.current_device = current_device
        self.freq = freq
        self.window_len = window_len
        self.eeg_channels_count = eeg_channels_count
        self.channels_in = eeg_channels_count
        self.channels_out = eeg_channels_count // 2

        #                          64,32,13
        self.conv1d_1 = nn.Conv1d(self.channels_in, self.channels_out, 13)
        #                          32,32,11
        self.res_block_1 = ResBlock(self.channels_out, self.channels_out, 11)
        self.max_pool_1 = nn.MaxPool1d(4)
        #                          32,64,9
        self.res_block_2 = ResBlock(self.channels_out, self.channels_in, 9)
        self.max_pool_2 = nn.MaxPool1d(4)
        #                          64,128,7
        self.res_block_3 = ResBlock(self.channels_in, self.channels_in * 2, 3)
        self.elu = nn.ELU()
        self.flatten = Flatten()

    @property
    def embedding_size(self):
        sample = torch.randn((1, self.eeg_channels_count, int(self.freq * self.window_len)))
        if self.current_device is not None:
            sample = sample.to(self.current_device)
            self.to(self.current_device)
        return self.forward(sample).shape[1]

    def forward(self, x):
        x = self.conv1d_1(x)
        x = self.res_block_1(x)
        x = self.max_pool_1(x)
        x = self.res_block_2(x)
        x = self.max_pool_2(x)
        x = self.res_block_3(x)
        x = self.elu(x)

        x = self.flatten(x)
        return x


class ModelF(nn.Module):
    def __init__(self, embedding_size, current_device):
        super(ModelF, self).__init__()
        self.current_device = current_device
        self.embedding_size = embedding_size
        self.channels_in = embedding_size
        self.channels_out = embedding_size // 2

        self.linear_1 = nn.Linear(self.channels_in, self.channels_out)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(self.channels_out, self.channels_out)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(self.channels_out, self.channels_out)
        self.relu_3 = nn.ReLU()
        self.linear_4 = nn.Linear(self.channels_out, self.channels_out // 2)

    @property
    def output_len(self):
        sample = torch.randn((1, self.embedding_size))
        if self.current_device is not None:
            sample = sample.to(self.current_device)
            self.to(self.current_device)
        return self.forward(sample).shape[1]

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.linear_3(x)
        x = self.relu_3(x)
        x = self.linear_4(x)
        return x


class SSLModel(nn.Module):
    def __init__(self, eeg_channels_count, freq,
                 window_len, current_device=None):
        super(SSLModel, self).__init__()
        self.encoder = EncoderG(eeg_channels_count, freq=freq, window_len=window_len, current_device=current_device)
        self.decoder = ModelF(self.encoder.embedding_size, current_device=current_device)

    @property
    def embedding_size(self):
        return self.encoder.embedding_size

    @property
    def output_len(self):
        return self.decoder.output_len

    @torch.no_grad()
    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SubjectIdentifier(nn.Module):
    def __init__(self, num_subjects, embedding_size):
        super(SubjectIdentifier, self).__init__()
        self.num_subjects = num_subjects
        self.embedding_size = embedding_size
        self.linear = nn.Linear(self.embedding_size, self.num_subjects)
        # self.linear_1 = nn.Linear(self.embedding_size, self.embedding_size // 2)
        # self.relu_1 = nn.ReLU()
        # self.linear_2 = nn.Linear(self.embedding_size // 2, self.embedding_size // 2)
        # self.relu_2 = nn.ReLU()
        # self.linear_3 = nn.Linear(self.embedding_size // 2, self.embedding_size // 2)
        # self.relu_3 = nn.ReLU()
        # self.linear_4 = nn.Linear(self.embedding_size // 2, self.num_subjects)

    @torch.no_grad()
    def identity(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.linear(x)
        # x = self.linear_1(x)
        # x = self.relu_1(x)
        # x = self.linear_2(x)
        # x = self.relu_2(x)
        # x = self.linear_3(x)
        # x = self.relu_3(x)
        # x = self.linear_4(x)
        return x


class MoCoLoss(torch.nn.Module):
    def __init__(self, lamb=1.0):
        super(MoCoLoss, self).__init__()
        self.lamb = lamb
        self.info_nce = torch.nn.CrossEntropyLoss()
        self.regularization_term = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels, fixed_identity, subject_ids):
        info_nce_loss = self.info_nce(logits, labels)
        regularization_term_loss = self.regularization_term(fixed_identity, subject_ids)
        final_loss = info_nce_loss + self.lamb * regularization_term_loss
        return final_loss


__all__ = ["EncoderG", "ModelF", "SSLModel", "SubjectIdentifier", "MoCoLoss"]
