import torch.nn as nn
import torch


class DownstreamNet(nn.Module):
    def __init__(self, embed_dim, num_classes):
        """
        Network for downstream prediction/classification tasks. Simply the embeder(s) and a final linear
        layer.

        embedders: list of embedding models (trained/untrained and trainable/frozen)
        num_classes: the total number of classes to be used in prediction/classification task
        """
        super(DownstreamNet, self).__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x


__all__ = ["DownstreamNet"]
