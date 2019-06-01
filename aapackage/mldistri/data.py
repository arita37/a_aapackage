from __future__ import print_function

import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms


def import_data(name="", mode="train", node_id=0):
    if name == "mnist" and mode == "train":
        dataset = datasets.MNIST(
            "data-%d" % node_id,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset

    if name == "mnist" and mode == "test":
        dataset = datasets.MNIST(
            "data-%d" % node_id,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset
