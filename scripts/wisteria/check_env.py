"""Step 1 helper: download MNIST into dataset/mnist, report torch / CUDA / GPU and require a GPU.

ALLOW_NO_GPU=1 skips the GPU requirement (for a first setup on a login node, which has no GPU).
"""
import os
import sys

import torch
import torchvision.datasets as dsets

dsets.MNIST("dataset/mnist", train=True, download=True)
dsets.MNIST("dataset/mnist", train=False, download=True)
print("MNIST ok")
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())
elif os.environ.get("ALLOW_NO_GPU") != "1":
    raise SystemExit("no GPU visible to torch")
