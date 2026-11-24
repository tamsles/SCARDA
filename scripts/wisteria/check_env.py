"""Step 1 helper: report torch / CUDA / GPU, require a GPU, download MNIST into dataset/mnist."""
import sys

import torch
import torchvision.datasets as dsets

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no GPU visible to torch")
print("gpu", torch.cuda.get_device_name(0), "count", torch.cuda.device_count())
dsets.MNIST("dataset/mnist", train=True, download=True)
dsets.MNIST("dataset/mnist", train=False, download=True)
print("MNIST ok")
