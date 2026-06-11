from pathlib import Path
from typing import TypeAlias
from collections import namedtuple

import torch

from model import Res_Ch, Layer_Ch, Arch

#════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cpu")
DTYPE = torch.float32
PADDING_MODE = 'replicate'  # for conv layers

#────────────────────────────────────────────────────────────────────
CONTINUE = False

EPOCH = 16
BATCH_SIZE = 8
LR = 0.00005

ADAM_BETAS = (0.9, 0.999)

#────────────────────────────────────────────────────────────────────
IMG_FLODER = Path(r"E:\CodeHub\Mydata\AnimeFace") # [IMPORTANT!!!]
SAVE_PTH_PATH  = Path(__file__).parent / 'ddim_cos.pth'
SAVE_IMG_PATH = Path(__file__).parent / 'samples'

#════════════════════════════════════════════════════════════════════
ARCH: Arch = (
    Layer_Ch(64,  64,  64, 64),
    Layer_Ch(64,  128, 0, 128),
    Layer_Ch(128, 256, 0, 256),
    Layer_Ch(256, 256, 0, 512)
) # Diffusion/unet_diffusion.png

TIME_DIM: int = 512
TIMESTEP: int = 1000
TAU: list[int] = list(range(0, TIMESTEP+1, 40))
ETA: float = 0.0
"TAU is the list of time steps for sampling, a subset of range(TIMESTEP+1) and include 0 and TIMESTEP"

#════════════════════════════════════════════════════════════════════
DATALOADER_CONFIG = {
    'shuffle': True,
    'batch_size': BATCH_SIZE,
    'drop_last': True,
    'generator': torch.Generator(device=DEVICE) # [IMPORTANT!!!]
}

#════════════════════════════════════════════════════════════════════
FID_T = 32  # 32 Epoch/ number of evaluation steps for FID
FID_BATCH = 40  # 40 Batch per evaluation step for FID