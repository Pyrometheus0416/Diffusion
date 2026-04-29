from dataclasses import dataclass
from pathlib import Path
#--------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler

import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image, write_jpeg
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from tqdm import tqdm

#--------------------------------------------------------------------
from model import DDIM
from model import ARCH, TIMESTEP, TIME_DIM
from img_dataset import AnimeFaceDataset

from utils import EMA
#--------------------------------------------------------------------
torch.set_default_device('cpu') # 'cuda:0  [IMPORTANT!!!]
DEVICE = torch.device('cpu') # 'cuda:0'
torch.set_default_dtype(torch.float32)

#--------------------------------------------------------------------
CONTINUE = False

EPOCH = 16
BATCH_SIZE = 8
LR = 0.00005

BETAS = (0.9, 0.999)

#--------------------------------------------------------------------
IMG_FLODER = Path(r"E:\CodeHub\Mydata\AnimeFace") # [IMPORTANT!!!]
SAVE_PTH_PATH  = Path(__file__).parent / 'ddim_cos.pth'
SAVE_IMG_PATH = Path(__file__).parent / 'samples'

assert IMG_FLODER.exists(), f"Image folder {IMG_FLODER} does not exist. Please check the path."
if not SAVE_PTH_PATH.exists():
    CONTINUE = False  # No checkpoint to continue
    print(f"Warning: Checkpoint {SAVE_PTH_PATH} already exists. "
          "It will be overwritten since CONTINUE is set to False.")
if not SAVE_IMG_PATH.exists():
    SAVE_IMG_PATH.mkdir(parents=True, exist_ok=True)

#--------------------------------------------------------------------
face_dataset = AnimeFaceDataset(IMG_FLODER)
# face_dataset = Subset(face_dataset, list(range(128)))
print("▤ The dataset capability is",len(face_dataset))

ddim_fid = FID(feature=64, reset_real_features=False)

if not CONTINUE:
    curr_epoch = 0
    # init the fid with real dataset
    face_dataset.transform = transforms.Resize(face_dataset.size)
    # temporary transform for FID evaluation (without data augmentation)
    fid_dataloader = DataLoader(
        face_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
        generator=torch.Generator(device=DEVICE) # [IMPORTANT!!!]
    )
    ddim_fid.reset()
    for img in tqdm(fid_dataloader, "Evaluating FID of real data"):
        ddim_fid.update(img, real=True)
    face_dataset.reset()

#--------------------------------------------------------------------

ddim = DDIM(ARCH, TIME_DIM, TIMESTEP)
ddim_optim = optim.Adam(ddim.parameters(), lr=LR, betas=BETAS)
scaler = GradScaler(DEVICE)
loss_logger = EMA()

dataloader = DataLoader(
        face_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
        generator=torch.Generator(device=DEVICE) # [IMPORTANT!!!]
    )

if CONTINUE:
    assert SAVE_PTH_PATH.exists(), "No pre-trained model detected, cannot continue training."

    print('Loading pre-trained model...')
    checkpoint: dict = torch.load(SAVE_PTH_PATH)
    curr_epoch = checkpoint['epoch'] + 1
    ddim.load_state_dict(checkpoint['ddim'])
    ddim_optim.load_state_dict(checkpoint['ddim_optim'])
    ddim_fid.load_state_dict(checkpoint['ddim_fid'])
    print('Start training from loaded model...')

for epoch in range(curr_epoch, curr_epoch+EPOCH):

    ddim.train() # 切换到训练模式
    for x0 in tqdm(dataloader, "Train"):
        t = torch.randint(0, ddim.T, (BATCH_SIZE,), device=DEVICE)
        eps = torch.randn_like(x0)
        alpha_bar_t = ddim.alpha_bar[t].view(BATCH_SIZE,1,1,1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        eps_pred = ddim.noise_predicter(xt, t)

        loss = nn.functional.mse_loss(eps_pred, eps)
        loss_logger.update(loss.item())

        loss.backward()
        ddim_optim.step()
        ddim_optim.zero_grad() # 梯度清零

    ddim.eval()
    with torch.inference_mode():
        # h = w = face_dataset.size
        h = w = 80
        x0_prod = ddim.sample((1,3,h,w))
        image = AnimeFaceDataset.inv_trans(x0_prod[0])
        # image = image.cpu() [IMPORTANT!!!]

        write_jpeg(image, SAVE_IMG_PATH/'test.jpg')

        for batch in tqdm(range(len(face_dataset) // BATCH_SIZE), "Evaluating FID of generated data"):
            x0_prod = ddim.sample((BATCH_SIZE,3,h,w))
            img_prod = face_dataset.inv_trans(x0_prod)
            ddim_fid.update(img_prod, real=False)
        fid_score = ddim_fid.compute().item()
        # ddim_fid.reset()  # reset FID generator features for the next epoch

    checkpoint = {
        'epoch': epoch,
        'ddim': ddim.state_dict(),
        'ddim_optim': ddim_optim.state_dict(),
        'loss': loss,
        'ddim_fid': ddim_fid.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        # 'rng_state': torch.get_rng_state(),  # 可选但推荐
    }

    torch.save(checkpoint, SAVE_PTH_PATH)

    m, s = loss_logger.value, loss_logger.deviation**0.5  # mean and std of training loss
    best = loss_logger.best

    test_img_path = SAVE_IMG_PATH / f'test_{epoch}.jpg'
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"EPOCH {epoch:>3d} COMPLETE")
    print(f"|Train Loss: {m:.4f} ± {s:.4f} (Best: {best:.4f}) | Valid Loss: ---")
    print(f"|BatchSize: {BATCH_SIZE} | LR: {LR} | Checkpoint: saved")
    print("───────────────────────────────────────────────────────────────────────")
    print(f"Preview: {test_img_path} | FID: {fid_score:.4f}")
    print("═══════════════════════════════════════════════════════════════════════\n")
