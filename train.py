import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image, write_jpeg
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from tqdm import tqdm

#────────────────────────────────────────────────────────────────────
from config import *  # import all config variables, CONSTANTS and TYPE ALIASES
from model import DDIM
from img_dataset import AnimeFaceDataset

from utils import EMA
#────────────────────────────────────────────────────────────────────
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

#════════════════════════════════════════════════════════════════════
assert IMG_FLODER.exists(), f"Image folder {IMG_FLODER} does not exist."
if not SAVE_PTH_PATH.exists():
    CONTINUE = False  # No checkpoint to continue
    print(f"Warning: Checkpoint {SAVE_PTH_PATH} don't exists.")
if not SAVE_IMG_PATH.exists():
    SAVE_IMG_PATH.mkdir(parents=True, exist_ok=True)

#────────────────────────────────────────────────────────────────────
face_dataset = AnimeFaceDataset(IMG_FLODER)
print("▤ The dataset capability is", len(face_dataset))
curr_epoch = 0  # init epoch as default

ddim_fid = FID(reset_real_features=False)

face_dataset.transform = transforms.Resize(face_dataset.size)
# temporary transform for FID evaluation (without data augmentation)
fid_dataloader = DataLoader( face_dataset, **DATALOADER_CONFIG)
fid_score = float('inf')
for img in tqdm(fid_dataloader, "Initialize FID of real data"):
    ddim_fid.update(img, real=True)
face_dataset.reset()

#────────────────────────────────────────────────────────────────────
ddim = DDIM(ARCH, TIMESTEP, TIME_DIM)
ddim_optim = optim.Adam(ddim.parameters(), lr=LR, betas=ADAM_BETAS)
scaler = GradScaler(DEVICE)
loss_logger = EMA()

dataloader = DataLoader( face_dataset, **DATALOADER_CONFIG)

if CONTINUE:
    assert SAVE_PTH_PATH.exists(), "No model detected, cannot continue training."

    print('Loading pre-trained model...')
    checkpoint: dict = torch.load(SAVE_PTH_PATH)
    curr_epoch = checkpoint['epoch'] + 1
    ddim.load_state_dict(checkpoint['ddim'])
    ddim_optim.load_state_dict(checkpoint['ddim_optim'])
    ddim_fid.load_state_dict(checkpoint['ddim_fid'])
    scaler.load_state_dict(checkpoint['scaler'])
    print('Start training from loaded model...')

for epoch in range(curr_epoch, curr_epoch+EPOCH):

    ddim.train()
    for x0 in tqdm(dataloader, "Train"):
        t = torch.randint(0, ddim.T, (BATCH_SIZE,), device=DEVICE)
        eps = torch.randn_like(x0)
        theta_t = ddim.theta[t].view(BATCH_SIZE,1,1,1)
        xt = theta_t.cos() * x0 + theta_t.sin() * eps

        with autocast(str(DEVICE)):
            x0_pred = ddim.predicter(xt, t)
            loss = nn.functional.mse_loss(x0_pred, x0)

        loss_logger.update(loss.item())

        scaler.scale(loss).backward()
        scaler.step(ddim_optim)
        scaler.update()
        ddim_optim.zero_grad()

    ddim.eval()
    test_img_path = SAVE_IMG_PATH / f'test_{epoch}.jpg'
    with torch.inference_mode():
        h = w = face_dataset.size
        x0_pred = ddim.sample((1,3,h,w), eta=ETA, tau=TAU)
        image: torch.Tensor = face_dataset.inv_trans(x0_pred[0])
        image = image.cpu() # compatible with CPU and GPU [IMPORTANT!!!]

        write_jpeg(image, test_img_path)

        if (epoch+1) % FID_T == 0:  # evaluate FID every 16 epochs
            for batch in tqdm(range(FID_BATCH), "Evaluating FID of generated data"):
                x0_pred = ddim.sample((BATCH_SIZE,3,h,w), eta=ETA, tau=TAU)
                img_pred = face_dataset.inv_trans(x0_pred)
                ddim_fid.update(img_pred, real=False)
            fid_score = ddim_fid.compute().item()
            ddim_fid.reset()  # reset FID generator features for the next epoch

    checkpoint = {
        'epoch': epoch,
        'ddim': ddim.state_dict(),
        'ddim_optim': ddim_optim.state_dict(),
        'loss': loss,
        'ddim_fid': ddim_fid.state_dict(),
        'scaler': scaler.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        # 'rng_state': torch.get_rng_state(),  # optional
    }

    torch.save(checkpoint, SAVE_PTH_PATH)

    m, s = loss_logger.value, loss_logger.stdev  # mean and std of training loss
    best, best_t = loss_logger.best

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"EPOCH {epoch:>3d} COMPLETE")
    print(f"|Train Loss: {m:.4f} ± {s:.4f} (Best: {best:.4f} in {best_t})")
    print(f"|BatchSize: {BATCH_SIZE} | LR: {LR} | Checkpoint: saved")
    print("───────────────────────────────────────────────────────────────────────")
    print(f"Preview: {test_img_path} | FID: {fid_score:.4f}")
    print("═══════════════════════════════════════════════════════════════════════\n")