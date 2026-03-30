from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
#--------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image, write_jpeg

from tqdm import tqdm

#--------------------------------------------------------------------
from model import DDPM
from model import ARCH, TIMESTEP, TIME_DIM

#--------------------------------------------------------------------
torch.set_default_device('cpu') # 'cuda:0  [IMPORTANT!!!]
DEVICE = torch.device('cpu') # 'cuda:0'
torch.set_default_dtype(torch.float32)

#--------------------------------------------------------------------
CONTINUE = False # 是否从上次中断的地方继续训练

EPOCH = 16
BATCH_SIZE = 8
LR = 0.00005

BETAS = (0.9, 0.999)

#--------------------------------------------------------------------
IMG_FLODER = Path(r"D:\CodeHub\Mydata\AnimeFace") # [IMPORTANT!!!]
SAVE_PATH  = Path(__file__).parent / 'ddpm_cos.pth'
SAVE_IMG_PATH = Path(__file__).parent / 'samples'

#--------------------------------------------------------------------
@dataclass
class AnimeFaceDataset(Dataset):
    floder: Path = IMG_FLODER
    size: int = 80

    # mean = (0.6881, 0.5887, 0.5722)
    # std = (0.2396, 0.2511, 0.2294)
    mean = (0.6946, 0.6517, 0.6813) # Quan_AnimeFace
    std = (0.2310, 0.2461, 0.2241)  # Quan_AnimeFace
    inv_std = tuple(1/std_i for std_i in std)
    inv_mean = tuple(-istdi*meani for istdi,meani in zip(inv_std,mean))

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, (0.9,1.0), (6/7,7/6)),
        # transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(0.05,0.05,0.05,0.02),
        transforms.RandomHorizontalFlip(),
        transforms.ToDtype(torch.float32,scale=True),
        # transforms.GaussianNoise(),
        transforms.Normalize(mean, std),
    ])

    inv_trans = transforms.Compose([
        transforms.Normalize(inv_mean,inv_std),
        transforms.Lambda(lambda x:torch.clamp(x,min=0.0,max=1.0)),
        transforms.ToDtype(torch.uint8,scale=True)
    ])

    def __post_init__(self):
        self.path: tuple[Path,...] = tuple(self.floder.iterdir())

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        img_path = self.path[index]
        img_t = decode_image(img_path, "RGB").to(device=DEVICE)
        return self.transform(img_t)

face_dataset = AnimeFaceDataset(IMG_FLODER)
# mini_face_dataset = Subset(face_dataset, list(range(128)))
print("The dataset capability is ",len(face_dataset))

#--------------------------------------------------------------------
ddpm = DDPM(ARCH, TIME_DIM, TIMESTEP)
ddpm_optim = optim.Adam(ddpm.parameters(), lr=LR, betas=BETAS)

dataloader = DataLoader(
        face_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
        generator=torch.Generator(device=DEVICE) # [IMPORTANT!!!]
    )

if CONTINUE:
    assert SAVE_PATH.exists(), "No pre-trained model detected, cannot continue training."

    print('Loading pre-trained model...')
    checkpoint: dict = torch.load(SAVE_PATH)
    ddpm.load_state_dict(checkpoint['ddpm'])
    ddpm_optim.load_state_dict(checkpoint['ddpm_optim'])
    print('Start training from loaded model...')

loss_logger = []

for epoch in range(EPOCH):

    ddpm.train() # 切换到训练模式
    for x0 in tqdm(dataloader, "Train"):
        t = torch.randint(0, ddpm.T, (BATCH_SIZE,), device=DEVICE)
        eps = torch.randn_like(x0)
        alpha_bar_t = ddpm.alpha_bar[t].view(BATCH_SIZE,1,1,1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        eps_pred = ddpm.denoise(xt, t)

        loss = nn.functional.mse_loss(eps_pred, eps)
        loss_logger.append(loss.item())

        loss.backward()
        ddpm_optim.step()
        ddpm_optim.zero_grad() # 梯度清零

    ddpm.eval()
    with torch.inference_mode():
        h = w = face_dataset.size
        x0_prod = ddpm.sample((1,3,h,w), DEVICE)
        image = AnimeFaceDataset.inv_trans(x0_prod[0])
        # image = image.cpu() [IMPORTANT!!!]

        write_jpeg(image, SAVE_IMG_PATH/'test.jpg')

    checkpoint = {
        'epoch': epoch,
        'ddpm': ddpm.state_dict(),
        'ddpm_optim': ddpm_optim.state_dict(),
        'loss': loss,
        # 'scheduler_state_dict': scheduler.state_dict(),
        # 'rng_state': torch.get_rng_state(),  # 可选但推荐
    }

    torch.save(checkpoint, SAVE_PATH)

    m,s = mean(loss_logger), pstdev(loss_logger)
    best = min(loss_logger)

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"EPOCH {epoch:>3d} COMPLETE")
    print(f"|Train Loss: {m:.4f} ± {s:.4f} (Best: {best:.4f}) | Valid Loss: ---")
    print(f"|BatchSize: {BATCH_SIZE} | LR: {LR} | Checkpoint: saved")
    print("───────────────────────────────────────────────────────────────────────")
    print(f"Preview: {SAVE_IMG_PATH/f'test_{epoch}.jpg'} | FID: --(↓2.15)")
    print("═══════════════════════════════════════════════════════════════════════\n")