import torch
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 1
NUM_EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True
TRAIN_IMG_DIR = "data\\D1_ds1\\input"
TRAIN_MASK_DIR = "data\\D1_ds1\\target"
VAL_IMG_DIR = "data\\TT_ds1\\input"
VAL_MASK_DIR = "data\\TT_ds1\\target"

class Tversky(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Tversky, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.7, gamma=3/4):
        
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()
        fn = (targets * (1 - inputs)).sum()
        fp = ((1 - targets) * inputs).sum()

        t = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
        if gamma is None:
            return 1 - t
        else:
            return torch.pow(1 - t, gamma)

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transforms = A.Compose(
        [
            A.Normalize(
                mean=0.0,
                std=1.0,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=0.0,
                std=1.0,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)

    loss_fn = Tversky()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()