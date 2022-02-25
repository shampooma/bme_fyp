import torch
from torch.utils.data import DataLoader
import helper_pytorch as H
from dataset import Dataset
import math
import numpy as np

# Values
run_name = "resnet"
run_id = 5

# Model
# model = H.segmenter(
#     img_height=352,
#     img_width=math.ceil(1250*np.pi/16)*16,
# )
# model = H.UACANet(
#     n_channels=1,
#     n_classes=4
# )
model = H.resnet18(
    capacity=64,
    n_classes=4,
    in_channels=1
)
# model = H.UNet(
#     capacity=64,
#     n_classes=4,
#     n_channels=1
# )
# model = H.UACANet(
#     n_channels=1,
#     n_classes=4,
# )
# model = H.CE_Net_(
#     num_channels=1,
#     num_classes=4,
#     resnet_version='resnet34',
# )

# Dataloaders
train_dataset = Dataset(
    split='train',
    do_transform=True,
)

valid_dataset = Dataset(
    split='valid',
    do_transform=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=True
)

# Train
H.train(
    run_name=run_name,
    run_id=run_id,
    model=model,
    loader_train=train_loader,
    loader_valid=valid_loader,
    n_classes=4,
    monitor_class=3,
    ob_size=1,
    eb_size=4,
    loss_function=H.dice_loss_torch,
    metric_function=H.dice_torch,
    ckpts_path="ckpts",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    repeat_per_epoch=10
)
