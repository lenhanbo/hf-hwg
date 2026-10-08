import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.datasets import (get_dataset, Hdf5Dataset)
from networks.module import SharedBackbone
from mfm.modules import (MFM_Pretrainer,ReconstructionHead)
from mfm.loss import frequency_loss
from mfm.optimizer import build_optimizer
from mfm.scheduler import build_scheduler
from mfm.utils import preview_frequency_mask
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(cfg, device):
    backbone = SharedBackbone(**cfg.SharedBackbone)
    loss = frequency_loss()
    recon_head = ReconstructionHead(input_dim=backbone.output_dim,
                                    upscale_factor=8)
    model = MFM_Pretrainer(backbone=backbone,
                           recon_head=recon_head,
                           loss_func=loss,
                           radius_ratio=cfg.training.radius_ratio,
                           p=cfg.training.low_pass_probability)
    
    return model.to(device)



def build_train_dataloader(cfg):
    dataset = get_dataset(
        cfg.dataset,
        cfg.training.dset_split
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=Hdf5Dataset.collect_fn,
        drop_last=True,
        pin_memory=True
    )
    return dataloader


def train_one_epoch(model, loader, optimizer, scheduler, grad_clip, epoch, global_step, device):
    model.train()
    total_loss = 0.0
    total_samples = 0.0
    for i, batch in enumerate(loader):
        images, img_lens, _, _, _ = batch

        images = images.to(device,
                           non_blocking=True)

        loss = model(images, img_lens)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        if grad_clip is not None: # clip độ lớn của gradient
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )
        optimizer.step()
        scheduler.step_update(global_step)
        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        global_step += 1
        if i % 20 == 0:
            print(
                f"epoch={epoch} "
                f"batch={i}/{len(loader)} "
                f"loss={loss.detach().item():.6f}"
            )

    mean_loss = total_loss/max(1, total_samples)
    return mean_loss, global_step


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, cfg):
    checkpoint = {
        "epoch" : epoch,
        'global_step' : global_step,
        'backbone': model.backbone.state_dict(),
        'reconstruction_head': model.recon_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': dict(cfg),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer,scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.backbone.load_state_dict(checkpoint['backbone'])
    model.recon_head.load_state_dict(checkpoint['reconstruction_head'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    scheduler.load_state_dict(checkpoint['scheduler'])
    return start_epoch, global_step



def main(cfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    if device.type  == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'do not support cuda'
        )
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)


    model = build_model(cfg, device)
    train_loader = build_train_dataloader(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))



    start_epoch = 1
    global_step = 0

    resume_path = cfg.training.resume

    if resume_path:
        if not os.path.exists(resume_path):
            raise RuntimeError('Khong co checkpoint !!')
        start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        print('Da load thanh cong checkpoint!!')
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        mean_loss, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.training.grad_clip, epoch, global_step, device)
        print(
            f"epoch={epoch} "
            f"mean_loss={mean_loss:.6f}"
        )

        latest_path = os.path.join(
            output_dir,
            "latest.pth",
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            cfg,
        )

        if epoch % cfg.training.save_every == 0:
            preview_batch = next(iter(train_loader))

            preview_imgs, preview_lens, _, _, _ = preview_batch

            preview_frequency_mask(preview_imgs, preview_lens,save_path=os.path.join(
            output_dir,
            "examples",
            f"masked_epoch_{epoch:04d}.png",
        ))

            epoch_path = os.path.join(
                output_dir,
                f"epoch_{epoch:04d}.pth",
            )

            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
            )
