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
    # freq_loss = frequency_loss()
    # recon_head = ReconstructionHead(input_dim=backbone.output_dim,
                                    # upscale_factor=8)

    model = MFM_Pretrainer(
        backbone=backbone,
        use_ocr=True,
        use_writer=True,
        n_class=cfg.multitask.n_class,
        n_writers=cfg.multitask.n_writer,
        lambda_ocr=cfg.multitask.lambda_ocr,
        lambda_writer=cfg.multitask.lambda_writer,
        radius_ratio=cfg.training.radius_ratio,
        p=cfg.training.low_pass_probability,
        upscale_factor=8,
    )
    
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
    # total_loss = 0.0
    loss_sums = {}
    total_samples = 0.0
    for i, batch in enumerate(loader):
        images, img_lens, label, label_lens, writer_ids = batch

        images = images.to(device,
                           non_blocking=True)

        loss, loss_dict = model(images, img_lens, label, label_lens, writer_ids)

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
        for name, value in loss_dict.items():
            if name not in loss_sums:
                loss_sums[name] = 0.0

            loss_sums[name] += (
                value.detach().item() * batch_size
            )
        total_samples += batch_size
        global_step += 1
        if i % 20 == 0:
            loss_message = " ".join(
                f"{name}={value.detach().item():.6f}"
                for name, value in loss_dict.items()
            )

            print(
                f"epoch={epoch} "
                f"batch={i}/{len(loader)} "
                f"{loss_message}"
            )

    mean_losses = {
        name: value / max(1, total_samples)
        for name, value in loss_sums.items()
    }

    return mean_losses, global_step


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, cfg):
    checkpoint = {
        "epoch" : epoch,
        'global_step' : global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': dict(cfg),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer,scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
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
        mean_losses, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, cfg.training.grad_clip, epoch, global_step, device)
        mean_message = " ".join(
            f"mean_{name}={value:.6f}"
            for name, value in mean_losses.items()
        )

        print(
            f"epoch={epoch} "
            f"{mean_message}"
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
        # print example
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
