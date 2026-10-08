import torch
from networks import get_model
# from networks.mode
# from pretrain_mfm import load_checkpoint
import os
import argparse
from datetime import datetime
from lib.utils import yaml2config
from mfm.download_checkpoints import ensure_mfm_checkpoint

def load_sharedbackbone_checkpoint(path, model, device):
    if not os.path.isfile(path):
        raise RuntimeError('Khong co checkpoint')
    backbone = torch.load(path, map_location=device, weights_only=False)['backbone']
    # if not os.path.exists(path):
    #     raise RuntimeError('Chua co checkpoint shared_backbone')
    model.models.S.load_state_dict(backbone)
    return model



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/fw_gan_iam.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    print(f"Config file: {args.config}")

    cfg = yaml2config(args.config)
    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))
    print(logdir)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        print(f"[INFO] CUDA available. Using device: {DEVICE} - {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("[INFO] CUDA not available. Falling back to CPU.")

    cfg['device'] = str(DEVICE)



    model = get_model(cfg.model)(cfg, logdir)

    
    # Check and load checkpoint
    epoch_done = 1
    if cfg.ckpt and os.path.exists(cfg.ckpt):
        print(f"Loading checkpoint from {cfg.ckpt}")
        epoch_done = model.load(cfg.ckpt, cfg.device)
    else:
        # path = os.path.join(cfg.mfm_output, 'latest.pth')
        path =ensure_mfm_checkpoint(cfg)
        load_sharedbackbone_checkpoint(path=path, model=model, device=DEVICE)
        print("No valid checkpoint found, starting from scratch.")

    model.train(epoch_done=epoch_done)
    

