import torch
import argparse
from mfm.pretrain_mfm import main
from lib.utils import yaml2config
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="mfm pretraining"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/mfm_iam.yml'
    )
    args = parser.parse_args()
    cfg = yaml2config(args.config)

    print('Da nap xong config')

    main(cfg)