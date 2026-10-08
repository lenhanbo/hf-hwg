from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def validate_mfm_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "backbone" not in checkpoint:
        raise KeyError(
            f"Checkpoint {path} không có key 'backbone'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    return checkpoint



import os


def get_huggingface_token():
    token = os.environ.get("HF_TOKEN")

    if token:
        return token

    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except (ImportError, KeyError):
        return None

def ensure_mfm_checkpoint(cfg):
    local_dir = Path(cfg.mfm_checkpoint.local_dir)
    filename = cfg.mfm_checkpoint.filename

    local_path = local_dir / filename

    # Gate 1: đã có file thì không tải lại.
    if local_path.is_file():
        print(f"[MFM] Found local checkpoint: {local_path}")
        validate_mfm_checkpoint(local_path)
        return str(local_path)

    print(f"[MFM] Local checkpoint not found: {local_path}")
    print(
        f"[MFM] Downloading "
        f"{cfg.mfm_checkpoint.repo_id}/{filename}"
    )

    token = get_huggingface_token()

    downloaded_path = hf_hub_download(
        repo_id=cfg.mfm_checkpoint.repo_id,
        filename=filename,
        repo_type="model",
        local_dir=str(local_dir),
        token=token,
    )

    # Gate 2: kiểm tra file sau khi tải.
    validate_mfm_checkpoint(downloaded_path)

    print(f"[MFM] Downloaded checkpoint: {downloaded_path}")

    return downloaded_path


from huggingface_hub import HfApi, hf_hub_download


def upload_mfm_checkpoint(
    checkpoint_path,
    repo_id,
    filename="latest.pth",
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy MFM checkpoint: "
            f"{checkpoint_path}"
        )

    # Kiểm tra đây thực sự là MFM checkpoint.
    validate_mfm_checkpoint(checkpoint_path)

    token = get_huggingface_token()

    if not token:
        raise RuntimeError(
            "Không tìm thấy HF_TOKEN. "
            "Hãy thêm HF_TOKEN vào Kaggle Secrets."
        )

    print(
        f"[MFM] Uploading {checkpoint_path} "
        f"to {repo_id}/{filename}"
    )

    api = HfApi(token=token)

    result = api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload latest MFM checkpoint",
    )

    print(f"[MFM] Upload completed: {result}")

    return result