# Kaggle setup for HF-HWT

Kaggle Notebook is already an isolated environment. Do not create a Conda environment inside it and do not reinstall PyTorch/torchvision, because Kaggle supplies a CUDA-compatible build.

## 1. Create the notebook

1. Create a new Kaggle Notebook.
2. Open `Notebook options`.
3. Select a GPU accelerator.
4. Enable Internet temporarily so the notebook can clone GitHub.
5. Attach the HDF5 dataset through `Add input`.

## 2. Clone the repository

Run in the first cell:

```bash
!git clone https://github.com/lenhanbo/hf-hwg.git /kaggle/working/hf-hwg
%cd /kaggle/working/hf-hwg
```

For later sessions, always clone again because `/kaggle/working` starts from the notebook state rather than acting as the source repository.

## 3. Install only missing dependencies

```bash
!python -m pip install -q -r requirements-kaggle.txt
```

Do not install `torch`, `torchvision`, CUDA Toolkit or Conda packages in the notebook.

## 4. Verify CUDA and imports

```bash
!python tools/check_kaggle_env.py
```

The final line must be:

```text
Kaggle environment is ready.
```

## 5. Locate the attached dataset

```python
from pathlib import Path

for path in Path("/kaggle/input").glob("*/*"):
    print(path)
```

Use the printed directory in the MFM config, for example:

```yaml
data:
  root: '/kaggle/input/hf-hwt-dataset'
  train_file: 'train.hdf5'
  valid_file: 'test.hdf5'

output:
  root: '/kaggle/working/mfm_backbone'
```

`/kaggle/input` is read-only. Checkpoints, logs and preview images must be written under `/kaggle/working`.

## 6. Update code in a new session

If the repository is already present in the running session:

```bash
%cd /kaggle/working/hf-hwg
!git pull --ff-only
```

If `git pull` reports local notebook changes, clone a fresh copy instead of force-resetting it.
