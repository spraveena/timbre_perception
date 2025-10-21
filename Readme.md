# timbre_perception — README

README to install core packages and get started training an RNN for temporal signal classification and reconstruction.

## Quick install (conda)
1. Create environment:
    conda create -n timbre python=3.9 -y
    conda activate timbre

2. Install packages:
    conda install numpy matplotlib pandas -y

3. Install PyTorch (choose CPU or appropriate CUDA via the official selector):
    # CPU
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
    # CUDA 11.8 example
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge

## Example requirements.txt
numpy>=1.24
matplotlib>=3.5
pandas>=1.5
torch>=2.0
torchvision>=0.15
torchaudio>=2.0

## Project layout (recommended)
- data/
  - train/
  - val/
- src/
  - train.py
  - eval.py
  - reconstruct.py
  - models/
     - rnn.py
- configs/
  - default.yaml
- requirements.txt

## Minimal commands (examples)
- Train:
  python src/train.py --data data/train --val data/val --config configs/default.yaml --epochs 50 --batch-size 64

- Evaluate/classify:
  python src/eval.py --checkpoint checkpoints/best.pt --data data/val

- Reconstruct signals:
  python src/reconstruct.py --checkpoint checkpoints/best.pt --input data/sample.wav --output out/recon.wav

Adjust flags to your scripts and config system (argparse / hydra / yaml).

## Notes for model training
- Use reproducible seeds (set torch.manual_seed, np.random.seed, random.seed).
- For sequences, prefer packed sequences or masking to handle variable length.
- Normalize inputs (mean/std) and keep dataset splits consistent.
- Check device availability in your scripts:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Troubleshooting
- If import errors appear for torch, ensure you installed the correct build for your CUDA version or use the CPU build.
- OOM on GPU: reduce batch size or sequence length; use gradient accumulation.
- Version conflicts: isolate with venv/conda and pin package versions in requirements.txt.

## Reproducible example (very small)
1. Install packages (see above).
2. Prepare a tiny dataset under data/train and data/val.
3. Run:
    python src/train.py --data data --epochs 10 --batch-size 32

That should get a simple RNN-based temporal classifier / reconstructor running.

License: add your preferred license file.