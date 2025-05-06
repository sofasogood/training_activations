# GPT-2 FSDP Training & Activation Visualization

This repository provides a full pipeline for distributed training of GPT-2 using PyTorch's Fully Sharded Data Parallel (FSDP), with tools for capturing, saving, and interactively visualizing internal activations for interpretability and research.

---

## Features

- **Distributed Training**: Train GPT-2 with FSDP and mixed precision on SLURM clusters.
- **Activation Capture**: Save block-4 activations and input tokens at regular intervals.
- **Checkpoints & Logging**: Robust checkpointing and SLURM log management.
- **Interactive Visualization**: Streamlit app for exploring activations, neuron statistics, and training progress.
- **Utilities**: Tools for tensor shape explanation and more.

---

## Directory Structure

```
.
├── train_experiment/
│   ├── train_fsdp.py                # Main FSDP training script
│   ├── run_fsdp.sh                  # SLURM job script for training
│   ├── inspect_activations_advanced.py  # Streamlit app for activation analysis
│   ├── activations/                 # Saved activation tensors
│   ├── checkpoints/                 # Model checkpoints
│   ├── logs/                        # SLURM and training logs
│   └── utils/
│       └── shape_explainer.py       # Tensor shape explanation utility
├── fsdp_tiny.sbatch                 # Example SLURM batch script
├── train_fsdp.py                    # (alt) FSDP training script
└── miniconda.sh                     # Miniconda installer (if needed)
```

---

## Getting Started

### 1. Environment Setup

- Install dependencies (PyTorch, transformers, datasets, streamlit, etc.)
- (Recommended) Use the provided `miniconda.sh` to set up a conda environment.

```bash
bash miniconda.sh
conda create -n fsdp python=3.10
conda activate fsdp
pip install torch torchvision torchaudio
pip install transformers datasets streamlit plotly
```

### 2. Training with FSDP

Submit the SLURM job:

```bash
cd train_experiment
sbatch run_fsdp.sh
```

- This will train GPT-2 on the TinyStories dataset (1% split), saving checkpoints and activations every 20 steps.
- Checkpoints and activations are saved in `train_experiment/checkpoints/` and `train_experiment/activations/`.

### 3. Visualizing Activations

After training, launch the Streamlit app:

```bash
cd train_experiment
streamlit run inspect_activations_advanced.py
```

- Explore neuron activations, statistics, and training progress interactively in your browser.

---

## Key Scripts

- **train_fsdp.py**: Distributed training with FSDP, activation and checkpoint saving.
- **run_fsdp.sh**: SLURM job script (edit for your cluster if needed).
- **inspect_activations_advanced.py**: Streamlit app for deep-dive analysis of saved activations.
- **utils/shape_explainer.py**: Utility for explaining tensor shapes (used in the app).

---

## Outputs

- **Activations**: `train_experiment/activations/act_block4_output_step_*.pt`
- **Checkpoints**: `train_experiment/checkpoints/checkpoint_step_*.pt`
- **Logs**: `train_experiment/logs/`

---

## Customization

- Change model, dataset, or training parameters in `train_fsdp.py`.
- Modify visualization or add new analysis in `inspect_activations_advanced.py`.

---

## Troubleshooting

- Ensure all dependencies are installed in your environment.
- For SLURM issues, check logs in `train_experiment/logs/`.
- For visualization errors, ensure activations and checkpoints exist.

---

## License

MIT License