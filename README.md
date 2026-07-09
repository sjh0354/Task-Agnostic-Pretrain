<div align="center">

# Learning to Move Before Learning to Do: Task-Agnostic Pretraining for VLAs

Junhao Shi¹², Siyin Wang¹², Xiaopeng Yu¹, Li Ji¹², Jingjing Gong²†, Xipeng Qiu¹²†

¹Fudan University   ²Shanghai Innovation Institute

**ICML 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sjh0354.github.io/task_agnostic_pretrain)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2607.02466)
[![OpenReview](https://img.shields.io/badge/OpenReview-PDF-8c1b13.svg)](https://openreview.net/pdf?id=KQiNdlhknK)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/collections/Michael0354/task-agnostic-pretrain)

</div>



## 📖 Overview

Vision-Language-Action (VLA) models are fundamentally bottlenecked by the scarcity of expert demonstrations. We argue that this bottleneck stems from **conflating two distinct learning objectives**: acquiring *physical competence* (how to move) and acquiring *semantic alignment* (what to do). Crucially, only the latter requires language supervision.

Building on this **Decomposition Hypothesis**, we propose **Task-Agnostic Pretraining (TAP)**, a two-stage framework:

- **Stage 1 (Task-Agnostic Pretraining):** Learn transferable motor priors from cheap, unlabeled interaction data via a self-supervised **Inverse Dynamics** objective — no language, no task labels required.
- **Stage 2 (Task-Specific Alignment):** Ground these physical priors in language using a minimal set of expert demonstrations.

<div align="center">
<img src="images/tap_method.png" alt="TAP Framework Overview" width="90%"/>
</div>

### 🔥 Key Highlights

- 🧩 **Decomposition Hypothesis:** Decouples "how to move" from "what to do" via self-supervised Inverse Dynamics.
- 📈 **Mitigating the Data Wall:** Matches VLAs trained on **1M+ expert trajectories** using only ~30 hours of autonomous play + as few as **200 expert demonstrations**.
- 🛡️ **Real-World Robustness:** Retains up to **65% success** under severe background/viewpoint shifts, where internet-scale baselines collapse to **0%**.
- 💰 **Academic-Scale Compute:** Fully reproducible on a single 8×H100 node.


## 📊 Main Results

### SIMPLER Benchmark (WidowX Subset)

| Model | Avg-Partial | Avg-Entire | Avg-All |
|---|:---:|:---:|:---:|
| RT-1-X | 6.05% | 0.00% | 3.03% |
| OpenVLA | 14.48% | 1.03% | 7.75% |
| Nora | 32.84% | 7.29% | 20.06% |
| Octo | 42.30% | 20.33% | 31.31% |
| Standard BC (baseline) | 31.79% | 14.50% | 23.15% |
| **TAP-20k (Ours)** | **45.82%** | **20.82%** | **33.32%** |

### Real-World Robustness (WidowX-250s, 200 expert demos)

| Scenario | From Scratch | **TAP (Ours)** | NORA (SOTA) |
|---|:---:|:---:|:---:|
| Standard Setup | 38% | 58% | **75%** |
| Visual Distractors | 5% | **48%** | 50% |
| Background Texture Shift | 0% | **45%** | 33% |
| Viewpoint Variation | 0% | **20%** | 0% |

Please refer to our [project page](https://sjh0354.github.io/task_agnostic_pretrain) for the full results tables.



## 📁 Repository Structure

This codebase is built on top of [NORA](https://github.com/declare-lab/nora), with the SIMPLER evaluation environment integrated from [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA).

```
Task-Agnostic-Pretraining/
├── training/                  # Stage 1 (Inverse Dynamics) & Stage 2 (Alignment) training code
│   └── datasets/              # RLDS dataset configs and processing
├── inference/                 # Model inference utilities
├── evaluation/                # SIMPLER evaluation environment (pre-configured)
│   ├── ManiSkill2_real2sim/   # Real-to-sim simulator backend
│   ├── simpler_env/           # SIMPLER benchmark tasks & policies
│   ├── scripts/               # Evaluation launch scripts (includes run_nora.sh)
│   ├── tools/                 # Metric computation & result gathering
│   └── result/                # Evaluation outputs (videos, logs)
├── experiments/               # Experiment configs and logs
│   ├── libero/                # LIBERO experiments
│   ├── bridge/                # Bridge experiments
│   └── logs/
├── download_models/           # Base backbone checkpoints
│   ├── Qwen2.5-VL-3B-Instruct
│   └── Qwen2.5-VL-3B-added-action-tokens
└── images/                    # Figures and visualizations
```


## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sjh0354/Task-Agnostic-Pretrain.git
cd Task-Agnostic-Pretrain
```

### 2. Set Up the Base Environment (Training / Inference)

We follow the same environment setup as [NORA](https://github.com/declare-lab/nora):

```bash
conda create -n tap python=3.10 -y
conda activate tap

pip install -r requirements.txt
pip install -e .

# Flash-Attention 2 is highly recommended for training
pip install flash-attn --no-build-isolation
```

We recommend **PyTorch ≥ 2.3** with **CUDA 12.1**.

### 3. Set Up the Evaluation Environment (SIMPLER)

The `evaluation/` folder contains a pre-configured version of SimplerEnv with `ManiSkill2_real2sim` already integrated. You can either reuse the same conda environment (recommended for lightweight setups) or create a separate one following the [SimplerEnv installation guide](evaluation/README.md).

Minimal setup:

```bash
conda create -n simpler_env python=3.10 -y
conda activate simpler_env

pip install numpy==1.24.4

cd evaluation/ManiSkill2_real2sim
pip install -e .

cd ..
pip install -e .
```

For full installation (including Vulkan drivers, TensorFlow, etc.), please refer to [`evaluation/README.md`](evaluation/README.md).

### 4. Prepare the Base Backbone

TAP builds on **Qwen2.5-VL-3B-Instruct** with additional discrete action tokens appended to its vocabulary. Preparation is a two-step process:

**Step 1 — Download the original Qwen2.5-VL-3B-Instruct:**

```bash
cd download_models
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL-3B-Instruct
```

**Step 2 — Resize the embedding layer to add action tokens:**

```bash
# Run from the repo root
python download_models/embed_resize.py
```

This will produce `download_models/Qwen2.5-VL-3B-added-action-tokens/`, which serves as the starting point for Stage 1 pretraining.


## 🎯 Model Zoo

We release the full family of TAP checkpoints on 🤗 Hugging Face, including Stage 1–only pretrained backbones and their Stage 2 fine-tuned counterparts across three data scales (8k / 14k / 20k episodes), plus the pure behavior-cloning baseline.

👉 **[Full Collection on Hugging Face](https://huggingface.co/collections/Michael0354/task-agnostic-pretraining)**

| Model | Description |
|---|---|
| [`Michael0354/Task-Agnostic-Preatraining-baseline-stage2-only`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-baseline-stage2-only) | **Baseline** — Standard behavior cloning without task-agnostic pretraining |
| [`Michael0354/Task-Agnostic-Preatraining-8k-stage1-pretrain-only`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-8k-stage1-pretrain-only) | Stage 1 backbone pretrained with 8k task-agnostic episodes |
| [`Michael0354/Task-Agnostic-Preatraining-8k-stage2-finetuned`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-8k-stage2-finetuned) | Stage 2 fine-tuned on top of the 8k Stage 1 checkpoint |
| [`Michael0354/Task-Agnostic-Preatraining-14k-stage1-pretrain-only`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-14k-stage1-pretrain-only) | Stage 1 backbone pretrained with 14k task-agnostic episodes |
| [`Michael0354/Task-Agnostic-Preatraining-14k-stage2-finetuned`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-14k-stage2-finetuned) | Stage 2 fine-tuned on top of the 14k Stage 1 checkpoint |
| [`Michael0354/Task-Agnostic-Preatraining-20k-stage1-pretrain-only`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-20k-stage1-pretrain-only) | Stage 1 backbone pretrained with 20k task-agnostic episodes |
| [`Michael0354/Task-Agnostic-Preatraining-20k-stage2-finetuned`](https://huggingface.co/Michael0354/Task-Agnostic-Preatraining-20k-stage2-finetuned) | **Best model** — Stage 2 fine-tuned on top of the 20k Stage 1 checkpoint |

> 💡 The `stage1-pretrain-only` models are useful if you want to run your **own Stage 2 alignment** on custom data. The `stage2-finetuned` models are ready for SIMPLER evaluation and deployment.


## 🏋️ Training

All training scripts are located under `training/`. Both stages take **two positional arguments** — the data mixture name and the path to a starting checkpoint — followed by optional overrides.

### Stage 1 — Task-Agnostic Pretraining (Inverse Dynamics)

Pretrain the backbone with the self-supervised Inverse Dynamics objective on unlabeled interaction data (e.g., autonomous robot play, off-task trajectories). No language annotations are used at this stage.

```bash
cd training

# Usage: bash train_stage1.sh <data_mix> <model_path> [additional_args...]
bash task_agnostic_pretrain.sh \
    <your_task_agnostic_data_mix> \
    ../download_models/Qwen2.5-VL-3B-added-action-tokens
```

Example:

```bash
bash stage2_finetune.sh bridge_task_agnostic ../download_models/Qwen2.5-VL-3B-added-action-tokens
```

The resulting Stage 1 checkpoint will be saved under `experiments/logs/`.

### Stage 2 — Task-Specific Alignment

Fine-tune a Stage 1 checkpoint on a small set of language-annotated expert demonstrations to align the learned motor priors with semantic instructions.

```bash
cd training

# Usage: bash train_stage2.sh <data_mix> <model_path> [additional_args...]
bash train_stage2.sh \
    libero_10_no_noops \
    /path/to/directory/checkpoint/extracted_model/task_agnostic/steps_10000
```

You can override any hyperparameter via extra flags, e.g.:

```bash
bash train_stage2.sh custom_dataset /path/to/stage1_ckpt --per_device_batch_size 8
```

Datasets follow the **RLDS format**, consistent with OpenVLA / NORA. Dataset registrations live under `training/datasets/`.




## 🧪 Evaluation on SIMPLER

The `evaluation/` folder is a pre-configured fork of [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA) — everything is set up out of the box. To reproduce our SIMPLER results:

```bash
cd evaluation
bash scripts/run_nora.sh
```

> 📌 **Note:** The script is named `run_nora.sh` because our policy inference plugs into the NORA interface within SimplerEnv. Feel free to modify it to point to your own trained checkpoints.

Our WidowX evaluation covers the following four tasks:

- `widowx_spoon_on_towel`
- `widowx_carrot_on_plate`
- `widowx_stack_cube`
- `widowx_put_eggplant_in_basket`

Results (success rates, videos, and logs) are saved under `evaluation/result/`. Aggregate metrics can be computed via:

```bash
cd evaluation
python result_gather/collect_results.py
```

For general SimplerEnv usage (adding new policies, environment building, troubleshooting Vulkan, etc.), please see [`evaluation/README.md`](evaluation/README.md).


## 🙏 Acknowledgements

This codebase is built upon several excellent open-source projects. We sincerely thank the authors of:

- [NORA](https://github.com/declare-lab/nora) — Our training and inference code is directly forked and modified from NORA.
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) & [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA) — Simulation benchmark used in our evaluation.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language backbone.



## 📝 Citation

If you find TAP useful in your research, please consider citing:

```bibtex
@inproceedings{shi2026tap,
  title     = {Learning to Move Before Learning to Do: Task-Agnostic Pretraining for Vision-Language-Action Models},
  author    = {Shi, Junhao and Wang, Siyin and Yu, Xiaopeng and Ji, Li and Gong, Jingjing and Qiu, Xipeng},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

Please also consider citing the works we build upon:

```bibtex
@article{nora2024,
  title   = {NORA: A Small Open-Sourced Generalist Vision Language Action Model},
  author  = {Hung, Chia-Yu and Majumder, Navonil and others},
  journal = {arXiv preprint},
  year    = {2024}
}

@article{li24simpler,
  title   = {Evaluating Real-World Robot Manipulation Policies in Simulation},
  author  = {Li, Xuanlin and Hsu, Kyle and Gu, Jiayuan and others},
  journal = {arXiv preprint arXiv:2405.05941},
  year    = {2024}
}
```

---

<div align="center">
Made with ❤️ at Fudan University × Shanghai Innovation Institute
</div>
