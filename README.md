# Cross-Architecture Adversarial Robustness Distillation: ViTs and CNNs as Interchangeable Teacher-Student Models

This project investigates **cross-architecture knowledge distillation** for adversarial robustness, using the **ARD+IGDM** (Adversarial Robustness Distillation + Indirect Gradient Distillation Module) framework. We benchmark all combinations of Vision Transformers (ViT-S, ViT-B, ViT-L) and CNNs (ResNet-18, WideResNet-28-10) as interchangeable teacher and student models on CIFAR-100.

---

## Overview

Standard adversarial robustness distillation methods use CNN teachers for CNN students. This project extends that to study:
- Can a **ViT teacher** effectively transfer robustness to a **CNN student**?
- Can a **CNN teacher** effectively transfer robustness to a **ViT student**?
- How does **teacher size** (ViT-S vs ViT-B vs ViT-L) affect distillation quality?

---

## Project Structure

```
IGDM/
├── phase1_adversarial_pretrain.py   
├── phase2_distillation.py           
├── evaluate_all.py                  
├── attacks.py                      
├── rslad_loss.py                   
├── args.py                          
├── status.py                        
├── cifar100_models/                 
├── result_models/
│   ├── pretrained_teachers/        
│   │   ├── RES-18_2026-06-23.pt
│   │   ├── WRN-28-10_2026-06-25.pt
│   │   ├── ViT-S_2026-06-25.pt
│   │   ├── ViT-B_epoch_200.pt
│   │   ├── ViT-L_final.pt
│   │   └── pretrain_results.json
│   ├── ard_IGDM_ViT-S_RES-18_*.pt
│   ├── ard_IGDM_ViT-S_WRN-28-10_*.pt
│   ├── ... (all 12 distilled student checkpoints)
│   ├── distillation_aa_results.json
│   └── eval_aa_results.json        
```

---

## Setup

### Requirements

```bash
pip install torch torchvision timm autoattack robustbench wandb
```

### Dataset
CIFAR-100 is downloaded automatically to `../dataset/` on first run.

---

## Pipeline

### Phase 1 — Adversarial Pretraining of Teacher Models

Trains all 5 models (ResNet-18, WideResNet-28-10, ViT-S, ViT-B, ViT-L) using standard **PGD adversarial training** on CIFAR-100.

```bash
python3 phase1_adversarial_pretrain.py
```

- Saves checkpoints every 10 epochs to `result_models/pretrained_teachers/`
- Runs AutoAttack evaluation after each model finishes
- **Resumes automatically** if interrupted — just rerun the script

To run multiple models in parallel on different GPUs:
```bash
# GPU 0 — runs sequentially through all models
CUDA_VISIBLE_DEVICES=0 python3 phase1_adversarial_pretrain.py

# GPU 1 — run ViT-L separately (slowest model)
CUDA_VISIBLE_DEVICES=1 python3 -c "... ViT-L training script ..."
```

### Phase 2 — ARD+IGDM Distillation (12 Combinations)

Trains all 12 teacher-student combinations using the **ARD+IGDM loss**:

```
loss = KL(student(x+β·δ) || teacher(x))
     + α·(epoch/200)·KL(student(x+β·δ) - student(x-β·δ) || teacher(x+β·δ) - teacher(x-β·δ))
     + CrossEntropy(student(x), y)
```

```bash
# Run all 12 combos
CUDA_VISIBLE_DEVICES=0 python3 phase2_distillation.py

# Run only ViT-L combos on a separate GPU in parallel
CUDA_VISIBLE_DEVICES=1 python3 phase2_vitl_only.py
```

- Saves checkpoints to `result_models/` after each combo
- Saves AA scores to `result_models/distillation_aa_results.json` after each combo
- **Resumes automatically** if interrupted

### Evaluation

Re-evaluates all saved checkpoints with full AutoAttack:

```bash
python3 evaluate_all.py
```

Results saved to `result_models/eval_aa_results.json`.

---

## Key Findings

1. **ViT → CNN distillation works better than CNN → ViT** (8%+ vs 0.4–3.6% robust accuracy)

2. **Larger ViT teachers marginally improve CNN student robustness** — ViT-L → WRN-28-10 achieves the best result (8.60%)

3. **WideResNet is a better CNN teacher than ResNet-18** for ViT students (3.1–3.6% vs 0.4–0.9%), consistent with its higher solo robust accuracy

4. **ViT students are significantly harder to make robust via distillation** than CNN students — likely due to ViT's patch-based attention mechanism being less suited for small 32×32 CIFAR images

5. **Teacher quality is the bottleneck** — all results are bounded by the teacher's own robustness (11–22%), compared to SOTA RobustBench teachers which achieve 28–30%+ on CIFAR-100

---

## Notes

- All experiments use CIFAR-100, Linf threat model, ε = 8/255
- Random seed fixed to 0 for reproducibility
- Training: 200 epochs, SGD optimizer, lr=0.1 with decay at 50% and 75% of training
- ViT models use `img_size=32` to adapt patch embeddings to CIFAR resolution
