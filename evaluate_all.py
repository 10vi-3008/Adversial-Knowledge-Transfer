"""
Re-evaluates all saved ARD+IGDM checkpoints with CORRECT AutoAttack scoring.
(Fixes the earlier bug where robust_acc was miscalculated as ~47-48% for all models.)

Usage:
    python3 evaluate_all.py
"""

import os
import sys
import json
import io
import contextlib
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from autoattack import AutoAttack

sys.path.insert(0, '/home/akshay_grp26/tanvi/IGDM')
from cifar100_models import *
from cifar100_models.wideresnet import WideResNet_28_10

NUM_CLASSES  = 100
RESULTS_FILE = './result_models/eval_aa_results.json'
AA_EPS       = 8 / 255.0

# ── All 12 combos and their checkpoint files ─────────────────────────────────
CHECKPOINTS = {
    'teacher=ViT-S_student=RES-18':    './result_models/ard_IGDM_ViT-S_RES-18_2026-06-25.pt',
    'teacher=ViT-S_student=WRN-28-10': './result_models/ard_IGDM_ViT-S_WRN-28-10_2026-06-27.pt',
    'teacher=ViT-B_student=RES-18':    './result_models/ard_IGDM_ViT-B_RES-18_2026-06-27.pt',
    'teacher=ViT-B_student=WRN-28-10': './result_models/ard_IGDM_ViT-B_WRN-28-10_2026-06-28.pt',
    'teacher=ViT-L_student=RES-18':    './result_models/ard_IGDM_ViT-L_RES-18_2026-06-26.pt',
    'teacher=ViT-L_student=WRN-28-10': './result_models/ard_IGDM_ViT-L_WRN-28-10_2026-06-27.pt',
    'teacher=RES-18_student=ViT-S':    './result_models/ard_IGDM_RES-18_ViT-S_2026-06-29.pt',
    'teacher=RES-18_student=ViT-B':    './result_models/ard_IGDM_RES-18_ViT-B_2026-06-29.pt',
    'teacher=RES-18_student=ViT-L':    './result_models/ard_IGDM_RES-18_ViT-L_2026-06-29.pt',
    'teacher=WRN-28-10_student=ViT-S': './result_models/ard_IGDM_WRN-28-10_ViT-S_2026-06-30.pt',
    'teacher=WRN-28-10_student=ViT-L': './result_models/ard_IGDM_WRN-28-10_ViT-L_2026-06-30.pt',
    'teacher=WRN-28-10_student=ViT-B': './result_models/ard_IGDM_WRN-28-10_ViT-B_2026-06-30.pt',
}

# which architecture each student is (CNN vs ViT, and exact name)
STUDENT_ARCH = {
    'teacher=ViT-S_student=RES-18':    ('cnn', 'RES-18'),
    'teacher=ViT-S_student=WRN-28-10': ('cnn', 'WRN-28-10'),
    'teacher=ViT-B_student=RES-18':    ('cnn', 'RES-18'),
    'teacher=ViT-B_student=WRN-28-10': ('cnn', 'WRN-28-10'),
    'teacher=ViT-L_student=RES-18':    ('cnn', 'RES-18'),
    'teacher=ViT-L_student=WRN-28-10': ('cnn', 'WRN-28-10'),
    'teacher=RES-18_student=ViT-S':    ('vit', 'vit_small_patch16_224'),
    'teacher=RES-18_student=ViT-B':    ('vit', 'vit_base_patch16_224'),
    'teacher=RES-18_student=ViT-L':    ('vit', 'vit_large_patch16_224'),
    'teacher=WRN-28-10_student=ViT-S': ('vit', 'vit_small_patch16_224'),
    'teacher=WRN-28-10_student=ViT-B': ('vit', 'vit_base_patch16_224'),
    'teacher=WRN-28-10_student=ViT-L': ('vit', 'vit_large_patch16_224'),
    'teacher=WRN-28-10_student=ViT-B': ('vit', 'vit_base_patch16_224'),
}


def build_student(kind, name):
    if kind == 'cnn':
        if name == 'RES-18':
            model = resnet18(num_classes=NUM_CLASSES)
        else:  # WRN-28-10
            model = WideResNet_28_10()
    else:  # vit
        import timm
        model = timm.create_model(name, pretrained=False,
                                  num_classes=NUM_CLASSES, img_size=32)
    return nn.DataParallel(model).cuda()


def run_autoattack(model, test_loader):
    """Correctly captures robust accuracy from AutoAttack's printed output."""
    model.eval()
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x); ys.append(y)
    x_total = torch.cat(xs, 0)
    y_total = torch.cat(ys, 0)

    aa = AutoAttack(model, norm='Linf', eps=AA_EPS, version='standard')

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        aa.run_standard_evaluation(x_total, y_total)
    output = buf.getvalue()
    print(output)  # still show progress to the terminal

    robust_acc = None
    for line in output.split('\n'):
        line_lower = line.lower()
        if line_lower.startswith('robust accuracy:'):
            val_str = line.split(':')[1].strip().replace('%', '')
            robust_acc = float(val_str) / 100
            break

    if robust_acc is None:
        raise RuntimeError(f"Could not parse robust accuracy from AA output:\n{output}")

    return robust_acc


def main():
    transform_test = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR100('../dataset', train=False,
                                download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=2)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"[→] Resuming from {RESULTS_FILE}")
    else:
        results = {}

    for key, ckpt_path in CHECKPOINTS.items():
        if key in results:
            print(f"[skip] {key}  AA={results[key]*100:.2f}%")
            continue

        if not os.path.exists(ckpt_path):
            print(f"[!] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {key}")
        print(f"{'='*60}")

        kind, arch_name = STUDENT_ARCH[key]
        model = build_student(kind, arch_name)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.module.load_state_dict(state_dict)
        model.eval()

        aa_score = run_autoattack(model, test_loader)
        print(f"[✓] {key}: AA = {aa_score*100:.2f}%")

        results[key] = aa_score
        os.makedirs('./result_models', exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # ── final table ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL CORRECTED RESULTS")
    print(f"{'='*65}")
    print(f"\n  ViT Teacher → CNN Student:")
    for k in CHECKPOINTS:
        if k.split('_student=')[1] in ('RES-18', 'WRN-28-10') and 'ViT' in k.split('_student=')[0]:
            if k in results:
                print(f"  {k:<45} {results[k]*100:.2f}%")
    print(f"\n  CNN Teacher → ViT Student:")
    for k in CHECKPOINTS:
        if 'ViT' in k.split('_student=')[1]:
            if k in results:
                print(f"  {k:<45} {results[k]*100:.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
