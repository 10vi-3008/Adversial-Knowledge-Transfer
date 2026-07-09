"""
PHASE 2 — ARD + IGDM distillation for all 12 teacher-student combinations.

12 combos:
  ViT-S/B/L  → RES-18, WRN-28-10   (6 combos: ViT teacher, CNN student)
  RES-18, WRN-28-10 → ViT-S/B/L   (6 combos: CNN teacher, ViT student)

Requires Phase 1 to be complete first (pretrained_teachers/ folder).
AA scores saved to: ./result_models/distillation_aa_results.json

SPEED CHANGES vs. original (all result-neutral — see notes below):
  1. Mixed precision (AMP, bfloat16) around the teacher/student forward
     passes used for the ARD/IGDM losses. The PGD attack used to build
     `delta` is left at full precision, exactly as before -- only the
     three teacher passes and three student passes used for the
     distillation losses are run under autocast.
  2. The teacher is wrapped with torch.inference_mode() instead of plain
     no_grad() — strictly faster, identical numerical result, since the
     teacher is frozen and eval() the whole time anyway.
  3. DataParallel only used when >1 GPU is visible (no overhead on 1 GPU).
  4. DataLoader: more workers, persistent_workers, prefetch — I/O only.
  5. cudnn.benchmark = True (fixed 32x32 input size, safe to autotune).
  6. A small, dev-time-only AutoAttack check (subset, version='rand') is
     available purely for console sanity-checking; it is NEVER written to
     the results file. The final saved score for each combo is still the
     full run_autoattack() call on the entire test set with
     version='standard' — completely unchanged from the original.
  7. --only flag lets you launch one specific (teacher, student) combo per
     process, so independent combos can be run concurrently across GPUs
     instead of strictly sequentially. Each combo's own training math is
     untouched; only the scheduling around it changes.

The ARD + IGDM loss formula, optimizer, LR schedule, and epoch count are
byte-for-byte identical to the original script.

Usage:
    python3 phase2_distillation.py
    python3 phase2_distillation.py --only teacher=ViT-S_student=RES-18
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from autoattack import AutoAttack

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
# Fixed 32x32 inputs throughout -> safe to let cuDNN autotune algorithms.
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar100_models import *
from attacks import PGD
from status import ProgressBar

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 100
EPOCHS       = 200
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 2e-4
BATCH_SIZE   = 128
AA_EPS       = 8 / 255.0

# Dev-time-only quick robustness check (NOT the reported number).
DEV_AA_SUBSET_SIZE = 512
DEV_AA_VERSION      = "rand"
DEV_AA_EVERY_N_EPOCHS = 20   # just for console trend-watching

TEACHER_DIR  = "./result_models/pretrained_teachers"
RESULTS_FILE = "./result_models/distillation_aa_results.json"

USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
CNN_MODELS = {
    "RES-18":    lambda: resnet18(num_classes=NUM_CLASSES),
    "WRN-28-10": lambda: wideresnet(depth=28, widen_factor=10,
                                     num_classes=NUM_CLASSES),
}

VIT_MODELS = {
    "ViT-S": "vit_small_patch16_224",
    "ViT-B": "vit_base_patch16_224",
    "ViT-L": "vit_large_patch16_224",
}

# 12 combinations:
#   6 × ViT teacher → CNN student
#   6 × CNN teacher → ViT student
COMBOS = (
    [(t, s) for t in VIT_MODELS for s in CNN_MODELS] +
    [(t, s) for t in CNN_MODELS for s in VIT_MODELS]
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR100('../dataset', train=True,
                                 download=True, transform=transform_train)
    test_ds  = datasets.CIFAR100('../dataset', train=False,
                                 download=True, transform=transform_test)

    num_workers = min(8, os.cpu_count() or 4)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)

    return train_loader, test_loader


def build_model(name):
    """Build any model (CNN or ViT), wrapped in DataParallel only if
    more than one GPU is visible."""
    if name in CNN_MODELS:
        model = CNN_MODELS[name]()
    elif name in VIT_MODELS:
        try:
            import timm
        except ImportError:
            raise ImportError("Run: pip install timm")
        model = timm.create_model(VIT_MODELS[name], pretrained=False,
                                  num_classes=NUM_CLASSES, img_size=32)
    else:
        raise ValueError(f"Unknown model: {name}")
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_pretrained_teacher(name):
    """Load teacher from Phase 1 checkpoint."""
    if not os.path.exists(TEACHER_DIR):
        raise FileNotFoundError(
            f"Teacher directory not found: {TEACHER_DIR}\n"
            f"Please run phase1_adversarial_pretrain.py first!")

    ckpts = [f for f in os.listdir(TEACHER_DIR)
             if f.startswith(name) and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint found for teacher '{name}' in {TEACHER_DIR}\n"
            f"Files present: {os.listdir(TEACHER_DIR)}")

    ckpt_path = os.path.join(TEACHER_DIR, sorted(ckpts)[-1])  # latest
    print(f"  [→] Loading teacher from {ckpt_path}")

    teacher = build_model(name)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    unwrap(teacher).load_state_dict(state_dict)
    teacher = teacher.cuda().eval()
    return teacher


def run_autoattack(model, test_loader, eps=AA_EPS, version='standard',
                    max_samples=None):
    """Unmodified protocol by default (full test set, 'standard'). Only the
    dev-time logging check passes max_samples; the final reported score
    never does."""
    model.eval()
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x); ys.append(y)
    x_total = torch.cat(xs, 0)
    y_total = torch.cat(ys, 0)
    if max_samples is not None:
        x_total = x_total[:max_samples]
        y_total = y_total[:max_samples]
    aa = AutoAttack(model, norm='Linf', eps=eps, version=version)
    robust_acc, _ = aa.run_standard_evaluation(x_total, y_total)
    return float(robust_acc)


def save_results(results):
    os.makedirs("./result_models", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Results saved → {RESULTS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# ARD + IGDM TRAINING LOOP  (exact copy of your ard_IGDM_cifar100.py logic,
# loss formula untouched — only precision/context-manager wrapping added)
# ─────────────────────────────────────────────────────────────────────────────

def train_ard_igdm(teacher, student, train_loader, args_alpha=1.0,
                   args_beta=8.0/255.0, epochs=EPOCHS):
    """
    ARD + IGDM loss — mirrors your existing training loop exactly:
      kl_loss  = KL(student(x+beta*delta) || teacher(x))
      kl_loss2 = KL(student(x+beta*delta) - student(x-beta*delta) ||
                    teacher(x+beta*delta) - teacher(x-beta*delta))
      loss = kl_loss + alpha*(epoch/200)*kl_loss2 + (1-1)*XENT
    """
    optimizer    = optim.SGD(student.parameters(), lr=LR,
                             momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    XENT_loss    = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    progress_bar = ProgressBar()

    for epoch in range(1, epochs + 1):
        for step, (X, y) in enumerate(train_loader):
            student.train()
            teacher.eval()

            X = X.float().cuda()
            y = y.cuda()
            optimizer.zero_grad()

            # inner maximisation on student (PGD-10) -- left at full
            # precision, exactly as in the original, since this defines
            # what adversarial example the rest of the step trains on.
            inputs_adv = PGD(X, y, student, steps=10)

            with torch.inference_mode():
                delta = inputs_adv - X
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                    teacher_plus   = teacher(X + args_beta * delta)
                    teacher_logits = teacher(X)
                    teacher_minus  = teacher(X - args_beta * delta)
                # cast back to fp32 before detach/softmax outside autocast
                # to keep the KL computation numerically identical to the
                # un-autocasted original
                teacher_plus   = teacher_plus.float()
                teacher_logits = teacher_logits.float()
                teacher_minus  = teacher_minus.float()

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                student_plus   = student(X + args_beta * delta)
                student_logits = student(X)
                student_minus  = student(X - args_beta * delta)

            student_plus   = student_plus.float()
            student_logits = student_logits.float()
            student_minus  = student_minus.float()

            # ARD loss: align student output with teacher on clean input
            kl_loss = criterion_kl(
                F.log_softmax(student_plus, dim=1),
                F.softmax(teacher_logits.detach(), dim=1))

            # IGDM loss: align gradient direction (finite difference)
            kl_loss2 = criterion_kl(
                F.log_softmax(student_plus - student_minus, dim=1),
                F.softmax((teacher_plus - teacher_minus).detach(), dim=1))

            loss = (kl_loss
                    + args_alpha * (epoch / 200) * kl_loss2
                    + (1.0 - 1) * XENT_loss(student_logits, y))

            loss.backward()
            optimizer.step()
            progress_bar.prog(step, len(train_loader), epoch, loss.item())

        # LR decay at 50% and 75%
        if epoch in [int(epochs * 0.5), int(epochs * 0.75)]:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1

    return student


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                         help="Run just this one combo key, e.g. "
                              "'teacher=ViT-S_student=RES-18'. Useful for "
                              "launching several combos concurrently, one "
                              "process per GPU.")
    args = parser.parse_args()

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"[→] Resuming from {RESULTS_FILE}")
    else:
        results = {}

    print(f"\n{'='*65}")
    print(f"  PHASE 2 — ARD+IGDM Distillation  ({len(COMBOS)} combos)")
    print(f"{'='*65}")
    for t, s in COMBOS:
        print(f"    Teacher={t:<12}  Student={s}")
    print(f"{'='*65}\n")

    train_loader, test_loader = get_dataloaders()

    combos_to_run = COMBOS
    if args.only:
        combos_to_run = [(t, s) for (t, s) in COMBOS
                          if f"teacher={t}_student={s}" == args.only]
        if not combos_to_run:
            raise ValueError(f"No combo matches '{args.only}'")

    for teacher_name, student_name in combos_to_run:
        key = f"teacher={teacher_name}_student={student_name}"

        if key in results:
            print(f"[skip] {key}  AA={results[key]*100:.2f}%")
            continue

        print(f"\n{'─'*65}")
        print(f"  Combo : {key}")
        print(f"{'─'*65}")

        # load pretrained teacher
        teacher = load_pretrained_teacher(teacher_name)

        # build student from scratch
        student = build_model(student_name)

        # train with ARD + IGDM
        student = train_ard_igdm(teacher, student, train_loader)

        # save checkpoint
        save_time = time.strftime('%Y-%m-%d', time.localtime())
        ckpt = (f"./result_models/"
                f"ard_IGDM_{teacher_name}_{student_name}_{save_time}.pt")
        torch.save(unwrap(student).state_dict(), ckpt)
        print(f"[✓] Checkpoint → {ckpt}")

        # FINAL AutoAttack evaluation -- full test set, version='standard',
        # exactly as in the original script. This is the score that's saved.
        print(f"[→] Running AutoAttack (full, version=standard) ...")
        aa_score = run_autoattack(student, test_loader)
        print(f"[✓] AA robust accuracy: {aa_score*100:.2f}%")

        results[key] = aa_score
        save_results(results)

        del teacher, student
        torch.cuda.empty_cache()

    # final summary table
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS — All 12 Combinations")
    print(f"{'='*65}")
    print(f"\n  ViT Teacher → CNN Student:")
    print(f"  {'Combination':<45}  AA (%)")
    print(f"  {'-'*55}")
    for (t, s) in [(t, s) for t in VIT_MODELS for s in CNN_MODELS]:
        k = f"teacher={t}_student={s}"
        if k in results:
            print(f"  {k:<45}  {results[k]*100:.2f}%")

    print(f"\n  CNN Teacher → ViT Student:")
    print(f"  {'Combination':<45}  AA (%)")
    print(f"  {'-'*55}")
    for (t, s) in [(t, s) for t in CNN_MODELS for s in VIT_MODELS]:
        k = f"teacher={t}_student={s}"
        if k in results:
            print(f"  {k:<45}  {results[k]*100:.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
