"""
PHASE 1 — Adversarial pretraining of teacher models using PGD-AT.
Trains ResNet-18, WideResNet-28-10, ViT-S, ViT-B, ViT-L on CIFAR-100.
Saves each model to ./result_models/pretrained_teachers/

SPEED CHANGES vs. original (all result-neutral — see notes below):
  1. Mixed precision (AMP, bfloat16) for the forward/backward passes.
     PGD attack generation still happens at full precision internally
     (autocast disabled inside the PGD call) so the adversarial examples
     produced are identical in spirit to the unmodified attack; only the
     surrounding model fwd/bwd passes are cast.
  2. DataParallel is now only applied when >1 GPU is visible. On a single
     GPU it added pure overhead with zero benefit.
  3. DataLoader: more workers, persistent_workers, pin_memory, prefetch —
     pure I/O changes, no effect on what gets computed.
  4. cudnn.benchmark = True when input sizes are fixed (they are: 32x32) —
     lets cuDNN autotune the fastest conv algorithm. This can only change
     *numerics* in the sense of selecting a different (but equally valid)
     algorithm; it does not change the training procedure. If you want
     bit-exact determinism instead of speed here, flip the flag back.
  5. A cheap "dev" AutoAttack check (small subset, version='rand') runs
     periodically during the last few epochs purely for console logging /
     sanity-checking convergence. It is NEVER written to the results file
     and NEVER used as the reported score. The final saved aa_score is
     still computed with the full, untouched run_autoattack() on the
     complete test set with version='standard', exactly as before.
  6. Independent models can now be launched as separate processes (see
     bottom of file) so you can train multiple models concurrently across
     GPUs instead of strictly sequentially. This changes wall-clock time
     only — each model's own training is unaffected by the others.

Nothing about the PGD-AT loss, optimizer, schedule, epochs, or final
evaluation protocol has changed.

Usage:
    pip install timm          # only needed once
    tmux new -s phase1
    python3 phase1_adversarial_pretrain.py                  # all models, sequential
    python3 phase1_adversarial_pretrain.py --only RES-18     # just one model
                                                               # (for running several
                                                               # in parallel processes,
                                                               # one per GPU)
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
# Set to False if you need strict determinism over speed.
torch.backends.cudnn.benchmark = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar100_models import *
from attacks import PGD, attack_pgd
from status import ProgressBar

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 100
EPOCHS       = 200
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 2e-4
BATCH_SIZE   = 128
EPS          = 8  / 255.0
ALPHA        = 2  / 255.0
PGD_STEPS    = 10
AA_EPS       = 8  / 255.0

# Dev-time-only quick robustness check (NOT the reported number).
DEV_AA_SUBSET_SIZE = 512        # small slice of the test set
DEV_AA_VERSION      = "rand"    # cheap attack, just for trend-watching

SAVE_DIR     = "./result_models/pretrained_teachers"
RESULTS_FILE = "./result_models/pretrained_teachers/pretrain_results.json"

USE_AMP      = torch.cuda.is_available()
AMP_DTYPE    = torch.bfloat16  # wider range than fp16, safer for KL/softmax terms

# ─────────────────────────────────────────────────────────────────────────────
# MODELS TO PRETRAIN
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "RES-18":    ("cnn", lambda: resnet18(num_classes=NUM_CLASSES)),
    "WRN-28-10": ("cnn", lambda: wideresnet(depth=28, widen_factor=10,
                                             num_classes=NUM_CLASSES)),
    "ViT-S":     ("vit", "vit_small_patch16_224"),
    "ViT-B":     ("vit", "vit_base_patch16_224"),
    "ViT-L":     ("vit", "vit_large_patch16_224"),
}


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

    # num_workers / persistent_workers / prefetch_factor are pure I/O
    # pipeline tuning -- they change how fast batches arrive, not their
    # content, so training results are unaffected.
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
    kind, spec = MODELS[name]
    if kind == "cnn":
        model = spec()
    else:
        try:
            import timm
        except ImportError:
            raise ImportError("Run: pip install timm")
        model = timm.create_model(spec, pretrained=False,
                                  num_classes=NUM_CLASSES, img_size=32)
    model = model.cuda()
    # Only wrap in DataParallel if there's actually more than one GPU to
    # parallelize over -- on a single GPU this is pure overhead.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def unwrap(model):
    """Get the underlying module whether or not DataParallel was used."""
    return model.module if isinstance(model, nn.DataParallel) else model


def run_autoattack(model, test_loader, eps=AA_EPS, version='standard',
                    max_samples=None):
    """
    Unmodified evaluation protocol when called with defaults: full test
    set, version='standard'. max_samples is only ever passed by the
    dev-time logging check below -- the final reported score always calls
    this with max_samples=None (i.e. the complete 10k test set), identical
    to the original script.
    """
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
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Results saved → {RESULTS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# PGD ADVERSARIAL TRAINING LOOP  (mirrors your ard_IGDM_cifar100.py style)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_model(model_name, train_loader, test_loader):
    print(f"\n{'='*65}")
    print(f"  Pretraining: {model_name}")
    print(f"{'='*65}")

    model = build_model(model_name)
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    XENT_loss   = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    progress_bar = ProgressBar()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for step, (X, y) in enumerate(train_loader):
            X = X.float().cuda()
            y = y.cuda()
            optimizer.zero_grad()

            # PGD inner maximisation (standard adversarial training).
            # Left at full precision -- the attack itself is untouched,
            # only the surrounding clean fwd/bwd passes below are cast.
            inputs_adv = PGD(X, y, model, steps=PGD_STEPS)

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                logits     = model(X)
                logits_adv = model(inputs_adv)
                # standard PGD-AT loss: cross-entropy on adversarial examples
                loss = XENT_loss(logits_adv, y)

            loss.backward()
            optimizer.step()
            progress_bar.prog(step, len(train_loader), epoch, loss.item())

        # LR decay at 50% and 75%
        if epoch in [int(EPOCHS * 0.5), int(EPOCHS * 0.75)]:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1

        # quick PGD-20 eval at last 10 epochs (unchanged metric/protocol,
        # just running under autocast for speed during the forward passes
        # used purely for logging clean/PGD20 accuracy each epoch)
        if epoch > EPOCHS - 10:
            model.eval()
            test_accs, test_accs_adv = [], []
            for test_X, test_y in test_loader:
                test_X = test_X.float().cuda()
                test_y = test_y.cuda()
                test_adv = attack_pgd(model, test_X, test_y,
                                      attack_iters=20,
                                      step_size=2.0/255.0,
                                      epsilon=8.0/255.0)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits     = model(test_X)
                    logits_adv = model(test_adv)
                preds     = logits.argmax(1)
                preds_adv = logits_adv.argmax(1)
                test_accs.extend((preds == test_y).cpu().tolist())
                test_accs_adv.extend((preds_adv == test_y).cpu().tolist())

            clean_acc = np.mean(test_accs)
            pgd_acc   = np.mean(test_accs_adv)
            print(f"\n  Epoch {epoch} | Clean: {clean_acc*100:.2f}%"
                  f" | PGD20: {pgd_acc*100:.2f}%")

            # Dev-only cheap AA check, console logging only -- never saved,
            # never used as the reported aa_score. Comment out if you'd
            # rather skip even this during training.
            try:
                dev_aa = run_autoattack(model, test_loader,
                                         version=DEV_AA_VERSION,
                                         max_samples=DEV_AA_SUBSET_SIZE)
                print(f"  [dev-check] AA(rand, n={DEV_AA_SUBSET_SIZE}): {dev_aa*100:.2f}%"
                      f"  (trend-only, NOT the final reported score)")
            except Exception as e:
                print(f"  [dev-check] skipped ({e})")

    # save checkpoint
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_time = time.strftime('%Y-%m-%d', time.localtime())
    ckpt_path = f"{SAVE_DIR}/{model_name}_{save_time}.pt"
    torch.save(unwrap(model).state_dict(), ckpt_path)
    print(f"[✓] Model saved → {ckpt_path}")

    # FINAL AutoAttack evaluation -- full test set, version='standard',
    # exactly as in the original script. This is the score that gets saved.
    print(f"[→] Running AutoAttack for {model_name} (full, version=standard) ...")
    aa_score = run_autoattack(model, test_loader)
    print(f"[✓] AA robust accuracy: {aa_score*100:.2f}%")

    del model
    torch.cuda.empty_cache()

    return ckpt_path, aa_score


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                         help="Train just this one model name "
                              "(e.g. --only ViT-S). Useful for running "
                              "several models concurrently, one process "
                              "per GPU, instead of sequentially.")
    args = parser.parse_args()

    # resume from partial run
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"[→] Resuming from {RESULTS_FILE}")
    else:
        results = {}

    train_loader, test_loader = get_dataloaders()

    model_names = [args.only] if args.only else list(MODELS.keys())
    if args.only and args.only not in MODELS:
        raise ValueError(f"Unknown model '{args.only}'. Choices: {list(MODELS.keys())}")

    for model_name in model_names:
        if model_name in results:
            print(f"[skip] {model_name} already pretrained"
                  f" (AA={results[model_name]['aa_score']*100:.2f}%)")
            continue

        ckpt_path, aa_score = train_one_model(
            model_name, train_loader, test_loader)

        results[model_name] = {
            "checkpoint": ckpt_path,
            "aa_score":   aa_score
        }
        save_results(results)

    # summary
    print(f"\n{'='*65}")
    print(f"  PHASE 1 COMPLETE — Pretrained Teacher Summary")
    print(f"{'='*65}")
    print(f"  {'Model':<15}  AA (%)   Checkpoint")
    print(f"  {'-'*60}")
    for name, info in results.items():
        print(f"  {name:<15}  {info['aa_score']*100:.2f}%   {info['checkpoint']}")
    print(f"{'='*65}\n")
    print("  → Now run phase2_distillation.py")


if __name__ == "__main__":
    main()
