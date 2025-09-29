import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from braindecode.models import EEGNeX
from ccd_windows import build_ccd_windows, split_ccd_by_subject
from steering import SteeringConfig, SteeringEngine


class EEGNeXRegressor(nn.Module):
    def __init__(self, n_chans=129, n_times=200, sfreq=100):
        super().__init__()
        self.backbone = EEGNeX(
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=1,
        )

    def forward(self, x):
        return self.backbone(x)


def _xy_from_batch(batch):
    if isinstance(batch, (tuple, list)):
        X, y = batch[0], batch[1]
    elif isinstance(batch, dict):
        X = batch.get("X") or batch.get("inputs") or next(iter(batch.values()))
        y = batch.get("y") or batch.get("target") or batch.get("labels")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    return X, y


def _to_device_float(X, y, device):
    X = X.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return X, y


def train_one_epoch(
    dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    target_space="rt",
    rt_min=0.2,
    rt_max=2.0,
):
    model.train()
    total_loss, sum_sq_err, n_samples = 0.0, 0.0, 0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        X, y = _xy_from_batch(batch)
        X, y = _to_device_float(X, y, device)

        if target_space == "logrt":
            y_loss = torch.log(torch.clamp(y, min=rt_min, max=rt_max) + 1e-6)
        else:
            y_loss = y

        optimizer.zero_grad(set_to_none=True)
        preds = model(X)
        loss = loss_fn(preds, y_loss)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

        if target_space == "logrt":
            preds_eval = torch.exp(preds.detach())
            y_eval = y.detach()
        else:
            preds_eval = preds.detach()
            y_eval = y.detach()

        diff = preds_eval.view(-1) - y_eval.view(-1)
        sum_sq_err += float(torch.sum(diff * diff).item())
        n_samples += int(y_eval.numel())

    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    return avg_loss, rmse


@torch.no_grad()
def valid_model(dataloader, model, loss_fn, device, desc="Valid", target_space="rt"):
    model.eval()
    total_loss, sum_sq_err, n_samples = 0.0, 0.0, 0
    ys = []
    for batch in tqdm(dataloader, desc=desc, leave=False):
        X, y = _xy_from_batch(batch)
        X, y = _to_device_float(X, y, device)

        if target_space == "logrt":
            y_loss = torch.log(y + 1e-6)
        else:
            y_loss = y

        preds = model(X)
        batch_loss = loss_fn(preds, y_loss)
        total_loss += float(batch_loss.item())

        if target_space == "logrt":
            preds_eval = torch.exp(preds)
            y_eval = y
        else:
            preds_eval = preds
            y_eval = y

        diff = preds_eval.view(-1) - y_eval.view(-1)
        sum_sq_err += float(torch.sum(diff * diff).item())
        n_samples += int(y_eval.numel())
        ys.append(y_eval.detach().view(-1).cpu())

    avg_loss = total_loss / max(len(dataloader), 1)
    rmse = (sum_sq_err / max(n_samples, 1)) ** 0.5
    y_true = torch.cat(ys)
    std_y = torch.std(y_true, unbiased=False).item()
    nrmse = rmse / (std_y + 1e-12)
    return avg_loss, rmse, nrmse


def parse_args():
    ap = argparse.ArgumentParser(
        description="EEGNeX → scalar RT regression on CCD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = ap.add_argument_group("data")
    data.add_argument(
        "--cache_root",
        type=str,
        required=True,
        help="Root dir containing R{i}_L100_bdf release folders.",
    )
    data.add_argument(
        "--mini",
        action="store_true",
        help="Use the small ‘mini’ subset for quick runs.",
    )

    target = ap.add_argument_group("target & loss")
    target.add_argument(
        "--target_space",
        choices=["rt", "logrt"],
        default="rt",
        help="Train on raw RT seconds or log(RT).",
    )
    target.add_argument(
        "--loss", choices=["huber", "mse"], default="huber", help="Regression loss."
    )
    target.add_argument(
        "--huber_delta",
        type=float,
        default=0.05,
        help="Huber δ; ignored if --loss=mse.",
    )
    target.add_argument(
        "--rt_min",
        type=float,
        default=0.2,
        help="Clamp min RT (used when target_space=logrt).",
    )
    target.add_argument(
        "--rt_max",
        type=float,
        default=2.0,
        help="Clamp max RT (used when target_space=logrt).",
    )

    steer = ap.add_argument_group("model steering")
    steer.add_argument(
        "--ref_ckpt",
        type=str,
        default="",
        help="Path to frozen reference model .pth/.pt",
    )
    steer.add_argument(
        "--steer_mode",
        choices=["none", "shift", "weight", "topk"],
        default="none",
        help="Model-steering strategy",
    )
    steer.add_argument(
        "--tau", type=float, default=0.05, help="Temperature for 'weight' mode"
    )
    steer.add_argument(
        "--topk_frac", type=float, default=0.25, help="Fraction for 'topk' mode"
    )
    steer.add_argument(
        "--mix_lambda", type=float, default=1.0, help="Blend for 'shift' mode"
    )
    steer.add_argument(
        "--distill_gamma",
        type=float,
        default=0.0,
        help="Optional tiny MSE to ref preds",
    )
    steer.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Train with steer_mode='none' for first N epochs",
    )

    opt = ap.add_argument_group("optimization")
    opt.add_argument("--batch_size", type=int, default=128)
    opt.add_argument("--epochs", type=int, default=50)
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight_decay", type=float, default=1e-5)
    opt.add_argument("--num_workers", type=int, default=2)

    misc = ap.add_argument_group("early stopping & I/O")
    misc.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Epochs without nRMSE improvement before stop.",
    )
    misc.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum nRMSE improvement to reset patience.",
    )
    misc.add_argument("--save_path", type=str, default="eegnex_rt.pt")

    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type}")

    print("[Data] Building CCD windows…")
    windows = build_ccd_windows(cache_root=Path(args.cache_root), mini=args.mini)

    train_set, valid_set, test_set = split_ccd_by_subject(windows)
    print(
        f"[Data] #windows — train: {len(train_set)}  valid: {len(valid_set)}  test: {len(test_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = EEGNeXRegressor(n_chans=129, n_times=200, sfreq=100).to(device)

    if args.loss == "huber":
        loss_fn = nn.HuberLoss(delta=0.05)
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - 1, 1)
    )

    best_nrmse, best_state = float("inf"), None
    patience, no_improve, min_delta = args.patience, 0, args.min_delta

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_rmse = train_one_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            target_space=args.target_space,
            rt_min=args.rt_min,
            rt_max=args.rt_max,
        )
        va_loss, va_rmse, va_nrmse = valid_model(
            valid_loader,
            model,
            loss_fn,
            device,
            desc="Valid",
            target_space=args.target_space,
        )
        scheduler.step()

        print(f"  train: loss={tr_loss:.4f}  rmse={tr_rmse:.4f}")
        print(f"  valid: loss={va_loss:.4f}  rmse={va_rmse:.4f}  nrmse={va_nrmse:.4f}")

        if va_nrmse < best_nrmse - min_delta:
            best_nrmse = va_nrmse
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"  ↳ new best nRMSE: {best_nrmse:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"Early stopping (patience={patience}). Best nRMSE={best_nrmse:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), Path(args.save_path))
    # print(f"[Save] Wrote: {Path(args.save_path).resolve()}")

    te_loss, te_rmse, te_nrmse = valid_model(
        test_loader, model, loss_fn, device, desc="Test", target_space=args.target_space
    )
    print(f"[Test] loss={te_loss:.4f}  rmse={te_rmse:.4f}  nrmse={te_nrmse:.4f}")


if __name__ == "__main__":
    main()
