import argparse
import copy
import time
import random
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from braindecode.models import (
    EEGNeX,
    BDTCN,
    TIDNet,
    ATCNet,
    EEGConformer,
    ShallowFBCSPNet,
    Deep4Net,
    Labram,
)

from ccd_windows import build_ccd_windows, split_ccd_by_subject


ARCH_CHOICES = [
    "eegnex",
    "bdtcn",
    "tidnet",
    "atcnet",
    "eegconformer",
    "shallow",
    "deep4",
    "labram",
]


def _fmt_float(x, n=4):
    return f"{x:.{n}f}"


def _print_run_header(args, model, n_train, n_val, n_test):
    print(
        "Run | "
        f"arch={args.arch} loss={args.loss} target={args.target_space} "
        f"bs={args.batch_size} lr={args.lr:.2e} wd={args.weight_decay:.1e} "
        f"epochs={args.epochs} patience={args.patience} "
        f"seed={args.seed} deterministic={args.deterministic} "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M | "
        f"splits: train={n_train} valid={n_val} test={n_test}"
    )


def _epoch_line(
    epoch,
    total_epochs,
    lr,
    t_sec,
    tr_loss,
    tr_rmse,
    va_loss,
    va_rmse,
    va_nrmse,
    best_nrmse,
    best_epoch,
    no_improve,
    patience,
):
    return (
        f"Epoch {epoch:>3}/{total_epochs:<3} | "
        f"lr={lr:.2e} | t={t_sec:5.1f}s | "
        f"train: loss={_fmt_float(tr_loss)} rmse={_fmt_float(tr_rmse)} | "
        f"valid: loss={_fmt_float(va_loss)} rmse={_fmt_float(va_rmse)} nrmse={_fmt_float(va_nrmse)} | "
        f"best={_fmt_float(best_nrmse)}@{best_epoch} | patience={no_improve}/{patience}"
    )


def _set_seeds(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _auto_seed() -> int:
    return int.from_bytes(os.urandom(8), "little", signed=False) & 0xFFFFFFFF


def make_model(arch: str, n_chans=129, n_times=200, sfreq=100):
    arch = arch.lower()
    if arch == "eegnex":
        return EEGNeX(n_chans=n_chans, n_times=n_times, sfreq=sfreq, n_outputs=1)
    if arch == "bdtcn":
        return BDTCN(n_chans=n_chans, n_times=n_times, sfreq=sfreq, n_outputs=1)
    if arch == "tidnet":
        return TIDNet(n_chans=n_chans, n_times=n_times, sfreq=sfreq, n_outputs=1)
    if arch == "atcnet":
        return ATCNet(
            n_chans=n_chans,
            n_outputs=1,
            input_window_seconds=float(n_times) / float(sfreq),
            sfreq=sfreq,
        )
    if arch == "eegconformer":
        return EEGConformer(n_chans=n_chans, n_times=n_times, sfreq=sfreq, n_outputs=1)
    if arch == "shallow":
        return ShallowFBCSPNet(
            n_chans=n_chans, n_times=n_times, sfreq=sfreq, n_outputs=1
        )
    if arch == "deep4":
        return Deep4Net(n_chans=n_chans, n_times=n_times, sfreq=sfreq, n_outputs=1)
    if arch == "labram":
        return Labram(n_times=n_times, n_outputs=1, sfreq=sfreq, chs_info=None)
    raise ValueError(f"Unknown arch: {arch}")


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
    y_true_list, y_pred_list = [], []
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

        y_true_list.append(y_eval.detach().view(-1).cpu().numpy())
        y_pred_list.append(preds_eval.detach().view(-1).cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.array([])
    rmse = (
        float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else float("inf")
    )
    nrmse = (
        float(rmse / (np.std(y_true, ddof=0) + 1e-12)) if y_true.size else float("inf")
    )
    return avg_loss, rmse, nrmse


def parse_args():
    ap = argparse.ArgumentParser(
        description="braindecode_rt: multi-architecture RT regression on CCD using Braindecode models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = ap.add_argument_group("data")
    data.add_argument("--cache_root", type=str, required=True)
    data.add_argument("--mini", action="store_true")

    target = ap.add_argument_group("target & loss")
    target.add_argument("--target_space", choices=["rt", "logrt"], default="rt")
    target.add_argument("--loss", choices=["huber", "mse"], default="huber")
    target.add_argument("--huber_delta", type=float, default=0.05)
    target.add_argument("--rt_min", type=float, default=0.2)
    target.add_argument("--rt_max", type=float, default=2.0)

    opt = ap.add_argument_group("optimization")
    opt.add_argument("--batch_size", type=int, default=128)
    opt.add_argument("--epochs", type=int, default=100)
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight_decay", type=float, default=1e-5)
    opt.add_argument("--num_workers", type=int, default=2)
    opt.add_argument("--arch", choices=ARCH_CHOICES, default="eegnex")

    misc = ap.add_argument_group("misc & run settings")
    misc.add_argument("--patience", type=int, default=15)
    misc.add_argument("--min_delta", type=float, default=0.0)
    misc.add_argument("--save_path", type=str, default=None)
    misc.add_argument("--seed", type=int, default=-1)
    misc.add_argument("--deterministic", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save_path is None:
        args.save_path = f"{args.arch}.pt"

    if args.seed is None or args.seed < 0:
        args.seed = _auto_seed()

    _set_seeds(args.seed, args.deterministic)

    windows = build_ccd_windows(cache_root=Path(args.cache_root), mini=args.mini)
    train_set, valid_set, test_set = split_ccd_by_subject(windows, seed=args.seed)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=g,
    )

    model = make_model(args.arch, n_chans=129, n_times=200, sfreq=100).to(device)

    if args.loss == "huber":
        loss_fn = nn.HuberLoss(delta=args.huber_delta)
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - 1, 1)
    )
    _print_run_header(args, model, len(train_set), len(valid_set), len(test_set))

    best_nrmse, best_state = float("inf"), None
    best_epoch = 0
    patience, no_improve, min_delta = args.patience, 0, args.min_delta

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
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

        t_sec = time.perf_counter() - t0
        lr_now = float(scheduler.get_last_lr()[0])

        print(
            _epoch_line(
                epoch,
                args.epochs,
                lr_now,
                t_sec,
                tr_loss,
                tr_rmse,
                va_loss,
                va_rmse,
                va_nrmse,
                best_nrmse,
                best_epoch,
                no_improve,
                patience,
            )
        )

        if va_nrmse < best_nrmse - min_delta:
            best_nrmse = va_nrmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"Early stopping. Best nRMSE={best_nrmse:.4f} @ epoch {best_epoch}."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), Path(args.save_path))

    te_loss, te_rmse, te_nrmse = valid_model(
        test_loader, model, loss_fn, device, desc="Test", target_space=args.target_space
    )
    print(f"[Test] loss={te_loss:.4f}  rmse={te_rmse:.4f}  nrmse={te_nrmse:.4f}")


if __name__ == "__main__":
    main()
