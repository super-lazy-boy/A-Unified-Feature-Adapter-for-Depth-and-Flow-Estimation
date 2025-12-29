# evaluate.py (MODIFIED)
# - Logs evaluation metrics to CSV in the same schema as train.py
# - Prints per-sample metrics similar to train.py validate_one_epoch()
# - CSV path is fixed as: csv_path = f"{args.save_dir}/eval_metrics.csv"

from __future__ import annotations

import os
import csv
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import model.datasets as datasets_module
from model.flowseek import FlowSeek
from model.loss import sequence_loss  # used in train.py for flow metrics


# -----------------------------
# Utils: robust state_dict load
# -----------------------------
def _strip_module_prefix(state_dict):
    """If checkpoint was saved from DataParallel, keys start with 'module.'; strip it."""
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = _strip_module_prefix(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if len(missing) > 0:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")


# -----------------------------
# CSV logging helpers
# -----------------------------
def _csv_append_row(csv_path: str, fieldnames: list[str], row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# Depth losses/metrics (copied from train.py logic)
# -----------------------------
def depth_l1_loss(depth_pred, depth_gt, depth_valid, eps: float = 1e-6):
    """
    depth_pred: [B,1,H,W]
    depth_gt:   [B,1,H,W]
    depth_valid:[B,1,H,W] or [B,H,W]
    """
    if depth_gt is None or depth_valid is None or depth_pred is None:
        return torch.tensor(0.0, device=depth_pred.device if depth_pred is not None else "cpu")

    if depth_valid.dim() == 3:
        depth_valid = depth_valid.unsqueeze(1)

    mask = depth_valid > 0.5
    denom = mask.sum().clamp_min(1.0)
    if denom.item() <= 1.0:
        return depth_pred.new_tensor(0.0)

    if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
        depth_pred = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

    return (mask * (depth_pred - depth_gt).abs()).sum() / denom


@torch.no_grad()
def depth_metrics(depth_pred, depth_gt, depth_valid, eps: float = 1e-6):
    """
    Returns dict with keys: depth_mae, depth_rmse, depth_abs_rel
    """
    if depth_gt is None or depth_valid is None or depth_pred is None:
        return {"depth_mae": 0.0, "depth_rmse": 0.0, "depth_abs_rel": 0.0}

    if depth_valid.dim() == 3:
        depth_valid = depth_valid.unsqueeze(1)

    if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
        depth_pred = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

    mask = depth_valid > 0.5
    denom = mask.sum().clamp_min(1.0)
    if denom.item() <= 1.0:
        return {"depth_mae": 0.0, "depth_rmse": 0.0, "depth_abs_rel": 0.0}

    diff = (depth_pred - depth_gt)
    mae = (mask * diff.abs()).sum() / denom
    rmse = torch.sqrt((mask * diff.pow(2)).sum() / denom)
    abs_rel = (mask * (diff.abs() / (depth_gt.abs() + eps))).sum() / denom

    return {
        "depth_mae": float(mae.item()),
        "depth_rmse": float(rmse.item()),
        "depth_abs_rel": float(abs_rel.item()),
    }


# -----------------------------
# Visualization helpers (kept from your evaluate.py)
# -----------------------------
def make_colorwheel():
    # RY, YG, GC, CB, BM, MR
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)

    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY).astype(np.uint8)
    col += RY

    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG).astype(np.uint8)
    colorwheel[col:col + YG, 1] = 255
    col += YG

    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(GC) / GC).astype(np.uint8)
    col += GC

    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB).astype(np.uint8)
    colorwheel[col:col + CB, 2] = 255
    col += CB

    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(BM) / BM).astype(np.uint8)
    col += BM

    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR).astype(np.uint8)
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_image(flow_uv, clip_flow=None):
    flow = flow_uv.copy()
    if clip_flow is not None:
        flow = np.clip(flow, -clip_flow, clip_flow)

    u, v = flow[..., 0], flow[..., 1]
    rad = np.sqrt(u * u + v * v)
    rad_max = np.max(rad) + 1e-5

    u = u / rad_max
    v = v / rad_max

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    for ci in range(3):
        col0 = colorwheel[k0, ci] / 255.0
        col1 = colorwheel[k1, ci] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - rad / (rad_max) * (1 - col)
        img[..., ci] = np.floor(255 * col).astype(np.uint8)

    return img


def depth_to_colormap(depth, valid_mask=None, dmin=None, dmax=None):
    import matplotlib.cm as cm
    dep = depth.copy()
    if valid_mask is None:
        valid_mask = np.isfinite(dep) & (dep > 0)

    if dmin is None:
        dmin = np.percentile(dep[valid_mask], 1) if np.any(valid_mask) else 0.0
    if dmax is None:
        dmax = np.percentile(dep[valid_mask], 99) if np.any(valid_mask) else 1.0
    dmax = max(dmax, dmin + 1e-6)

    dep = np.clip(dep, dmin, dmax)
    dep_norm = (dep - dmin) / (dmax - dmin + 1e-6)

    cmap = cm.get_cmap("magma")
    colored = cmap(dep_norm)[:, :, :3]
    colored[~valid_mask] = 0.0
    return (colored * 255.0).astype(np.uint8)


def tensor_image_to_uint8(img_t):
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def save_png(path, arr_uint8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_uint8).save(path)


def concat_horiz(img_list):
    H = img_list[0].shape[0]
    outs = []
    for im in img_list:
        if im.shape[0] != H:
            w = int(im.shape[1] * (H / im.shape[0]))
            im = np.array(Image.fromarray(im).resize((w, H), resample=Image.BILINEAR))
        outs.append(im)
    return np.concatenate(outs, axis=1)


# -----------------------------
# Build args (aligned with train.py)
# -----------------------------
def build_args():
    base = os.path.dirname(__file__)
    args = SimpleNamespace(
        # paths
        kitti_root=os.path.join(base, "data", "KITTI_split"),
        split="testing",  # set to "training" if you want GT metrics
        ckpt_path=os.path.join(base, "train_checkpoints", "deeplearning_depth.pth"),

        # inference
        batch_size=1,
        num_workers=2,
        gpus=[0],
        mixed_precision=True,
        iters=4,
        save_dir=os.path.join(base, "result_test", "deeplearning_depth"),

        # FlowSeek config (must match training)
        pretrain="resnet34",
        initial_dim=64,
        block_dims=[64, 128, 256, 512],
        feat_type="resnet",

        radius=4,
        dim=128,
        num_blocks=2,

        # flow uncertainty branch configs
        use_var=True,
        var_min=0,
        var_max=10,

        # DepthAnythingV2 backbone
        da_size="vitb",

        dataset="kitti",
        stage="test",
    )
    return args


def build_kitti_loader(args):
    ds = datasets_module.KITTI(split=args.split, root=args.kitti_root)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[INFO] KITTI {args.split} samples: {len(ds)}")
    return loader


# -----------------------------
# Main inference + metrics logging
# -----------------------------
@torch.no_grad()
def run_kitti_inference_and_log(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # REQUIRED by user:
    csv_path = f"{args.save_dir}/eval_metrics.csv"

    loader = build_kitti_loader(args)

    model = FlowSeek(args).to(device)
    model.eval()
    load_checkpoint(model, args.ckpt_path, device)

    # output dirs for visuals
    out_rgb = os.path.join(args.save_dir, "rgb")
    out_flow = os.path.join(args.save_dir, "flow")
    out_depth = os.path.join(args.save_dir, "depth")
    out_triplet = os.path.join(args.save_dir, "triplet")
    for d in [out_rgb, out_flow, out_depth, out_triplet]:
        os.makedirs(d, exist_ok=True)

    # CSV schema aligned to train.py metric names
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    fieldnames = [
        "run_id", "split", "iter", "frame_name",
        "flow_loss", "epe", "f1",
        "depth_loss", "depth_mae", "depth_rmse", "depth_abs_rel",
        "note"
    ]

    # accumulators for summary (only when GT exists)
    sums = {k: 0.0 for k in ["flow_loss", "epe", "f1", "depth_loss", "depth_mae", "depth_rmse", "depth_abs_rel"]}
    n_metric = 0

    for i, batch in enumerate(loader):
        # Two possible batch formats (as in train.py validate_one_epoch):
        #  - test: (img1, img2, extra_info)
        #  - supervised: (img1,img2,flow,flow_valid,depth,depth_valid)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            image1, image2, extra = batch
            frame_name = extra[0][0] if isinstance(extra, (list, tuple)) else f"{i:06d}_10.png"
            flow = flow_valid = depth = depth_valid = None
        else:
            image1, image2, flow, flow_valid, depth, depth_valid = batch
            frame_name = f"{i:06d}_10.png"

        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        if flow is not None:
            flow = flow.to(device, non_blocking=True)
            flow_valid = flow_valid.to(device, non_blocking=True)
        if depth is not None:
            depth = depth.to(device, non_blocking=True)
        if depth_valid is not None:
            depth_valid = depth_valid.to(device, non_blocking=True)

        # forward (if GT exists, pass flow_gt and test_mode=False like train.py)
        if flow is not None:
            out = model(image1, image2, iters=args.iters, flow_gt=flow, test_mode=False)
        else:
            out = model(image1, image2, iters=args.iters, flow_gt=None, test_mode=True)

        # predictions for visualization
        flow_pred = out["final"][0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        flow_img = flow_to_image(flow_pred)

        depth_pred = out.get("depth", None)
        if depth_pred is not None:
            dep = depth_pred[0, 0].detach().cpu().numpy().astype(np.float32)
            dep_img = depth_to_colormap(dep, valid_mask=np.isfinite(dep) & (dep > 0))
        else:
            dep_img = np.zeros((flow_img.shape[0], flow_img.shape[1], 3), dtype=np.uint8)

        rgb_img = tensor_image_to_uint8(image1[0])

        stem = os.path.splitext(frame_name)[0]
        save_png(os.path.join(out_rgb, f"{stem}_rgb.png"), rgb_img)
        save_png(os.path.join(out_flow, f"{stem}_flow.png"), flow_img)
        save_png(os.path.join(out_depth, f"{stem}_depth.png"), dep_img)
        save_png(os.path.join(out_triplet, f"{stem}_triplet.png"), concat_horiz([rgb_img, flow_img, dep_img]))

        # metrics (ONLY if GT available)
        note = ""
        if flow is not None and flow_valid is not None:
            flow_loss_t, metrics = sequence_loss(out, flow, flow_valid, gamma=0.85)  # train.py uses gamma=0.85
            dloss_t = depth_l1_loss(out.get("depth", None), depth, depth_valid)
            dmet = depth_metrics(out.get("depth", None), depth, depth_valid)

            flow_loss = float(flow_loss_t.item())
            epe = float(metrics.get("epe", 0.0))
            f1 = float(metrics.get("f1", 0.0))

            depth_loss = float(dloss_t.item())
            depth_mae = float(dmet["depth_mae"])
            depth_rmse = float(dmet["depth_rmse"])
            depth_abs_rel = float(dmet["depth_abs_rel"])

            print(
                f"[eval] {i+1:04d}/{len(loader):04d} "
                f"flow_loss={flow_loss:.4f} epe={epe:.3f} f1={f1:.2f} "
                f"d_loss={depth_loss:.4f} d_mae={depth_mae:.3f} d_rmse={depth_rmse:.3f} d_abs_rel={depth_abs_rel:.3f} "
                f"name={frame_name}"
            )

            sums["flow_loss"] += flow_loss
            sums["epe"] += epe
            sums["f1"] += f1
            sums["depth_loss"] += depth_loss
            sums["depth_mae"] += depth_mae
            sums["depth_rmse"] += depth_rmse
            sums["depth_abs_rel"] += depth_abs_rel
            n_metric += 1
        else:
            flow_loss = epe = f1 = ""
            depth_loss = depth_mae = depth_rmse = depth_abs_rel = ""
            note = "no_gt(split=testing). Use split=training to compute metrics like train.py."
            if (i + 1) % 5 == 0 or (i + 1) == len(loader):
                print(f"[INFO] Processed {i+1}/{len(loader)} (no GT metrics)")

        row = {
            "run_id": run_id,
            "split": args.split,
            "iter": i,
            "frame_name": frame_name,
            "flow_loss": flow_loss,
            "epe": epe,
            "f1": f1,
            "depth_loss": depth_loss,
            "depth_mae": depth_mae,
            "depth_rmse": depth_rmse,
            "depth_abs_rel": depth_abs_rel,
            "note": note,
        }
        _csv_append_row(csv_path, fieldnames, row)

    if n_metric > 0:
        summary = {
            "run_id": run_id,
            "split": args.split,
            "iter": "summary",
            "frame_name": "",
            "flow_loss": sums["flow_loss"] / n_metric,
            "epe": sums["epe"] / n_metric,
            "f1": sums["f1"] / n_metric,
            "depth_loss": sums["depth_loss"] / n_metric,
            "depth_mae": sums["depth_mae"] / n_metric,
            "depth_rmse": sums["depth_rmse"] / n_metric,
            "depth_abs_rel": sums["depth_abs_rel"] / n_metric,
            "note": f"average over {n_metric} samples",
        }
        _csv_append_row(csv_path, fieldnames, summary)
        print(
            f"[eval-summary] flow_loss={summary['flow_loss']:.4f} epe={summary['epe']:.3f} f1={summary['f1']:.2f} | "
            f"d_loss={summary['depth_loss']:.4f} d_mae={summary['depth_mae']:.3f} "
            f"d_rmse={summary['depth_rmse']:.3f} d_abs_rel={summary['depth_abs_rel']:.3f}"
        )

    print(f"[eval] CSV saved to: {csv_path}")


if __name__ == "__main__":
    args = build_args()
    run_kitti_inference_and_log(args)
