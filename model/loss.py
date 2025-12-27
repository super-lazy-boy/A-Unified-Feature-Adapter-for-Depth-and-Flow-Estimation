# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """
    FlowSeek loss over a sequence of predictions.

    Notes for beginners:
    - flow_gt: [B,2,H,W]
    - valid:   valid mask for pixels. In practice it can be:
        [B,H,W] or [B,1,H,W] and values can be 0/1 or 0/255.
      So we normalize it to [B,H,W] and use >0 as valid.
    - KITTI "F1" is usually the OUTLIER rate (%), not classification F1-score.
    """
    n_predictions = len(output['flow'])
    flow_loss = 0.0

    # --- 1) Normalize valid mask shape to [B,H,W] ---
    # valid may come as [B,1,H,W] or [B,H,W]
    if valid.dim() == 4 and valid.size(1) == 1:
        valid = valid[:, 0]  # [B,H,W]
    elif valid.dim() == 2:
        # very rare edge case: [H,W] -> add batch dim
        valid = valid.unsqueeze(0)

    # --- 2) Normalize valid values ---
    # Many datasets use 0/1, some use 0/255. Using >0 is safer than >=0.5
    valid_mask = valid > 0

    # --- 3) Exclude extremely large motions (optional, standard practice) ---
    # mag: [B,H,W]
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    if max_flow is not None:
        valid_mask = valid_mask & (mag < max_flow)

    # --- 4) Accumulate loss from nf maps ---
    # output['nf'][i] is per-pixel loss map, shape [B,1,H,W] or [B,H,W]
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        loss_i = output['nf'][i]
        if loss_i.dim() == 3:
            loss_i = loss_i.unsqueeze(1)  # -> [B,1,H,W]

        # safe mask to avoid NaN/Inf pixels
        safe = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach()))

        # final_mask: [B,1,H,W]
        final_mask = safe & valid_mask[:, None]

        denom = final_mask.sum().clamp_min(1.0)  # avoid division by zero
        flow_loss = flow_loss + i_weight * ((final_mask * loss_i).sum() / denom)

    # --- 5) Metrics for logging (no grad) ---
    with torch.no_grad():
        flow_final = output['flow'][-1]  # [B,2,H,W]
        epe_map = torch.sum((flow_final - flow_gt) ** 2, dim=1).sqrt()  # [B,H,W]
        epe_valid = epe_map[valid_mask]

        # KITTI-style outlier rate (often printed as "F1")
        # outlier if EPE > 3 px AND relative error > 0.05
        # relative error uses gt magnitude; avoid /0 by clamp
        mag_safe = mag.clamp_min(1e-6)
        outlier = (epe_map > 3.0) & ((epe_map / mag_safe) > 0.05) & valid_mask

        num_valid = valid_mask.sum().item()
        f1 = (100.0 * outlier.sum().item() / max(1.0, num_valid))  # percentage

        metrics = {
            "epe": float(epe_valid.mean().item()) if epe_valid.numel() > 0 else 0.0,
            "f1": float(f1),  # outlier percentage in KITTI style
            "loss": float(flow_loss.item()),
            "valid_ratio": float(valid_mask.float().mean().item()),
            "num_valid": int(num_valid),
        }

    return flow_loss, metrics
