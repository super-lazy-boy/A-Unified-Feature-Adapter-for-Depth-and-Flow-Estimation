# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from model.utils import frame_utils
from model.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

def _read_kitti_disparity_png(path: str) -> np.ndarray:
    """
    KITTI disparity PNG (disp_noc_0) is typically 16-bit PNG where disp = value / 256.
    Returns float32 disparity map with shape [H, W].
    """
    disp = frame_utils.read_gen(path)          # usually PIL.Image
    disp = np.array(disp).astype(np.float32)
    # common KITTI encoding:
    disp = disp / 256.0
    return disp

def _pad_to(t: torch.Tensor, H: int, W: int):
    # t: [C,H,W]
    _, h, w = t.shape
    pad_h = H - h
    pad_w = W - w
    if pad_h == 0 and pad_w == 0:
        return t
    # pad format: (left, right, top, bottom)
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))

def pad_collate_fn(batch):
    """
    batch is list of tuples:
    (img1, img2, flow, flow_valid, depth, depth_valid)
    Each tensor is [C,H,W] except flow_valid might be [H,W] in some implementations.
    We pad all spatial tensors to max(H,W) in this batch.
    """
    imgs1, imgs2, flows, fvalids, depths, dvalids = zip(*batch)

    # find max H,W over all tensors that have 3 dims
    H = 0
    W = 0
    for t in list(imgs1) + list(imgs2) + list(flows) + list(depths) + list(dvalids):
        if t is None:
            continue
        if t.dim() == 3:
            H = max(H, t.shape[1])
            W = max(W, t.shape[2])

    imgs1 = torch.stack([_pad_to(t, H, W) for t in imgs1], dim=0)
    imgs2 = torch.stack([_pad_to(t, H, W) for t in imgs2], dim=0)
    flows = torch.stack([_pad_to(t, H, W) for t in flows], dim=0)

    # flow_valid may be [H,W] or [1,H,W] depending on your dataset code
    fvalid_list = []
    for v in fvalids:
        if v.dim() == 2:
            v = v.unsqueeze(0)
        fvalid_list.append(_pad_to(v, H, W))
    fvalids = torch.stack(fvalid_list, dim=0)

    depths = torch.stack([_pad_to(t, H, W) for t in depths], dim=0)
    dvalids = torch.stack([_pad_to(t, H, W) for t in dvalids], dim=0)

    return imgs1, imgs2, flows, fvalids, depths, dvalids

def read_kitti_fB_from_cam_to_cam_txt(calib_txt_path: str):
    """
    从 KITTI 每帧 calib_cam_to_cam/*.txt 里解析 P2/P3 或 P_rect_02/P_rect_03
    返回 fx(像素焦距) 和 B(基线, 米)
    """
    P2 = None
    P3 = None

    with open(calib_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 兼容两种常见 key：
            # - raw 风格：P_rect_02 / P_rect_03
            # - benchmark 风格：P2 / P3 或 P2: / P3:
            if line.startswith("P_rect_02:") or line.startswith("P2:") or line.startswith("P2 "):
                vals = [float(x) for x in line.replace("P_rect_02:", "")
                                           .replace("P2:", "")
                                           .replace("P2", "")
                                           .split()]
                if len(vals) >= 12:
                    P2 = np.array(vals[:12], dtype=np.float64).reshape(3, 4)

            if line.startswith("P_rect_03:") or line.startswith("P3:") or line.startswith("P3 "):
                vals = [float(x) for x in line.replace("P_rect_03:", "")
                                           .replace("P3:", "")
                                           .replace("P3", "")
                                           .split()]
                if len(vals) >= 12:
                    P3 = np.array(vals[:12], dtype=np.float64).reshape(3, 4)

    if P2 is None or P3 is None:
        raise ValueError(f"Cannot parse P2/P3 (or P_rect_02/P_rect_03) from: {calib_txt_path}")

    fx = float(P2[0, 0])

    # KITTI 投影矩阵形式：P[0,3] = -fx * Tx
    Tx2 = -float(P2[0, 3]) / fx
    Tx3 = -float(P3[0, 3]) / fx
    B = abs(Tx3 - Tx2)

    return fx, B


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.depth_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        # worker seed init
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # ---------- flow ----------
        flow_valid = None
        if self.sparse:
            flow, flow_valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        # ---------- images ----------
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale -> 3ch
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # ---------- depth / disparity supervision ----------
        depth = None
        depth_valid = None
        if len(self.depth_list) > 0:
            dpath = self.depth_list[index]

            # KITTI disparity supervision (disp_noc_0) is 16-bit png: disp = val/256
            if isinstance(dpath, str) and (("disp_noc_0" in dpath) or ("disp_occ_0" in dpath)) and dpath.endswith(".png"):
                depth = _read_kitti_disparity_png(dpath)  # actually disparity
            else:
                depth_img = frame_utils.read_gen(dpath)
                depth = np.array(depth_img).astype(np.float32)

            depth_valid = (depth > 0).astype(np.float32)

        # augmentation (if enabled)
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, flow_valid = self.augmentor(img1, img2, flow, flow_valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # to torch
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # IMPORTANT: never return None for DataLoader default_collate
        # If depth is missing, return zeros + all-zero valid mask.
        if depth is None:
            H, W = img1.shape[1], img1.shape[2]
            depth = torch.zeros(1, H, W).float()
            depth_valid = torch.zeros(1, H, W).float()
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            depth_valid = torch.from_numpy(depth_valid).unsqueeze(0).float()

        if flow_valid is not None:
            flow_valid = torch.from_numpy(flow_valid).float()
        else:
            flow_valid = ((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float()

        # Return order MUST match train.py unpacking
        # img1, img2, flow, flow_valid, depth, depth_valid
        return img1, img2, flow, flow_valid, depth, depth_valid


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        if len(self.depth_list) > 0:
            self.depth_list = v * self.depth_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)

        self.root = osp.join(root, split)
        self.calib_dir = osp.join(self.root, "calib_cam_to_cam")

        # 1) 收集 image_2 对
        images1 = sorted(glob(osp.join(self.root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(self.root, 'image_2/*_11.png')))

        img_pairs = {}
        for img1, img2 in zip(images1, images2):
            frame_name = osp.basename(img1)          # 000000_10.png
            frame_id = frame_name.split('_')[0]      # 000000
            img_pairs[frame_id] = (img1, img2, frame_name)

        # 2) calib
        calib_map = {}
        calib_files = sorted(glob(osp.join(self.calib_dir, "*.txt")))
        for c in calib_files:
            fid = osp.splitext(osp.basename(c))[0]
            calib_map[fid] = c

        # 3) flow / disp（可能没有）
        flow_map = {}
        flow_files = sorted(glob(osp.join(self.root, 'flow_occ/*_10.png')))
        for f in flow_files:
            fname = osp.basename(f)          # 000000_10.png
            fid = fname.split('_')[0]
            flow_map[fid] = f

        disp_map = {}
        disp_files = sorted(glob(osp.join(self.root, 'disp_noc_0/*_10.png')))
        for d in disp_files:
            dname = osp.basename(d)
            fid = dname.split('_')[0]
            disp_map[fid] = d

        # 4) 决定是否为 test：必须同时具备 flow 和 disp 才能做“有监督验证”
        #    否则就是 test，只返回图像（不算指标）
        has_supervision = (len(flow_map) > 0) and (len(disp_map) > 0)

        if not has_supervision:
            # test 模式：只要图像+calib存在即可（calib用于你未来需要 depth 推理时做转换也可）
            keys = sorted(set(img_pairs.keys()) & set(calib_map.keys()))
            self.is_test = True
        else:
            # 有监督：取四者交集，保证严格对齐
            keys = sorted(set(img_pairs.keys()) & set(calib_map.keys()) & set(flow_map.keys()) & set(disp_map.keys()))
            self.is_test = False

        # 5) 构造列表（严格同序）
        self.image_list = []
        self.extra_info = []
        self.flow_list = []
        self.depth_list = []   # 这里先放 disp_noc_0 路径，父类会读出来；子类再转 depth
        self.calib_list = []

        for fid in keys:
            img1, img2, frame_name = img_pairs[fid]
            self.image_list.append([img1, img2])
            self.extra_info.append([frame_name])
            self.calib_list.append(calib_map[fid])

            if not self.is_test:
                self.flow_list.append(flow_map[fid])
                self.depth_list.append(disp_map[fid])

        # 最后做一次强校验，避免潜在错位
        if not self.is_test:
            assert len(self.image_list) == len(self.flow_list) == len(self.depth_list) == len(self.calib_list), \
                f"List length mismatch after alignment: images={len(self.image_list)}, flow={len(self.flow_list)}, " \
                f"disp={len(self.depth_list)}, calib={len(self.calib_list)}"

    def __getitem__(self, index):
        if self.is_test:
            # 直接复用父类 test 分支行为：只返回 (img1, img2, extra_info)
            img1, img2, extra = super(KITTI, self).__getitem__(index)
            return img1, img2, extra

        # 训练/验证：父类会读 flow + disp(raw) 并返回
        img1, img2, flow, flow_valid, disp_raw, disp_valid = super(KITTI, self).__getitem__(index)

        # disp_raw: [1,H,W]，其真实值通常需要 /256 还原到像素视差
        calib_path = self.calib_list[index]
        fx, B = read_kitti_fB_from_cam_to_cam_txt(calib_path)  # :contentReference[oaicite:5]{index=5}

        disp = disp_raw.clone().float()
        disp_max = float(disp.max().item())
        if disp_max > 512.0:
            disp = disp / 256.0  # convert to pixel disparity

        # mask：disp > 0
        valid = disp > 0.0

        # depth = fx * B / disp
        depth = (fx * B) / (disp + 1e-6)

        # KITTI 常用最大深度 80m（或 100m），避免数值爆炸影响训练
        max_depth = 80.0
        valid = valid & (depth > 0.1) & (depth < max_depth)

        depth_valid = valid.float()
        depth = depth.clamp(min=0.0, max=max_depth)

        return img1, img2, flow, flow_valid.float(), depth, depth_valid


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.dataset == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training', root=args.paths['kitti'])

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True, collate_fn=pad_collate_fn)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

