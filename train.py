import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, init_weights
from util.data_util import collate_fn_limit, collation_fn_voxelmean, collation_fn_voxelmean_tta, collation_fn_voxelmean_tta_custom, collate_fn_tempo, collation_fn_voxelmean_tempo

from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from util.nuscenes import nuScenes
from util.semantic_kitti import SemanticKITTI
from util.waymo import Waymo
from util.semantic_custom import SemanticCustom

from functools import partial
import spconv.pytorch as spconv
from tqdm import tqdm
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/semantic_kitti/semantic_kitti_ms_25.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def assign_seed(args):
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

def get_val_data_loader(args, logger):
    
    args.use_tta = getattr(args, "use_tta", False)
    if args.data_name == 'nuscenes':
        val_data = nuScenes(data_path=args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_val.pkl'], 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == 'semantic_kitti':
        val_data = SemanticKITTI(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='test' if args.valTest else 'val', 
            label_mapping=args.label_mapping, 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
            tempo_sample_num=args.tempo_sample_num
        )
    elif args.data_name == 'semantic_custom':
        val_data = SemanticCustom(data_path=args.data_root,
            label_mapping=args.label_mapping,
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
            tempo_sample_num=args.tempo_sample_num
        )
    elif args.data_name == 'waymo':
        val_data = Waymo(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.is_main_process:
        logger.info("val_data samples: '{}'".format(len(val_data)))

    if args.is_ddp_train:
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        val_sampler = None
        
    if getattr(args, "use_tta", False):
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean_tta if args.data_name != 'semantic_custom' else collation_fn_voxelmean_tta_custom
        )
    else:
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean if args.tempo_sample_num == 1 else collation_fn_voxelmean_tempo
        )

    return val_loader

def get_train_data_loader(args, logger):
    
    if args.data_name == 'nuscenes':
        train_data = nuScenes(args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_train.pkl'], 
            voxel_size=args.voxel_size, 
            split='train',
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    
    elif args.data_name == 'semantic_kitti':
        train_data = SemanticKITTI(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=args.label_mapping, 

            # rotate_aug=args.use_tta, 
            # flip_aug=args.use_tta, 
            # scale_aug=args.use_tta, 
            # transform_aug=args.use_tta, 

            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            transform_aug=True, 

            scale_params=[0.95,1.05], 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
            tempo_sample_num=args.tempo_sample_num
        )
    
    elif args.data_name == 'semantic_custom':
        train_data = SemanticCustom(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            # rotate_aug=True, 
            # flip_aug=True, 
            # scale_aug=True, 
            scale_params=[0.95,1.05], 
            # transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
            tempo_sample_num=args.tempo_sample_num
        )

    elif args.data_name == 'waymo':
        train_data = Waymo(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95, 1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.is_main_process:
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.is_ddp_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    collate_fn = partial(collate_fn_limit if args.tempo_sample_num == 1 else collate_fn_tempo, max_batch_points=args.max_batch_points, logger=logger if args.is_main_process else None)
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=args.batch_size, 
        # shuffle=False if args.open_tempo else (train_sampler is None), 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True, 
        collate_fn=collate_fn
    )

    return train_loader, train_sampler

def main_worker(args, logger, tb_log):
    best_iou = 0

    assign_seed(args)

    # get model
    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer import Semantic as Model
        
        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c, 
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
            open_tempo=args.open_tempo,
            mamba_layers=args.mamba_layers
        ).float()
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    
    if args.is_ddp_train:
        # args.batch_size = int(args.batch_size / args.world_size)
        # args.batch_size_val = int(args.batch_size_val / args.world_size)
        # args.workers = int((args.workers + args.world_size - 1) / args.world_size)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(args.device)
    # model.apply(init_weights)
    if args.is_ddp_train:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    if args.is_main_process:
        # logger.info("gpu list: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        # logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(model)
        # logger.info(model.state_dict().keys())
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))
        logger.info(f'batch_size: {args.batch_size}, batch_size_val: {args.batch_size_val}')

    # set loss func 
    class_weight = args.get("class_weight", None)
    if class_weight is not None:
        class_weight = torch.tensor(class_weight)
        class_weight = class_weight.to(args.device)
    if args.is_main_process:
        logger.info("class_weight: {}".format(class_weight))
        logger.info(f"loss_name: {args.loss_name}"
                    f" ignore_label: {args.ignore_label}")
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=args.ignore_label, reduction='none' if args.loss_name == 'focal_loss' else 'mean')
    criterion = criterion.to(args.device)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, eps=1e-4, weight_decay=args.weight_decay)

    if args.weight:
        if os.path.isfile(args.weight):
            if args.is_main_process:
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            if args.world_size == 1:
                checkpoint['state_dict'] = remove_module_prefix(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            
            if args.is_main_process:
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if args.is_main_process:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if args.is_main_process:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if args.is_main_process:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, train_sampler = get_train_data_loader(args, logger)
    val_loader = get_val_data_loader(args, logger)

    # set scheduler
    if args.scheduler == 'Poly':
        if args.is_main_process:
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.val:
        if args.use_tta:
            validate_tta(val_loader, model, criterion, args, logger, tb_log)
        else:
            validate(val_loader, model, criterion, args, logger, tb_log)
            # validate_distance(val_loader, model, criterion)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        if args.is_ddp_train:
            train_sampler.set_epoch(epoch)

        if args.is_main_process:
            logger.info("lr: {}".format(scheduler.get_last_lr()))

        model.train()
           
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, 
                                                                 epoch, scaler, scheduler, args, logger, tb_log)
        
        # if args.scheduler_update == 'epoch':
        #     scheduler.step()
        if args.is_ddp_train:
            dist.barrier()
        epoch_log = epoch + 1
        
        if args.is_main_process:
            tb_log.add_scalar('loss_train', loss_train, epoch_log)
            tb_log.add_scalar('mIoU_train', mIoU_train, epoch_log)
            tb_log.add_scalar('mAcc_train', mAcc_train, epoch_log)
            tb_log.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion, args, logger, tb_log)

            if args.is_main_process:
                tb_log.add_scalar('loss_val', loss_val, epoch_log)
                tb_log.add_scalar('mIoU_val', mIoU_val, epoch_log)
                tb_log.add_scalar('mAcc_val', mAcc_val, epoch_log)
                tb_log.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and args.is_main_process:
            model_save_path = args.save_path / 'model'
            if not model_save_path.exists():
                model_save_path.mkdir(parents=True, exist_ok=True)
            filename = model_save_path / ('checkpoint_' + str(epoch) + '.pth')
            # filename = model_save_path / ('checkpoint_%s.pth' % datetime.datetime.now().strftime('%Y%m%d-%H%M'))
            logger.info(f"Saving checkpoint to: {filename}")
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, str(args.save_path / "model/model_best.pth"))

    if args.is_main_process:
        tb_log.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def focal_loss(output, target, class_weight, ignore_label, gamma, need_softmax=True, eps=1e-8):
    mask = (target != ignore_label)
    output_valid = output[mask]
    if need_softmax:
        output_valid = F.softmax(output_valid, -1)
    target_valid = target[mask]
    p_t = output_valid[torch.arange(output_valid.shape[0], device=target_valid.device), target_valid] #[N, ]
    class_weight_per_sample = class_weight[target_valid]
    focal_weight_per_sample = (1.0 - p_t) ** gamma
    loss = -(class_weight_per_sample * focal_weight_per_sample * torch.log(p_t + eps)).sum() / (class_weight_per_sample.sum() + eps)
    return loss

def pre_process_data(tempo_data):
            
    for batch_data in tempo_data:
        for unit_data in batch_data:
            coord, xyz, feat = unit_data['coords'], unit_data['xyz'], unit_data['feats']

            batch = torch.Tensor([0]*xyz.shape[0]).long()

            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
        
            coord, feat = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True)
            s_feat = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, 1)
            unit_data['new_coords'] = coord
            unit_data['spconv_feat'] = s_feat

def check_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def check_layer_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            if grad_norm > 100:
                print(f"Layer {name} has large gradient: {grad_norm}")

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # remove 'module.'
        new_state_dict[k] = v
    return new_state_dict

def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler, args, logger, tb_log):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    loss_name = args.loss_name
    
    if args.is_main_process:
        progress_bar = tqdm(train_loader, desc=f"Rank {args.rank} Epoch {epoch+1}/{args.epochs}", position=args.rank, disable=(args.rank != 0))
    clip_counts = 0

    for i, batch_data in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)

        data_time.update(time.time() - end)
        
        if args.tempo_sample_num > 1:
            coord, xyz, feat, target, offset, tempo_feats = batch_data
        else:
            coord, xyz, feat, target, offset = batch_data
            
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)

        data_to_device = [coord, xyz, feat, target, offset, batch]
        data_to_device = [x.to(args.device) for x in data_to_device]
        coord, xyz, feat, target, offset, batch = data_to_device

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

        assert batch.shape[0] == feat.shape[0]
        if args.tempo_sample_num > 1:
            pre_process_data(tempo_data=tempo_feats)
        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(f'Trainable params: {Trainable_params/ 1e6}M')

            optimizer.zero_grad()

            if args.tempo_sample_num > 1:
                output = model(sinput, xyz, batch, tempo_feats)
            else:
                output = model(sinput, xyz, batch)
            assert output.shape[1] == args.classes

            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls

            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        if use_amp:
            scaler.scale(loss).backward()
            grad_norm_before_clip = check_gradient_norm(model)
            if epoch < 1 and grad_norm_before_clip > 100:  
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                # logger.warning(f"Clip gradient norm {grad_norm}! epoch: {epoch}, rank: {args.rank}")
                clip_counts += 1
            # grad_norm_after_clip = check_gradient_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.is_ddp_train:
            dist.barrier()
        # 检查各进程学习率是否一致
        # lr_list = torch.tensor([param_group['lr'] for param_group in optimizer.param_groups])
        # _, counts = torch.unique(lr_list, return_counts=True)
        # assert counts == 1
        # 检查损失Nan or Inf
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     logger.warning(f"Loss is NaN or Inf! epoch: {epoch}, rank: {args.rank}")
        # 检查梯度爆炸
        # grad_norm = check_gradient_norm(model)
        # if grad_norm > 100:
        #     logger.warning(f"Warning: Gradient norm {grad_norm} is too large! epoch: {epoch}, rank: {args.rank}")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             param_grad_norm = param.grad.norm(2).item()
        #             if param_grad_norm > 100:
        #                 logger.warning(f"{name}: Gradient L2 norm = {param_grad_norm:.4f}, rank: {args.rank}")
        # 消失
        # if grad_norm < 1e-5:
        #     logger.warning(f"Warning: Gradient norm {grad_norm} is too small! epoch: {epoch}, rank: {args.rank}")

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         logger.warning(f"⚠️ No gradient for {name}")

        if args.scheduler_update == 'step' and args.is_main_process:
            scheduler.step()
        if args.is_ddp_train:
            dist.barrier()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.is_ddp_train:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.is_ddp_train:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        # if (i + 1) % args.print_freq == 0 and args.is_main_process:
        #     lr = scheduler.get_last_lr()
        #     if isinstance(lr, list):
        #         lr = [round(x, 8) for x in lr]
        #     elif isinstance(lr, float):
        #         lr = round(lr, 8)
        #     logger.info('Epoch: [{}/{}][{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'Remain {remain_time} '
        #                 'Loss {loss_meter.val:.4f} '
        #                 'Lr: {lr} '
        #                 'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
        #                                                 batch_time=batch_time, data_time=data_time,
        #                                                 remain_time=remain_time,
        #                                                 loss_meter=loss_meter,
        #                                                 lr=lr,
        #                                                 accuracy=accuracy))
        if args.is_main_process:
            tb_log.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            tb_log.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            tb_log.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            tb_log.add_scalar('allAcc_train_batch', accuracy, current_iter)
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}", clip_counts=f"{clip_counts}", 
                                     grad_norm=f"{grad_norm_before_clip:.4f}", 
                                     remain_time=remain_time, lr=f"{scheduler.get_last_lr()}")
    if args.is_main_process:
        progress_bar.close()
            
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if args.is_main_process:
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion, args, logger, tb_log):
    if args.is_main_process:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    if (args.save_model_output):
        output_save_path_root = Path(args.save_path) / "test_results" / "sequences"
        if not output_save_path_root.exists():
            output_save_path_root.mkdir(parents=True, exist_ok=True)

    if args.is_main_process:
        progress_bar = tqdm(val_loader, desc=f"Rank {args.rank}", position=args.rank, disable=(args.rank != 0))

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        if args.tempo_sample_num > 1:
            (coord, xyz, feat, target, offset, inds_reconstruct, tempo_data) = batch_data
        else:
            (coord, xyz, feat, target, offset, inds_reconstruct, filename) = batch_data
        inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
 
        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        # coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        # batch = batch.cuda(non_blocking=True)

        data_to_device = [coord, xyz, feat, target, offset, batch]
        data_to_device = [x.to(args.device) for x in data_to_device]
        coord, xyz, feat, target, offset, batch = data_to_device

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        if args.tempo_sample_num > 1:
            pre_process_data(tempo_data)
        
        with torch.no_grad():
            if args.tempo_sample_num > 1:
                output = model(sinput, xyz, batch, tempo_data)
            else:
                output = model(sinput, xyz, batch)

            output = output[inds_reconstruct, :]
            
            if (args.save_model_output):
                origin_path = Path(filename[0])
                output_save_path = output_save_path_root / origin_path.parts[-3] / 'predictions'
                if not output_save_path.exists():
                    output_save_path.mkdir(parents=True, exist_ok=True)
                f_name = origin_path.with_suffix('.label').name
                output_save_path = output_save_path / f_name
                output_label = output.max(1)[1]
                label_numpy_value = output_label.cpu().numpy().astype(np.uint32)
                annotated_data = np.vectorize(val_loader.dataset.learning_map_inv.__getitem__, otypes=[np.uint32])(label_numpy_value)
                annotated_data.tofile(str(output_save_path))
                if args.is_main_process:
                    progress_bar.update(1)
                continue
        
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.is_ddp_train:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.is_ddp_train:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        # if (i + 1) % args.print_freq == 0 and args.is_main_process:
            # logger.info('Test: [{}/{}] '
            #             'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
            #             'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
            #             'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
            #             'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
            #                                               data_time=data_time,
            #                                               batch_time=batch_time,
            #                                               loss_meter=loss_meter,
            #                                               accuracy=accuracy))
        if args.is_main_process:
            progress_bar.update(1)
        
                
    if (args.save_model_output):
        import sys
        logger.info("saved model output, now exit!!!!!")
        sys.exit()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if args.is_main_process:
        progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}", mIoU=f"{mIoU:.4f}", mAcc=f"{mAcc:.4f}", allAcc=f"{allAcc:.4f}")
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        # logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        progress_bar.close()

    return loss_meter.avg, mIoU, mAcc, allAcc

def validate_tta(val_loader, model, criterion, args, logger, tb_log):
    if args.is_main_process:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation tta >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name
    
    if (args.save_model_output):
        output_save_path_root = Path(args.save_path) / "test_results" / "sequences"
        if not output_save_path_root.exists():
            output_save_path_root.mkdir(parents=True, exist_ok=True)

    if args.is_main_process:
        progress_bar = tqdm(val_loader, desc=f"Rank {args.rank}", position=args.rank, disable=(args.rank != 0))

    model.eval()
    end = time.time()
    for i, batch_data_list in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        with torch.no_grad():
            output = 0.0
            for batch_data in batch_data_list:

                if args.tempo_sample_num > 1:
                    (coord, xyz, feat, target, offset, inds_reconstruct, tempo_data) = batch_data
                else:
                    (coord, xyz, feat, target, offset, inds_reconstruct, filename) = batch_data
                inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)
 
                offset_ = offset.clone()
                offset_[1:] = offset_[1:] - offset_[:-1]
                batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

                coord = torch.cat([batch.unsqueeze(-1), coord], -1)
                spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
                # coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
                # batch = batch.cuda(non_blocking=True)

                data_to_device = [coord, xyz, feat, target, offset, batch]
                data_to_device = [x.to(args.device) for x in data_to_device]
                coord, xyz, feat, target, offset, batch = data_to_device

                sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

                assert batch.shape[0] == feat.shape[0]

                if args.tempo_sample_num > 1:
                    pre_process_data(tempo_data)

                if args.tempo_sample_num > 1:
                    output_i = model(sinput, xyz, batch, tempo_data)
                else:
                    output_i = model(sinput, xyz, batch)

                output_i = F.softmax(output_i[inds_reconstruct, :], -1)
                
                output = output + output_i
            output = output / len(batch_data_list)
            
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
            

        if (args.save_model_output):
            origin_path = Path(filename[0])
            output_save_path = output_save_path_root / origin_path.parts[-3] / 'predictions'
            if not output_save_path.exists():
                output_save_path.mkdir(parents=True, exist_ok=True)
            f_name = origin_path.with_suffix('.label').name
            output_save_path = output_save_path / f_name
            output_label = output
            label_numpy_value = output_label.cpu().numpy().astype(np.uint32)
            label_numpy_value = label_numpy_value + 1 # define by classes, ignore_label and label map set in semantic_kitti.py
            annotated_data = np.vectorize(val_loader.dataset.learning_map_inv.__getitem__, 
                                            otypes=[np.uint32])(label_numpy_value)
            annotated_data.tofile(str(output_save_path))
            if args.is_main_process:
                progress_bar.update(1)
            continue

        n = coord.size(0)
        if args.is_ddp_train:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.is_ddp_train:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        # if (i + 1) % args.print_freq == 0 and args.is_main_process:
        #     logger.info('Test: [{}/{}] '
        #                 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                 'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #                 'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
        #                 'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
        #                                                   data_time=data_time,
        #                                                   batch_time=batch_time,
        #                                                   loss_meter=loss_meter,
        #                                                   accuracy=accuracy))
        if args.is_main_process:
            progress_bar.update(1)
            
    if (args.save_model_output):
        import sys
        logger.info("saved model output, now exit!!!!!")
        sys.exit()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if args.is_main_process:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate_distance(val_loader, model, criterion, args, logger, tb_log):
    if args.is_main_process:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # For validation on points with different distance
    intersection_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    union_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    target_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        if args.tempo_sample_num > 1:
            (coord, xyz, feat, target, offset, inds_reverse, tempo_data) = batch_data
        else:
            (coord, xyz, feat, target, offset, inds_reverse) = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]        
        if args.tempo_sample_num > 1:
            pre_process_data(tempo_data)
        
        with torch.no_grad():
            if args.tempo_sample_num > 1:
                output = model(sinput, xyz, batch, tempo_data)
            else:
                output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
        
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.is_ddp_train:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        r = torch.sqrt(feat[:, 0] ** 2 + feat[:, 1] ** 2 + feat[:, 2] ** 2)
        r = r[inds_reverse]
        
        # For validation on points with different distance
        masks = [r <= 20, (r > 20) & (r <= 50), r > 50]

        for ii, mask in enumerate(masks):
            intersection, union, tgt = intersectionAndUnionGPU(output[mask], target[mask], args.classes, args.ignore_label)
            if args.is_ddp_train:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(tgt)
            intersection, union, tgt = intersection.cpu().numpy(), union.cpu().numpy(), tgt.cpu().numpy()
            intersection_meter_list[ii].update(intersection), union_meter_list[ii].update(union), target_meter_list[ii].update(tgt)

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.is_ddp_train:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and args.is_main_process:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    iou_class_list = [intersection_meter_list[i].sum / (union_meter_list[i].sum + 1e-10) for i in range(3)]
    accuracy_class_list = [intersection_meter_list[i].sum / (target_meter_list[i].sum + 1e-10) for i in range(3)]
    mIoU_list = [np.mean(iou_class_list[i]) for i in range(3)]
    mAcc_list = [np.mean(accuracy_class_list[i]) for i in range(3)]
    allAcc_list = [sum(intersection_meter_list[i].sum) / (sum(target_meter_list[i].sum) + 1e-10) for i in range(3)]

    if args.is_main_process:

        metrics = ['close', 'medium', 'distant']
        for ii in range(3):
            logger.info('Val result_{}: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(metrics[ii], mIoU_list[ii], mAcc_list[ii], allAcc_list[ii]))
            for i in range(args.classes):
                logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class_list[ii][i], accuracy_class_list[ii][i]))

        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc

def setup_device(args):
    args.device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(args.device)

def setup_ddp():
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    dist.destroy_process_group()

def create_logger(output_dir=None, rank=0, log_level=logging.INFO):
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H'))
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s [%(filename)s:%(lineno)d] %(message)s')

    # console = logging.StreamHandler()
    # console.setLevel(log_level if rank == 0 else 'ERROR')
    # console.setFormatter(formatter)
    # logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    class TqdmLoggingHandler(logging.StreamHandler):
        """让 logger 兼容 tqdm，防止进度条重绘"""
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)  # ✅ 终端日志用 tqdm.write() 兼容进度条
            except Exception:
                self.handleError(record)

    console = TqdmLoggingHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def init_log(args):
    args.save_path = Path(args.save_path)

    logger_path = args.save_path / 'log'
    if not logger_path.exists():
        logger_path.mkdir(parents=True, exist_ok=True)
    logger = create_logger(logger_path)

    tb_log_path = args.save_path / 'tb_log'
    if not tb_log_path.exists():
        tb_log_path.mkdir(parents=True, exist_ok=True)
    tb_log = SummaryWriter(tb_log_path) if args.is_main_process else None

    return logger, tb_log

def init_rank(args):
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.is_ddp_train = True if args.world_size > 1 else False
    args.is_main_process = True if args.rank == 0 else False

def main():
    if not torch.cuda.is_available():
        exit()

    args = get_parser()

    init_rank(args)

    logger, tb_log = init_log(args)

    if args.is_ddp_train:
        setup_ddp()

    setup_device(args)

    logger.warning(f"-----------------------------------------------------------")
    logger.warning(f"-----------------------start model!------------------------")
    logger.warning(f"-----------------------------------------------------------")
    logger.info(f"rank: {args.rank}, world_size: {args.world_size}, "
                f"local_rank: {args.local_rank}, device: {args.device}, "
                f"is ddp train: {args.is_ddp_train}")
    
    main_worker(args=args, logger=logger, tb_log=tb_log)
    
    if args.is_ddp_train:
        cleanup_ddp()

if __name__ == '__main__':
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main()
