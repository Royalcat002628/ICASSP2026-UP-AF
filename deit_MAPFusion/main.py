# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.serialization
from PIL import ImageFile
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from torch.utils.data import WeightedRandomSampler

import utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from models import DistilledVisionTransformer

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from engine import evaluate_test


# 忽略警告信息，避免训练时输出过多无关提示
warnings.filterwarnings('ignore')

# 允许 argparse.Namespace 类型在 torch.load 反序列化时安全加载
torch.serialization.add_safe_globals([argparse.Namespace])

# 允许加载损坏但仍可读取的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args_parser():
    """
    构建命令行参数解析器。

    作用：
        定义训练、验证、测试、模型、优化器、数据集等所有运行参数。

    返回：
        parser: argparse.ArgumentParser 对象
    """
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    # ============================================================
    # 一、基础训练参数
    # ============================================================
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=1000, type=int)  # 原 300，修改为 1000

    # ============================================================
    # 二、测试相关参数
    # ============================================================
    parser.add_argument('--test', action='store_true',
                        help='使用最佳模型做测试集评估')
    parser.add_argument('--best_ckpt', type=str, default='',
                        help='测试时指定最佳 checkpoint 路径')
    parser.add_argument('--result_dir', type=str, default='test_results',
                        help='测试结果输出目录，如混淆矩阵和评估报告保存位置')

    # ============================================================
    # 三、模型参数
    # ============================================================
    parser.add_argument('--model', default='deit_small_distilled_patch16_224',
                        type=str, metavar='MODEL',
                        help='训练模型名称')
    parser.add_argument('--input-size', default=224, type=int,
                        help='输入图像尺寸')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout 比例')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop Path 比例')

    # EMA（Exponential Moving Average）相关参数
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # ============================================================
    # 四、优化器参数
    # ============================================================
    parser.add_argument('--opt', default='SGD', type=str, metavar='OPTIMIZER',
                        help='优化器类型（原默认 adamw，这里改为 SGD）')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='优化器 eps 参数')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='优化器 betas 参数')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='梯度裁剪阈值')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD 的 momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减（原 0.05，这里改为 1e-4）')

    # ============================================================
    # 五、学习率调度参数
    # ============================================================
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='学习率调度器类型')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='初始学习率')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='学习率噪声区间')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='学习率噪声比例')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='学习率噪声标准差')
    parser.add_argument('--warmup-lr', type=float, default=5e-7, metavar='LR',
                        help='warmup 初始学习率')
    parser.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                        help='最小学习率')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='学习率衰减间隔 epoch')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='warmup 轮数')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='cooldown 轮数')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='Plateau 调度器耐心轮数')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='学习率衰减率')

    # ============================================================
    # 六、数据增强参数
    # ============================================================
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT',
                        help='颜色扰动强度')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='AutoAugment 策略')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label Smoothing 系数')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='训练图像插值方式')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random Erase 概率')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random Erase 模式')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random Erase 次数')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='是否不对第一份增强结果做 random erase')

    # Mixup / CutMix
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix 最小/最大比例')
    parser.add_argument('--mixup-prob', type=float, default=0,
                        help='启用 mixup/cutmix 的概率')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='mixup 和 cutmix 切换概率')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='mixup/cutmix 应用方式')

    # ============================================================
    # 七、蒸馏参数
    # ============================================================
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='教师模型名称')
    parser.add_argument('--teacher-path', type=str, default='',
                        help='教师模型权重路径')
    parser.add_argument('--distillation-type', default='none',
                        choices=['none', 'soft', 'hard'], type=str, help='蒸馏类型')
    parser.add_argument('--distillation-alpha', default=0.5, type=float,
                        help='蒸馏损失权重')
    parser.add_argument('--distillation-tau', default=1.0, type=float,
                        help='蒸馏温度系数')

    # ============================================================
    # 八、微调参数
    # ============================================================
    parser.add_argument('--finetune', default='',
                        help='从某个 checkpoint 微调')

    # ============================================================
    # 九、数据集参数
    # ============================================================
    parser.add_argument('--data-path', default='../datasets/Ucity/images/', type=str,
                        help='数据集路径')
    parser.add_argument('--data-set', default='IMNETSUBSET',
                        choices=[
                            'CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19',
                            'CIFAR10SUBSET', 'CIFAR100SUBSET', 'IMNETSUBSET'
                        ],
                        type=str, help='数据集类型')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order',
                                 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='INAT 语义层级')

    # ============================================================
    # 十、运行与分布式参数
    # ============================================================
    parser.add_argument('--output_dir', default='',
                        help='输出目录，为空则不保存')
    parser.add_argument('--device', default='cuda',
                        help='训练/测试设备')
    parser.add_argument('--seed', default=0, type=int,
                        help='随机种子')
    parser.add_argument('--finetune', default='',
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='起始 epoch')
    parser.add_argument('--eval', action='store_true',
                        help='只进行验证，不训练')
    parser.add_argument('--dist-eval', action='store_true', default=False,
                        help='是否启用分布式验证')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='DataLoader 的 worker 数')
    parser.add_argument('--pin-mem', action='store_true',
                        help='是否启用 pin memory')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='分布式进程数')
    parser.add_argument('--dist_url', default='env://',
                        help='分布式初始化地址')

    parser.add_argument('--subset_ids', default=None, type=str,
                        help='子集样本 id 的 json 文件')
    parser.add_argument('--eval_interval', default=1, type=int,
                        help='每隔多少个 epoch 做一次验证')

    return parser


def main(args):
    """
    主函数。

    整体流程：
        1. 初始化分布式与随机种子
        2. 构建训练集、验证集与采样器
        3. 构建模型、优化器、学习率调度器、损失函数
        4. 支持 finetune / resume / eval / test 等模式
        5. 进入训练循环
        6. 定期验证、保存 best checkpoint、记录日志并绘图
    """
    # ============================================================
    # 一、初始化运行环境
    # ============================================================
    utils.init_distributed_mode(args)
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # 固定随机种子，保证实验可复现
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ============================================================
    # 二、构建数据集
    # ============================================================
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # ============================================================
    # 三、构建采样器
    # ============================================================
    # 这里实际固定进入这个分支，相当于始终启用下面的采样逻辑
    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        # -----------------------------
        # 训练集：WeightedRandomSampler 平衡采样
        # -----------------------------
        # 如果数据集有 targets 属性，直接取标签；否则遍历数据集获得标签
        if hasattr(dataset_train, 'targets'):
            targets = dataset_train.targets
        else:
            targets = [label for (_, label) in dataset_train]

        # 根据类别频次构造反比权重，类别越少权重越高
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[t] for t in targets]

        total_samples = len(sample_weights)
        bs = args.batch_size

        # 为了和 batch 对齐，向下取整到 batch 的整数倍
        aligned_samples = (total_samples // bs) * bs
        if aligned_samples == 0:
            # 若样本数还不够一个 batch，则至少采一个 batch
            aligned_samples = bs

        sampler_train = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=aligned_samples,
            replacement=True
        )

        # -----------------------------
        # 验证集采样器
        # -----------------------------
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # ============================================================
    # 四、构建 DataLoader
    # ============================================================
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(3 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # ============================================================
    # 五、构建 Mixup
    # ============================================================
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes
        )

    # ============================================================
    # 六、创建模型
    # ============================================================
    print(f"Creating model: {args.model}")

    # 这里直接实例化自定义的 DistilledVisionTransformer
    model = DistilledVisionTransformer(
        img_size=[args.input_size],
        patch_size=16,
        in_chans=3,
        num_classes=args.nb_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=args.drop,
        attn_drop_rate=args.drop_path,
        drop_path_rate=args.drop_path,
        norm_layer=torch.nn.LayerNorm,
    )

    # ============================================================
    # 七、加载 finetune 权重
    # ============================================================
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()

        # 删除分类头相关参数，避免类别数不匹配
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model:
                print(f"⚠️ 删除预训练 checkpoint 中的 {k}")
                checkpoint_model.pop(k)

        # -----------------------------
        # 位置编码插值
        # -----------------------------
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    # ============================================================
    # 八、创建 EMA 模型
    # ============================================================
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    # ============================================================
    # 九、分布式封装
    # ============================================================
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # ============================================================
    # 十、创建优化器、AMP、学习率调度器
    # ============================================================
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ============================================================
    # 十一、构建基础损失函数
    # ============================================================
    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # mixup 模式下标签已是软标签，使用 SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ============================================================
    # 十二、构建教师模型（若启用蒸馏）
    # ============================================================
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")

        teacher_model = DistilledVisionTransformer(
            img_size=[args.input_size],
            patch_size=16,
            in_chans=3,
            num_classes=args.nb_classes,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=args.drop,
            attn_drop_rate=args.drop_path,
            drop_path_rate=args.drop_path,
            norm_layer=torch.nn.LayerNorm
        )

        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')

        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # 用 DistillationLoss 包裹基础损失
    criterion = DistillationLoss(
        criterion, teacher_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau
    )

    output_dir = Path(args.output_dir)

    # ============================================================
    # 十四、只验证模式
    # ============================================================
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    # ============================================================
    # 十五、测试模式
    # ============================================================
    if args.test:
        print("Performing test evaluation with the best model ...")

        if args.best_ckpt:
            checkpoint_path = Path(args.best_ckpt)
        else:
            checkpoint_path = output_dir / "best_checkpoint.pth"

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            print("Best checkpoint not found at:", checkpoint_path)
            return

        dataset_test, _ = build_dataset(is_train=False, args=args)
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            sampler=torch.utils.data.SequentialSampler(dataset_test),
            batch_size=int(3 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        result_dir = args.result_dir if args.result_dir else "test_results"
        test_metrics = evaluate_test(test_loader, model, device, result_dir)

        print("Test Evaluation Metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
        return

    # ============================================================
    # 十六、训练前初始化记录变量
    # ============================================================
    epochs_list = []
    train_losses = []
    test_acc1s = []
    test_acc5s = []

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    # ============================================================
    # 十七、主训练循环
    # ============================================================
    for epoch in tqdm(range(args.start_epoch, args.epochs + 1),
                      desc="Training Progress", ncols=100):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # -----------------------------
        # 1）训练一个 epoch
        # -----------------------------
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == ''
        )

        # -----------------------------
        # 2）更新学习率调度器
        # -----------------------------
        lr_scheduler.step(epoch)

        # -----------------------------
        # 3）保存最新 checkpoint
        # -----------------------------
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # -----------------------------
        # 4）按间隔做验证
        # -----------------------------
        if epoch % args.eval_interval == 0:
            test_stats = evaluate(data_loader_val, model, device)

            current_acc1 = test_stats["acc1"]
            current_acc5 = test_stats["acc5"]
            current_loss = train_stats.get("loss", None)

            print(f"Epoch {epoch}: Test Acc1 = {current_acc1:.2f}%, Test Acc5 = {current_acc5:.2f}%")

            epochs_list.append(epoch)
            if current_loss is not None:
                train_losses.append(current_loss)
            test_acc1s.append(current_acc1)
            test_acc5s.append(current_acc5)

            # 若当前 Top-1 最优，则保存 best checkpoint
            if max_accuracy < current_acc1:
                max_accuracy = current_acc1
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                        }, checkpoint_path)

            print(f'Max accuracy: {max_accuracy:.2f}%')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        else:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

        # -----------------------------
        # 5）写日志
        # -----------------------------
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # -----------------------------
        # 6）打印总训练时间
        # -----------------------------
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # ========================================================
        # 十八、绘制训练/验证曲线
        # ========================================================
        sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2)
        font = {'family': 'sans-serif', 'size': 12}
        plt.rc('font', **font)

        # -----------------------------
        # 1）训练损失曲线
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax.plot(
            epochs_list, train_losses,
            linewidth=2, marker='o', markersize=4,
            label='Train Loss'
        )
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training Loss Over All Epochs', fontsize=16)
        ax.grid(which='major', linestyle='--', alpha=0.6)
        ax.legend(fontsize=12, loc='upper right')

        for x, y in zip(epochs_list, train_losses):
            ax.text(x, y + max(train_losses) * 0.02, f'{y:.2f}',
                    ha='center', va='bottom', fontsize=8, alpha=0.8)

        fig.tight_layout()
        full_loss_path = os.path.join(args.output_dir, "train_loss_full_refined.png")
        fig.savefig(full_loss_path)
        plt.close(fig)

        # -----------------------------
        # 2）验证集准确率曲线
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax.plot(
            epochs_list, test_acc1s,
            linewidth=2, marker='s', markersize=4,
            label='Top-1 Acc'
        )
        ax.plot(
            epochs_list, test_acc5s,
            linewidth=2, marker='^', markersize=4,
            label='Top-5 Acc'
        )

        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('Test Accuracy Over All Epochs', fontsize=16)
        ax.set_xlim(epochs_list[0], epochs_list[-1])
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.grid(which='major', linestyle='--', alpha=0.6)
        ax.legend(fontsize=12, loc='lower right')

        # 标注 Top-1 最优点
        max_acc1 = max(test_acc1s)
        max_idx1 = test_acc1s.index(max_acc1)
        epoch_max1 = epochs_list[max_idx1]
        ax.text(epoch_max1, max_acc1 + 1,
                f'{max_acc1:.1f}%',
                ha='center', va='bottom', fontsize=10, alpha=0.8)

        # 标注 Top-5 最优点
        max_acc5 = max(test_acc5s)
        max_idx5 = test_acc5s.index(max_acc5)
        epoch_max5 = epochs_list[max_idx5]
        ax.text(epoch_max5, max_acc5 + 1,
                f'{max_acc5:.1f}%',
                ha='center', va='bottom', fontsize=10, alpha=0.8)

        fig.tight_layout()
        full_acc_path = os.path.join(args.output_dir, "test_accuracy_full_refined.png")
        fig.savefig(full_acc_path)
        plt.close(fig)

        print(f"Saved refined plots to\n  {full_loss_path}\n  {full_acc_path}")


if __name__ == '__main__':
    """
    程序入口。

    作用：
        1. 创建总参数解析器
        2. 解析命令行参数
        3. 若设置了输出目录，则自动创建
        4. 调用 main(args) 启动训练 / 验证 / 测试流程
    """
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
