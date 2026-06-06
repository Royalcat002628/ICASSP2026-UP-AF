# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import sys

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def extract_features(args):
    """
    主特征提取函数。

    功能概述：
        1. 根据命令行参数构建模型
        2. 加载预训练权重
        3. 构建数据集与 DataLoader
        4. 调用 validate_network 提取全部样本的特征
        5. 将特征与标签拼接后保存为 .npy 文件

    输入：
        args: 命令行参数解析结果，包含模型结构、数据集名称、权重路径、
              batch size、输出目录等信息

    输出：
        无直接返回值。
        最终会在 args.output_dir 下保存一个 .npy 文件，
        文件内容为 [features, labels] 的拼接结果。
    """
    # utils.init_distributed_mode(args)
    # 打印当前所有参数，方便检查实验配置
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # 开启 cudnn benchmark，通常在输入尺寸固定时可提升卷积性能
    cudnn.benchmark = True

    # ============================================================
    # 一、构建网络
    # ============================================================
    # 如果是 Vision Transformer（如 vit_tiny / vit_small / vit_base）
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    # 如果是 XCiT 结构，则从 torch.hub 加载
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)

    # 否则检查是否为 torchvision 中已有的模型
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()

        # 这里对 torchvision 模型做了一些适配修改：
        # 1. conv1 改为 3x3，适合较小图像输入
        # 2. 去掉 maxpool
        # 3. 去掉最后分类层 fc，仅保留特征输出
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    model.cuda()
    model.eval()

    # ============================================================
    # 二、加载预训练权重
    # ============================================================
    # 若路径中含 dino 或权重路径为空，则使用项目内部封装的加载逻辑
    if 'dino' in args.pretrained_weights or args.pretrained_weights == "":
        utils.load_pretrained_weights(
            model,
            args.pretrained_weights,
            args.checkpoint_key,
            args.arch,
            args.patch_size
        )
    else:
        # 手动加载预训练权重文件
        ckpt_dict = torch.load(args.pretrained_weights, map_location="cpu")

        # ------------------------------------------------------------
        # 处理位置编码 pos_embed
        # ------------------------------------------------------------
        # 一些 ViT 权重可能带有 distillation token，导致 pos_embed 长度不一致。
        # 这里针对 [1,198,*] -> [1,197,*] 做了兼容处理：
        # 若预训练权重多一个位置编码，则裁掉最后一个。
        pretrained_pos_embed = ckpt_dict.get('pos_embed')

        if pretrained_pos_embed is not None:
            if pretrained_pos_embed.shape[1] == 198 and model.pos_embed.shape[1] == 197:
                print(f"Adjusting pos_embed from {pretrained_pos_embed.shape} to {model.pos_embed.shape}")
                # 裁剪掉最后一个位置编码（通常对应蒸馏 token）
                pretrained_pos_embed = pretrained_pos_embed[:, :-1, :]

            # 手动赋值给模型
            model.pos_embed = nn.Parameter(pretrained_pos_embed)

        # ------------------------------------------------------------
        # 兼容不同 checkpoint 的字段包装格式
        # ------------------------------------------------------------
        # 有些权重会包在 "model" 或 "state_dict" 下，需要先取出来
        if "model" in ckpt_dict:
            ckpt_dict = ckpt_dict["model"]
        if "state_dict" in ckpt_dict:
            ckpt_dict = ckpt_dict["state_dict"]

        # ------------------------------------------------------------
        # 构造新的权重字典
        # ------------------------------------------------------------
        # 这里的处理逻辑是：
        # 1. 忽略 head 相关参数（因为这里只做特征提取，不做最终分类）
        # 2. 如果 key 以 backbone 开头，则去掉该前缀
        # 3. pos_embed 单独处理过了，这里跳过，避免重复覆盖
        new_ckpt_dict = {}
        for key in ckpt_dict:
            if "head" not in key:
                if "backbone" in key:
                    new_key = key[9:]
                else:
                    new_key = key

                if new_key == 'pos_embed':
                    continue

                new_ckpt_dict[new_key] = ckpt_dict[key]

        # strict=False 允许部分层名不匹配
        model.load_state_dict(new_ckpt_dict, strict=False)

    print(f"Model {args.arch} built.")

    # ============================================================
    # 三、准备数据集与预处理
    # ============================================================
    data_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
    ])

    # 下面这套 transform 被注释掉了，应该是用于 CIFAR 风格归一化的备选方案
    # data_transform = pth_transforms.Compose([
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.200, 0.201)),
    # ])

    # 根据数据集名称构建对应的数据集对象
    if args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(
            root="data",
            train=args.split == "train",
            download=True,
            transform=data_transform
        )

    elif args.dataset == "CIFAR100":
        dataset = datasets.CIFAR100(
            root="data",
            train=args.split == "train",
            download=True,
            transform=data_transform
        )

    elif args.dataset == "ImageNet":
        root = os.path.join("data/ImageNet", 'train' if args.split == "train" else 'val')
        dataset = datasets.ImageFolder(root, transform=data_transform)

    elif args.dataset == "Ucity":
        root = os.path.join("../datasets/Ucity/images", 'train' if args.split == "train" else 'val')
        dataset = datasets.ImageFolder(root, transform=data_transform)

    elif args.dataset == "MIT_Place_Pulse":
        root = os.path.join("../datasets/MIT_Place_Pulse/images", 'train' if args.split == "train" else 'val')
        dataset = datasets.ImageFolder(root, transform=data_transform)

    else:
        raise NotImplementedError

    # 构建 DataLoader
    # 注意：这里 shuffle=False，因为提取特征时通常希望输出顺序与数据集原顺序一致
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Data loaded with {len(dataset)} imgs.")

    # ============================================================
    # 四、提取特征并保存
    # ============================================================
    features, labels = validate_network(
        data_loader,
        model,
        args.n_last_blocks,
        args.avgpool_patchtokens
    )

    # 将标签 reshape 成列向量，便于与特征按列拼接
    labels = labels.reshape(-1, 1)

    # 输出格式：
    # 每一行 = [feature_dim..., label]
    outputs = np.concatenate([features, labels], axis=1)

    # 保存为 .npy 文件
    np.save(
        os.path.join(args.output_dir, "%s%s%s.npy" % (args.dataset, args.extra_name, args.split)),
        outputs
    )


@torch.no_grad()
def validate_network(data_loader, model, n, avgpool):
    """
    使用给定模型对整个数据集做前向推理，提取特征与标签。

    功能：
        遍历 data_loader 中的所有样本，逐批送入模型，
        得到每个样本的特征表示，并收集对应标签。

    参数：
        data_loader : 数据加载器
        model       : 已构建并加载好权重的模型
        n           : 对 ViT 而言，取最后多少个 block 的 [CLS] token 进行拼接
        avgpool     : 是否额外拼接 patch token 的平均池化特征

    返回：
        features : numpy 数组，形状约为 [N, D]
        targets  : numpy 数组，形状约为 [N]
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    features = None
    targets = None

    for inp, target in metric_logger.log_every(data_loader, 20, header):
        # ------------------------------------------------------------
        # 1. 将输入图像搬到 GPU
        # ------------------------------------------------------------
        inp = inp.cuda(non_blocking=True)

        # ------------------------------------------------------------
        # 2. 前向推理，提取特征
        # ------------------------------------------------------------
        with torch.no_grad():
            # 如果是 ViT，则取中间层输出
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)

                # 拼接最后 n 个 block 的 CLS token
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

                # 如果开启 avgpool，则再额外拼接最后一层 patch token 的平均池化结果
                if avgpool:
                    output = torch.cat(
                        (
                            output.unsqueeze(-1),
                            torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)
                        ),
                        dim=-1
                    )
                    output = output.reshape(output.shape[0], -1)

            # 如果不是 ViT，则直接把模型输出作为特征
            else:
                output = model(inp)

        # 转成 numpy，便于后续统一拼接保存
        output = output.cpu().numpy()
        target = target.numpy()

        # ------------------------------------------------------------
        # 3. 收集当前 batch 的特征与标签
        # ------------------------------------------------------------
        if features is None:
            features = output
            targets = target
        else:
            features = np.concatenate([features, output], axis=0)
            targets = np.concatenate([targets, target], axis=0)

    return features, targets


if __name__ == '__main__':
    # ============================================================
    # main 主入口
    # ============================================================
    # 整体流程：
    #   1. 解析命令行参数
    #   2. 根据参数构建模型、数据集和输出路径
    #   3. 调用 extract_features 进行特征提取
    #   4. 将提取出的特征和标签保存成 .npy 文件
    #
    # 这个脚本本质上是一个“离线特征抽取工具”：
    #   给定一个预训练模型 + 一个数据集，
    #   输出每张图片对应的特征向量及标签。
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')

    parser.add_argument(
        '--n_last_blocks',
        default=1,
        type=int,
        help="""Concatenate [CLS] tokens for the `n` last blocks.
        We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base."""
    )

    parser.add_argument(
        '--avgpool_patchtokens',
        default=False,
        type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token. 
        We typically set this to False for ViT-Small and to True with ViT-Base."""
    )

    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")'
    )
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dataset', default="MIT_Place_Pulse", type=str, help='Dataset to extract features')
    parser.add_argument('--output_dir', default="subsets/MIT_Place_Pulse", help='Path to save extracted features')

    parser.add_argument('--split', default="train", type=str, help="Data split to extract features")
    parser.add_argument('--extra_name', default="_", type=str, help="extra filename")

    args = parser.parse_args()

    # 调用主功能函数，开始提取特征
    extract_features(args)
