import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import numpy as np


# ============================================================
# 1. INaturalist 数据集自定义类
# ============================================================
class INatDataset(ImageFolder):
    """
    INaturalist 数据集读取类（自定义实现）。

    作用：
        - 解析官方 JSON 标注文件
        - 构建 (image_path, label) 的映射关系
        - 支持按不同 taxonomy 层级（如 genus / family）分类

    与普通 ImageFolder 的区别：
        - 标签不是直接由文件夹决定，而是来自 JSON 标注
        - 使用 OpenCV + Albumentations 处理图像

    输入参数：
        root: 数据集根目录
        train: 是否训练集
        year: 数据集年份（2018 / 2019 等）
        category: 分类层级（如 name / genus / family 等）
    """

    def __init__(self, root, train=True, year=2025, transform=None,
                 target_transform=None, category='name', loader=default_loader):

        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year

        # -----------------------------
        # 1. 读取数据标注 JSON
        # -----------------------------
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        # 类别信息（category_id → 具体类别）
        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        # 用训练集构建 label 映射
        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        # -----------------------------
        # 2. 构建 label 映射（类别名 → index）
        # -----------------------------
        targeter = {}
        indexer = 0

        for elem in data_for_targeter['annotations']:
            # 根据 category_id 获取类别名称（如 genus）
            cat_name = data_catg[int(elem['category_id'])][category]

            if cat_name not in targeter:
                targeter[cat_name] = indexer
                indexer += 1

        self.nb_classes = len(targeter)

        # -----------------------------
        # 3. 构建样本列表
        # -----------------------------
        self.samples = []

        for elem in data['images']:
            cut = elem['file_name'].split('/')

            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]

            self.samples.append((path_current, target_current_true))

    def __getitem__(self, index):
        """
        重写 ImageFolder 的读取方式：

        改动点：
            - 使用 OpenCV 读取图像
            - 使用 Albumentations 做数据增强

        返回：
            image: Tensor
            target: int
        """
        path, target = self.samples[index]

        # OpenCV 读取（BGR）
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Albumentations 需要传入 dict
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


# ============================================================
# 2. ImageFolder + Albumentations 版本
# ============================================================
class AlbumentationsImageFolder(datasets.ImageFolder):
    """
    用 Albumentations 替代 torchvision transform 的 ImageFolder。
    """

    def __getitem__(self, index):
        path, target = self.samples[index]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, target


# ============================================================
# 3. 构建数据集（核心入口函数）
# ============================================================
def build_dataset(is_train, args):
    """
    根据参数构建数据集（训练 / 验证）。

    功能：
        - 根据 args.data_set 选择不同数据集
        - 自动应用 Albumentations 增强
        - 返回 dataset 和类别数

    返回：
        dataset: torch Dataset
        nb_classes: 类别数量
    """

    transform = build_transform(is_train, args)

    # -----------------------------
    # CIFAR100
    # -----------------------------
    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train,
                                    transform=None, download=True)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = 100

    # -----------------------------
    # CIFAR10
    # -----------------------------
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train,
                                   transform=None, download=True)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = 10

    # -----------------------------
    # ImageNet（简化版）
    # -----------------------------
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=None)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = 6  # ⚠️ 注意：这里是写死的

    # -----------------------------
    # INaturalist
    # -----------------------------
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train,
                              year=2018, category=args.inat_category,
                              transform=None)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train,
                              year=2019, category=args.inat_category,
                              transform=None)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = dataset.nb_classes

    # -----------------------------
    # 子集版本（用于实验）
    # -----------------------------
    elif args.data_set == 'CIFAR10SUBSET':
        dataset = datasets.CIFAR10(args.data_path, train=is_train,
                                   transform=None, download=True)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = 10

        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)

    # 其他类似（省略重复说明）
    elif args.data_set == 'CIFAR100SUBSET':
        dataset = datasets.CIFAR100(args.data_path, train=is_train,
                                    transform=None, download=True)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = 100

        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)

    elif args.data_set == 'IMNETSUBSET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=None)
        dataset = AlbumentationsWrapper(dataset, transform)
        nb_classes = 6

        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)

    return dataset, nb_classes


# ============================================================
# 4. 数据增强构建
# ============================================================
def build_transform(is_train, args):
    """
    构建 Albumentations 数据增强 pipeline。

    区别：
        - 训练集：随机裁剪 + 翻转 + 颜色扰动
        - 验证集：中心裁剪（更稳定）

    输出：
        Albumentations Compose 对象
    """

    if is_train:
        return A.Compose([
            A.SmallestMaxSize(max_size=args.input_size + 32),
            A.RandomCrop(height=args.input_size, width=args.input_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.8
            ),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=args.input_size + 32),
            A.CenterCrop(height=args.input_size, width=args.input_size),
            A.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ])


# ============================================================
# 5. Albumentations 包装器
# ============================================================
class AlbumentationsWrapper(torch.utils.data.Dataset):
    """
    将 torchvision dataset 包装成支持 Albumentations 的版本。

    作用：
        - 原 dataset 输出 PIL Image
        - 转换为 numpy → Albumentations → Tensor
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        # PIL → numpy
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, target

    def visualize_augmentations(dataset, idx=0, samples=5):
        """
        可视化数据增强效果（调试用）。

        功能：
            - 显示原图
            - 显示多次随机增强结果
        """
        import matplotlib.pyplot as plt

        # 去掉 Normalize / ToTensor 便于显示
        vis_transform = A.Compose([
            t for t in dataset.transform.transforms
            if not isinstance(t, (A.Normalize, ToTensorV2))
        ])

        original_image, _ = dataset[idx]

        if isinstance(original_image, torch.Tensor):
            original_image = original_image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(15, 3))
        plt.subplot(1, samples + 1, 1)
        plt.imshow(original_image.astype('uint8'))
        plt.title("Original")
        plt.axis('off')

        for i in range(samples):
            augmented = vis_transform(image=original_image)
            plt.subplot(1, samples + 1, i + 2)
            plt.imshow(augmented['image'])
            plt.title(f"Aug {i + 1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
