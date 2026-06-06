# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

# 自定义特征融合模块：用于融合局部 patch 特征与 Transformer 全局特征
from MAPFusion import MAPFusion

# 自定义 Vision Transformer 及分类头
from vision_transformer import VisionTransformer, ClassificationHead

# timm 中的默认配置函数
from timm.models.vision_transformer import _cfg

# timm 的模型注册装饰器与参数初始化函数
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import warnings
warnings.filterwarnings('ignore')


# 该文件对外暴露的模型名称列表
__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 'deit_small_head12_patch16_224'
]


class DistilledVisionTransformer(VisionTransformer):
    """
    带蒸馏分支的 Vision Transformer。

    相比基础 VisionTransformer，这个类做了几件重要的扩展：

    1. 新增 distillation token（dist_token）
       - 用于蒸馏分支的特征提取
       - 因此位置编码 pos_embed 长度也从 (num_patches + 1) 变为 (num_patches + 2)

    2. 新增蒸馏头 head_dist
       - 与主分类头 head 并行
       - 训练阶段返回两个输出：(主输出, 蒸馏输出)
       - 推理阶段返回二者平均结果

    3. 将原始单层分类头替换为两层 MLP 分类头 ClassificationHead
       - 提升分类头表达能力

    4. 新增 MAPFusion 模块
       - 将 patch embedding 后得到的“局部特征”与 Transformer block 输出的“全局特征”融合
       - 融合后做全局平均池化，再送入分类头和蒸馏头

    整体思路：
        输入图像
          -> patch embedding
          -> 得到局部 patch 特征 local_feat
          -> 加入 cls_token / dist_token 后通过 Transformer blocks
          -> 得到全局 patch 特征 global_feat
          -> 用 MAPFusion(local_feat, global_feat) 融合
          -> 全局池化
          -> 送入 head 和 head_dist
    """

    def __init__(self, *args, **kwargs):
        """
        初始化蒸馏版 Vision Transformer。

        主要新增模块：
            - dist_token
            - 新的 pos_embed
            - 蒸馏头 head_dist
            - 两层 MLP 主分类头 head
            - MAPFusion 融合模块
        """
        super().__init__(*args, **kwargs)

        # ============================================================
        # 1. 蒸馏 token 与位置编码
        # ============================================================
        # 原始 ViT 中只有 cls_token，
        # 这里新增一个 dist_token 以支持蒸馏分支
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches

        # 原始 ViT 的位置编码长度为 num_patches + 1（只含 CLS）
        # 这里需要变成 num_patches + 2（CLS + DIST）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))

        # ============================================================
        # 2. 蒸馏分支分类头
        # ============================================================
        # 若 num_classes > 0，则使用线性层输出分类结果
        # 否则直接使用 Identity
        self.head_dist = (
            nn.Linear(self.embed_dim, self.num_classes)
            if self.num_classes > 0 else nn.Identity()
        )

        # ============================================================
        # 3. 主分类头
        # ============================================================
        # 这里将原本可能是单层线性分类头，替换为两层 MLP 分类头
        # hidden_dim=512, dropout=0.5 为当前设置
        self.head = ClassificationHead(
            self.embed_dim,
            hidden_dim=512,
            num_classes=self.num_classes,
            dropout=0.5
        )

        # ============================================================
        # 4. 参数初始化
        # ============================================================
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

        # ============================================================
        # 5. MAPFusion 融合模块
        # ============================================================
        # dim 使用 transformer 的 embed_dim
        self.cga = MAPFusion(dim=self.embed_dim, reduction=8)

    def forward_features(self, x):
        """
        提取融合后的特征表示。

        具体流程：
            1. 先对输入图像做 patch embedding，得到 patch token
            2. 将 patch token reshape 成二维特征图，作为“局部特征” local_feat
            3. 插入 cls_token 和 dist_token，并加上位置编码
            4. 经过 Transformer blocks 提取“全局特征”
            5. 取出 Transformer 输出中的 patch token，再 reshape 成二维特征图 global_feat
            6. 将 local_feat 和 global_feat 送入 MAPFusion 融合
            7. 对融合结果做全局平均池化，得到 [B, C] 向量
            8. 返回两份 pooled 特征，分别给主分类头和蒸馏头使用

        参数：
            x: 输入图像张量，形状一般为 [B, 3, H, W]

        返回：
            pooled, pooled:
                两个形状为 [B, C] 的张量
                当前实现中两者相同，分别提供给 head 和 head_dist
        """
        B = x.shape[0]

        # ------------------------------------------------------------
        # 1. Patch Embedding
        # ------------------------------------------------------------
        # 输出形状：[B, N, C]
        # N 为 patch 数量，C 为 embedding 维度
        x = self.patch_embed(x)
        N = x.shape[1]
        C = x.shape[2]

        # 假设 patch 数量 N 可以构成 H × W 的正方形网格
        H = W = int(N ** 0.5)

        # ------------------------------------------------------------
        # 2. 构造局部特征 local_feat
        # ------------------------------------------------------------
        # 将 patch token 从 [B, N, C] 转成 [B, C, H, W]
        patch_feats = x
        local_feat = patch_feats.permute(0, 2, 1).view(B, C, H, W)

        # ------------------------------------------------------------
        # 3. 插入 CLS token 与 DIST token
        # ------------------------------------------------------------
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, C]
        dist_token = self.dist_token.expand(B, -1, -1)  # [B, 1, C]

        # 拼接后形状：[B, N+2, C]
        x = torch.cat((cls_tokens, dist_token, patch_feats), dim=1)

        # 加上位置编码并做 dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # ------------------------------------------------------------
        # 4. Transformer Blocks
        # ------------------------------------------------------------
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # ------------------------------------------------------------
        # 5. 构造全局特征 global_feat
        # ------------------------------------------------------------
        # x[:, 0] 是 cls_token
        # x[:, 1] 是 dist_token
        # x[:, 2:] 是 patch tokens
        patch_tokens = x[:, 2:]
        global_feat = patch_tokens.permute(0, 2, 1).view(B, C, H, W)

        # ------------------------------------------------------------
        # 6. MAPFusion 融合局部与全局特征
        # ------------------------------------------------------------
        fused = self.cga(local_feat, global_feat)

        # ------------------------------------------------------------
        # 7. 全局平均池化
        # ------------------------------------------------------------
        # 将 [B, C, H, W] 压缩为 [B, C]
        pooled = fused.mean(dim=(2, 3))

        # ------------------------------------------------------------
        # 8. 返回两份特征
        # ------------------------------------------------------------
        # 当前实现中主分类头和蒸馏头共享同一个融合特征
        return pooled, pooled

    def forward(self, x):
        """
        前向传播。

        训练阶段：
            返回 (x, x_dist)
            - x      : 主分类头输出
            - x_dist : 蒸馏头输出

        推理阶段：
            返回 (x + x_dist) / 2
            即主分类头与蒸馏头输出的平均

        参数：
            x: 输入图像张量

        返回：
            训练时返回 tuple
            测试时返回平均 logits
        """
        # 提取融合后的 pooled 特征
        x, x_dist = self.forward_features(x)

        # 分别经过主分类头和蒸馏头
        x = self.head(x)
        x_dist = self.head_dist(x_dist)

        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


# ============================================================
# 以下是不同尺寸 / 配置的 DeiT 模型注册函数
# 它们的共同逻辑基本一致：
#   1. 从 kwargs 中弹出 timm 可能额外传入的配置项
#   2. 构建具体模型
#   3. 设置 default_cfg
#   4. 如需 pretrained，则加载官方预训练权重
# ============================================================

@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    构建 deit_tiny_patch16_224。

    配置：
        - patch_size = 16
        - embed_dim = 192
        - depth = 12
        - num_heads = 3
        - 输入尺寸默认 224

    参数：
        pretrained: 是否加载官方预训练权重
        **kwargs  : 其余模型参数

    返回：
        model: VisionTransformer 实例
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """
    构建 deit_small_patch16_224。

    配置：
        - embed_dim = 384
        - depth = 12
        - num_heads = 6
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_small_head12_patch16_224(pretrained=False, **kwargs):
    """
    构建一个自定义版本的 deit_small：
    与 deit_small_patch16_224 类似，但将注意力头数改为 12。

    配置：
        - embed_dim = 384
        - depth = 12
        - num_heads = 12
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """
    构建 deit_base_patch16_224。

    配置：
        - embed_dim = 768
        - depth = 12
        - num_heads = 12
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """
    构建蒸馏版 deit_tiny_patch16_224。

    特点：
        - 使用 DistilledVisionTransformer
        - 包含主分类头与蒸馏头
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """
    构建蒸馏版 deit_small_patch16_224。

    特点：
        - embed_dim = 384
        - num_heads = 6
        - 使用 DistilledVisionTransformer
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """
    构建蒸馏版 deit_base_patch16_224。

    特点：
        - embed_dim = 768
        - num_heads = 12
        - 使用 DistilledVisionTransformer
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """
    构建 deit_base_patch16_384。

    特点：
        - 输入尺寸为 384
        - 非蒸馏版
        - embed_dim = 768
        - num_heads = 12
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """
    构建蒸馏版 deit_base_patch16_384。

    特点：
        - 输入尺寸为 384
        - 使用 DistilledVisionTransformer
        - 推理时输出主头与蒸馏头平均结果
    """
    pretrained_cfg = kwargs.pop('pretrained_cfg', None)
    pretrained_cfg_overlay = kwargs.pop('pretrained_cfg_overlay', None)
    cache_dir = kwargs.pop('cache_dir', None)

    model = DistilledVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )

    default_cfg = _cfg()
    if pretrained_cfg is not None:
        default_cfg.update(pretrained_cfg)
    if pretrained_cfg_overlay is not None:
        default_cfg.update(pretrained_cfg_overlay)
    model.default_cfg = default_cfg

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu",
            check_hash=True,
            cache_dir=cache_dir
        )
        model.load_state_dict(checkpoint["model"])

    return model
