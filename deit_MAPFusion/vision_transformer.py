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

"""
Vision Transformer 相关模块文件（vision_transformer.py）

文件作用：
    该文件实现了一个较完整的 Vision Transformer 体系，包括：

    1. ClassificationHead
       - 自定义分类头

    2. drop_path / DropPath
       - 随机深度（Stochastic Depth）

    3. Mlp
       - Transformer block 中的前馈网络（FFN）

    4. Attention
       - 多头自注意力模块

    5. Block
       - Transformer 基本块（Attention + MLP + 残差 + Norm）

    6. PatchEmbed
       - 将图像切分为 patch，并映射为 token

    7. VisionTransformer
       - 核心 ViT 主体网络

    8. vit_tiny / vit_small / vit_base
       - 三个常见 ViT 配置构造函数

    9. DINOHead
       - DINO 自监督训练中使用的投影头

整体数据流可以概括为：
    输入图像
      -> PatchEmbed 切块并线性映射
      -> 拼接 CLS token
      -> 加位置编码
      -> 多层 Transformer Block
      -> 取 CLS token
      -> 分类头输出结果
"""

import math
from functools import partial

import torch
import torch.nn as nn

from util import trunc_normal_


# ============================================================
# 1. 分类头
# ============================================================
class ClassificationHead(nn.Module):
    """
    自定义分类头。

    作用：
        接收 Vision Transformer 输出的 CLS token 特征，
        通过两层全连接网络完成最终分类。

    结构：
        Linear -> ReLU -> Dropout -> Linear

    参数：
        in_features : 输入特征维度
        hidden_dim  : 隐藏层维度
        num_classes : 分类类别数
        dropout     : Dropout 概率
    """
    def __init__(self, in_features, hidden_dim=512, num_classes=6, dropout=0.5):
        super().__init__()

        # 第一层全连接：输入特征 -> 隐藏层
        self.fc1 = nn.Linear(in_features, hidden_dim)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

        # Dropout 正则化
        self.dropout = nn.Dropout(dropout)

        # 第二层全连接：隐藏层 -> 类别数
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化分类头中的线性层权重。

        这里使用 Kaiming Normal 初始化，
        更适合搭配 ReLU 激活函数。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播。

        输入：
            x : 一般是 CLS token，对应形状 [B, in_features]

        输出：
            x : 分类 logits，形状 [B, num_classes]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 2. Drop Path（随机深度）
# ============================================================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop Path 的函数实现。

    作用：
        在训练过程中随机丢弃某些残差分支，
        用于提高深层网络训练稳定性，类似于 Stochastic Depth。

    参数：
        x         : 输入张量
        drop_prob : 丢弃概率
        training  : 是否处于训练模式

    返回：
        处理后的张量
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob

    # 这里 shape 的写法兼容不同维度张量，不局限于 2D 特征
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化为 0/1 mask

    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop Path 的模块封装版本。

    作用：
        方便直接作为 nn.Module 插入网络结构中。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播时调用 drop_path 函数。
        """
        return drop_path(x, self.drop_prob, self.training)


# ============================================================
# 3. MLP 前馈网络
# ============================================================
class Mlp(nn.Module):
    """
    Transformer Block 中的前馈网络（Feed-Forward Network）。

    结构：
        Linear -> GELU -> Dropout -> Linear -> Dropout

    参数：
        in_features     : 输入特征维度
        hidden_features : 隐藏层维度，默认等于输入维度
        out_features    : 输出特征维度，默认等于输入维度
        act_layer       : 激活函数类型
        drop            : Dropout 概率
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播。
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# 4. 多头自注意力
# ============================================================
class Attention(nn.Module):
    """
    多头自注意力模块（Multi-Head Self-Attention）。

    作用：
        对输入 token 序列建模全局依赖关系，
        是 Transformer 的核心模块。

    参数：
        dim        : token 特征维度
        num_heads  : 注意力头数
        qkv_bias   : 是否给 q/k/v 线性映射添加 bias
        qk_scale   : 注意力缩放系数，若为空则自动使用 head_dim^(-0.5)
        attn_drop  : 注意力权重 dropout
        proj_drop  : 输出投影 dropout
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 注意力缩放系数
        self.scale = qk_scale or head_dim ** -0.5

        # 一次性线性映射出 q / k / v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播。

        输入：
            x : [B, N, C]
                B: batch size
                N: token 数量
                C: token 特征维度

        返回：
            x    : 注意力输出，形状仍为 [B, N, C]
            attn : 注意力权重，形状约为 [B, num_heads, N, N]
        """
        B, N, C = x.shape

        # 线性映射得到 qkv，并调整维度
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # softmax 归一化
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和并恢复维度
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


# ============================================================
# 5. Transformer 基本块
# ============================================================
class Block(nn.Module):
    """
    Vision Transformer 的基本 Block。

    结构：
        x -> Norm -> Attention -> Residual
          -> Norm -> MLP       -> Residual

    参数：
        dim         : token 特征维度
        num_heads   : 注意力头数
        mlp_ratio   : MLP 隐藏层扩展倍率
        qkv_bias    : qkv 是否带 bias
        qk_scale    : 注意力缩放系数
        drop        : dropout 概率
        attn_drop   : attention dropout
        drop_path   : stochastic depth 概率
        act_layer   : 激活函数
        norm_layer  : 归一化层
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, return_attention=False):
        """
        前向传播。

        参数：
            x                : 输入 token
            return_attention : 若为 True，则只返回注意力图

        返回：
            - 若 return_attention=False：返回更新后的 token
            - 若 return_attention=True ：返回注意力权重
        """
        y, attn = self.attn(self.norm1(x))

        if return_attention:
            return attn

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ============================================================
# 6. 图像切块嵌入
# ============================================================
class PatchEmbed(nn.Module):
    """
    图像到 Patch Token 的嵌入模块。

    作用：
        将输入图像按 patch_size 切成若干不重叠 patch，
        然后通过卷积映射到 embed_dim 维度，得到 token 序列。

    本质上：
        Conv2d(kernel_size=patch_size, stride=patch_size)
        等价于“分块 + 线性投影”。

    参数：
        img_size  : 输入图像尺寸
        patch_size: patch 大小
        in_chans  : 输入通道数
        embed_dim : token 嵌入维度
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        前向传播。

        输入：
            x : [B, C, H, W]

        输出：
            x : [B, N, C]
                N 为 patch 数量，C 为嵌入维度
        """
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ============================================================
# 7. Vision Transformer 主体
# ============================================================
class VisionTransformer(nn.Module):
    """
    Vision Transformer 主模型。

    整体流程：
        1. 输入图像切分为 patch，并映射为 token
        2. 在 token 序列最前面拼接 CLS token
        3. 加上位置编码
        4. 通过多层 Transformer Block 提取特征
        5. 取 CLS token 作为全局图像表示
        6. 送入分类头得到 logits
        7. 当前实现中，最后还做了 softmax，返回类别概率

    参数：
        img_size       : 输入图像尺寸（列表形式，实际使用 img_size[0]）
        patch_size     : patch 大小
        in_chans       : 输入通道数
        num_classes    : 类别数
        embed_dim      : token 维度
        depth          : Transformer Block 层数
        num_heads      : 注意力头数
        mlp_ratio      : MLP 扩展倍率
        qkv_bias       : qkv 是否使用 bias
        qk_scale       : 注意力缩放因子
        drop_rate      : dropout 概率
        attn_drop_rate : attention dropout
        drop_path_rate : stochastic depth 概率
        norm_layer     : 归一化层
    """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=6,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # ------------------------------------------------------------
        # 1. Patch Embedding
        # ------------------------------------------------------------
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # ------------------------------------------------------------
        # 2. CLS token 与位置编码
        # ------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ------------------------------------------------------------
        # 3. Transformer Blocks
        # ------------------------------------------------------------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # ------------------------------------------------------------
        # 4. 分类头
        # ------------------------------------------------------------
        self.head = ClassificationHead(
            embed_dim,
            hidden_dim=512,
            num_classes=num_classes,
            dropout=0.5
        )

        # ------------------------------------------------------------
        # 5. 参数初始化
        # ------------------------------------------------------------
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化模块权重。

        规则：
            - Linear 使用截断正态初始化
            - Linear bias 初始化为 0
            - LayerNorm 的 bias 为 0，weight 为 1
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        对位置编码做插值，适配不同输入分辨率。

        作用：
            若当前输入图像尺寸与训练时不同，patch 数也会变化，
            这时需要对位置编码进行插值。

        参数：
            x : 当前 token 序列
            w : 输入图像宽
            h : 输入图像高

        返回：
            与当前输入 token 数量匹配的位置编码
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size

        # 加一个很小的偏移，避免浮点误差
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """
        将原始图像转换为 Transformer 可处理的 token 序列。

        流程：
            1. Patch embedding
            2. 拼接 CLS token
            3. 加位置编码
            4. 做 dropout

        输入：
            x : [B, C, W, H]

        返回：
            token 序列，形状 [B, N+1, C]
        """
        B, nc, w, h = x.shape

        # 1. 图像 -> patch token
        x = self.patch_embed(x)

        # 2. 拼接 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. 加位置编码
        x = x + self.interpolate_pos_encoding(x, w, h)

        # 4. dropout
        return self.pos_drop(x)

    def forward(self, x):
        """
        主前向传播函数。

        流程：
            1. 调用 prepare_tokens 生成 token 序列
            2. 通过多层 Transformer block
            3. LayerNorm
            4. 取 CLS token
            5. 送入分类头得到 logits
            6. 对 logits 做 softmax，输出类别概率

        输入：
            x : 输入图像

        返回：
            probs : 每个类别的预测概率，形状 [B, num_classes]
        """
        # 1. 图像转 token
        x = self.prepare_tokens(x)

        # 2. 通过 Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 3. 归一化
        x = self.norm(x)

        # 4. 提取 CLS token
        cls_token = x[:, 0]

        # 5. 分类头输出 logits
        logits = self.head(cls_token)

        # 6. softmax 转为概率
        probs = torch.softmax(logits, dim=1)

        return probs

    def get_last_selfattention(self, x):
        """
        获取最后一个 Transformer Block 的自注意力图。

        用途：
            常用于可视化模型关注区域。
        """
        x = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        """
        获取最后 n 个 Transformer Block 的输出。

        用途：
            常用于特征提取、可解释性分析、线性探测等任务。

        参数：
            x : 输入图像
            n : 返回最后多少层的输出

        返回：
            output : 长度为 n 的列表，每个元素是对应层的 token 输出
        """
        x = self.prepare_tokens(x)

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))

        return output


# ============================================================
# 8. 三个常见 ViT 配置构造函数
# ============================================================
def vit_tiny(patch_size=16, **kwargs):
    """
    构建 ViT-Tiny。

    默认配置：
        - embed_dim = 192
        - depth = 12
        - num_heads = 3
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    """
    构建 ViT-Small。

    默认配置：
        - embed_dim = 384
        - depth = 12
        - num_heads = 6
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    """
    构建 ViT-Base。

    默认配置：
        - embed_dim = 768
        - depth = 12
        - num_heads = 12
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# ============================================================
# 9. DINO 投影头
# ============================================================
class DINOHead(nn.Module):
    """
    DINO 自监督学习中使用的投影头。

    作用：
        将 backbone 输出特征映射到对比 / 自蒸馏训练所需的表示空间。

    结构：
        - 若 nlayers=1，则仅一个线性层
        - 若 nlayers>1，则为多层 MLP
        - 最后接 weight-normalized linear layer

    参数：
        in_dim          : 输入特征维度
        out_dim         : 输出特征维度
        use_bn          : 是否使用 BatchNorm
        norm_last_layer : 是否冻结最后一层的 weight_g
        nlayers         : MLP 层数
        hidden_dim      : 隐藏层维度
        bottleneck_dim  : bottleneck 维度
    """
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()

        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]

            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        # 最后一层采用 weight norm
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """
        初始化 DINOHead 中线性层的权重。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播。

        流程：
            1. MLP 映射
            2. L2 归一化
            3. 最后一层线性映射

        返回：
            投影后的特征
        """
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
