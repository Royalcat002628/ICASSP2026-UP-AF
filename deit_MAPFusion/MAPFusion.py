import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    """
    空间注意力模块。

    作用：
        从空间维度上判断“哪些位置更重要”，为后续特征融合提供空间权重。

    基本思路：
        1. 对输入特征在通道维分别做平均池化和最大池化
        2. 将两者拼接后输入卷积层
        3. 通过 sigmoid 得到空间注意力图

    输入：
        x: [B, C, H, W]

    输出：
        sattn: [B, 1, H, W]
               表示每个空间位置的重要性权重
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 这里使用卷积来融合 avg/max 两种空间统计信息
        # 输入通道为 2（均值图 + 最大值图），输出通道为 1
        self.sa = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            padding_mode='reflect',
            bias=True
        )

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征张量，形状为 [B, C, H, W]

        返回：
            sattn: 空间注意力权重图，形状为 [B, 1, H, W]
        """
        # 在通道维度上做平均池化，得到每个位置的平均响应
        x_avg = torch.mean(x, dim=1, keepdim=True)

        # 在通道维度上做最大池化，得到每个位置的最强响应
        x_max, _ = torch.max(x, dim=1, keepdim=True)

        # 将平均图和最大图拼接，作为空间注意力的输入
        x2 = torch.cat([x_avg, x_max], dim=1)

        # 卷积提取空间注意力
        sattn = self.sa(x2)
        sattn = torch.sigmoid(sattn)

        return sattn


class ChannelAttention(nn.Module):
    """
    通道注意力模块。

    作用：
        从通道维度判断“哪些通道更重要”，突出有用特征通道，抑制无关通道。

    基本思路：
        1. 对输入特征做全局平均池化，得到每个通道的全局描述
        2. 经过两层 1x1 卷积形成通道注意力
        3. 通过 sigmoid 输出通道权重

    输入：
        x: [B, C, H, W]

    输出：
        cattn: [B, C, 1, 1]
               表示每个通道的重要性权重
    """
    def __init__(self, dim, reduction=8):
        """
        参数：
            dim: 输入通道数
            reduction: 通道压缩比例，用于降低中间层通道数
        """
        super(ChannelAttention, self).__init__()

        # 全局平均池化，将空间信息压缩成每个通道的全局描述
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 两层 1x1 卷积实现轻量级通道注意力
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        """
        前向传播。

        参数：
            x: 输入特征张量，形状为 [B, C, H, W]

        返回：
            cattn: 通道注意力权重，形状为 [B, C, 1, 1]
        """
        # 得到每个通道的全局统计特征
        x_gap = self.gap(x)

        # 生成通道注意力
        cattn = self.ca(x_gap)
        cattn = torch.sigmoid(cattn)

        return cattn


class PixelAttention(nn.Module):
    """
    像素注意力模块。

    作用：
        在更细粒度的像素层面上，对特征进行自适应加权。

    模块思路：
        1. 将原始特征 x 与上一阶段得到的注意力特征 pattn1 进行拼接
        2. 通过深度可分组卷积提取像素级注意力信息
        3. 使用 LayerNorm 对输出做归一化

    输入：
        x:      [B, C, H, W]，原始输入特征
        pattn1: [B, C, H, W]，由空间注意力和通道注意力组合得到的特征

    输出：
        out:    [B, C, H, W]，像素级注意力结果
    """
    def __init__(self, dim):
        """
        参数：
            dim: 输入通道数
        """
        super(PixelAttention, self).__init__()

        # groups=dim 表示按通道分组卷积，可理解为一种轻量级逐通道卷积方式
        self.pa2 = nn.Conv2d(
            in_channels=2 * dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            padding_mode='reflect',
            groups=dim,
            bias=True
        )

        # LayerNorm 只在通道维上做归一化
        self.norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        """
        前向传播。

        参数：
            x: 原始输入特征，形状 [B, C, H, W]
            pattn1: 中间注意力特征，形状 [B, C, H, W]

        返回：
            out: 像素注意力输出，形状 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 增加一个临时维度，便于和 pattn1 做拼接
        x = x.unsqueeze(dim=2)          # [B, C, 1, H, W]
        pattn1 = pattn1.unsqueeze(dim=2)  # [B, C, 1, H, W]

        # 在新维度上拼接，形成 [B, C, 2, H, W]
        x2 = torch.cat([x, pattn1], dim=2)

        # 重新整理维度，把 “通道数 × 2” 合并成卷积输入通道
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)  # [B, 2C, H, W]

        # 提取像素级注意力
        out = self.pa2(x2)

        # LayerNorm 的输入习惯是最后一维做归一化，因此先转成 [B, H, W, C]
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)

        # 再转回卷积常用格式 [B, C, H, W]
        out = out.permute(0, 3, 1, 2).contiguous()

        return out


class MAPFusion(nn.Module):
    """
    MAPFusion 融合模块。

    模块目标：
        对两个输入特征 x 和 y 进行融合，融合时综合利用：
        1. 空间注意力（Spatial Attention）
        2. 通道注意力（Channel Attention）
        3. 像素注意力（Pixel Attention）
        4. 可学习门控机制（Gate）
        5. BatchNorm 与残差增强

    核心思路：
        - 先把 x 和 y 做初步相加，得到 initial
        - 基于 initial 分别计算空间注意力和通道注意力
        - 再进一步生成像素级注意力 pattn2
        - 用 pattn2 控制 x 和 y 的融合比例
        - 同时再增加一个门控分支 alpha，对 x 和 y 做额外的可学习融合
        - 最后再经过归一化、残差和 1×1 卷积输出最终结果

    输入：
        x: [B, C, H, W]
        y: [B, C, H, W]

    输出：
        result: [B, C, H, W]
    """
    def __init__(self, dim, reduction=8):
        """
        参数：
            dim: 输入特征通道数
            reduction: 通道注意力中的压缩比例
        """
        super(MAPFusion, self).__init__()

        # 三种注意力模块
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

        # 输出映射卷积
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)

        # 融合后归一化
        self.bn = nn.BatchNorm2d(dim)
        self.sigmoid = nn.Sigmoid()

        # 可学习门控分支：
        # 输入 [B,C,H,W]，输出 [B,1,H,W]，表示每个位置对 x/y 的偏好
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        前向传播。

        参数：
            x: 输入特征 1，形状 [B, C, H, W]
            y: 输入特征 2，形状 [B, C, H, W]

        返回：
            result: 融合后的输出特征，形状 [B, C, H, W]
        """
        # ------------------------------------------------------------
        # 第一步：初步融合
        # ------------------------------------------------------------
        # 先进行简单相加，作为后续注意力计算的基础特征
        initial = x + y

        # ------------------------------------------------------------
        # 第二步：计算通道注意力与空间注意力
        # ------------------------------------------------------------
        cattn = self.ca(initial)   # [B, C, 1, 1]
        sattn = self.sa(initial)   # [B, 1, H, W]

        # 将两种注意力直接相加，形成联合注意力特征
        # 依赖广播机制，形状可对齐到 [B, C, H, W]
        pattn1 = sattn + cattn

        # ------------------------------------------------------------
        # 第三步：计算像素注意力
        # ------------------------------------------------------------
        # pa 输出是一个像素级特征，再经过 sigmoid 后作为权重调制项
        pattn2 = initial + self.sigmoid(self.pa(initial, pattn1))

        # ------------------------------------------------------------
        # 第四步：基于 pattn2 融合 x 与 y
        # ------------------------------------------------------------
        # pattn2 越大，越偏向 x；越小，越偏向 y
        result = initial + pattn2 * x + (1 - pattn2) * y

        # ------------------------------------------------------------
        # 第五步：额外的可学习门控融合
        # ------------------------------------------------------------
        # gate 输出 alpha，形状 [B, 1, H, W]
        # 表示每个位置上更偏向 x 还是 y
        alpha = torch.sigmoid(self.gate(initial))

        # 用 alpha 对 x 和 y 做门控融合
        gated = alpha * x + (1 - alpha) * y

        # 将门控结果再叠加到原有融合结果上
        result = result + gated

        # ------------------------------------------------------------
        # 第六步：归一化 + 残差增强 + 输出卷积
        # ------------------------------------------------------------
        result = self.bn(result) + result
        result = self.conv(result)

        return result


if __name__ == '__main__':
    """
    main 测试入口。

    作用：
        这个部分不是训练代码，而是一个最简单的功能测试示例，
        用来验证 MAPFusion 模块是否可以正常前向传播。

    流程：
        1. 实例化一个 MAPFusion 模块
        2. 构造两个随机输入张量
        3. 将两个张量输入模块
        4. 打印输出张量的尺寸
    """

    # 实例化融合模块，设定输入通道数为 64
    block = MAPFusion(64)

    # 构造两个随机输入特征
    # 形状均为 [B, C, H, W] = [3, 64, 64, 64]
    input1 = torch.rand(3, 64, 64, 64)
    input2 = torch.rand(3, 64, 64, 64)

    # 执行前向传播，得到融合结果
    output = block(input1, input2)

    # 打印输出大小，检查是否与预期一致
    print(output.size())
