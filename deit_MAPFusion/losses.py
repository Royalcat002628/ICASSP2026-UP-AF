# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
本文件实现知识蒸馏损失（Knowledge Distillation Loss）。

核心思想：
    在普通监督学习损失的基础上，引入教师模型（teacher model）的输出，
    作为额外监督信号，帮助学生模型（student model）训练得更好。

支持三种模式：
    1. none : 不使用蒸馏，只计算基础分类损失
    2. soft : 使用 soft distillation（教师输出概率分布）
    3. hard : 使用 hard distillation（教师输出的 argmax 类别）
"""

import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    知识蒸馏损失封装类。

    作用：
        这个类在“基础损失函数”之上，再额外加入一个蒸馏损失项。
        训练时既利用真实标签 labels，也利用教师模型 teacher_model 的预测结果。

    整体损失形式：
        loss = (1 - alpha) * base_loss + alpha * distillation_loss

    其中：
        - base_loss：普通监督损失，例如 CrossEntropyLoss
        - distillation_loss：蒸馏损失
        - alpha：控制两者权重的系数

    适用场景：
        - DeiT / ViT 这类带蒸馏 token 的模型
        - Teacher-Student 训练框架
        - 希望借助教师模型输出提升学生模型性能时
    """

    def __init__(self,
                 base_criterion: torch.nn.Module,
                 teacher_model: torch.nn.Module,
                 distillation_type: str,
                 alpha: float,
                 tau: float):
        """
        初始化蒸馏损失模块。

        参数：
            base_criterion:
                基础损失函数，例如 CrossEntropyLoss。
                用于衡量学生模型主输出与真实标签之间的差异。

            teacher_model:
                教师模型。
                在训练过程中不参与反向传播，只提供额外监督信号。

            distillation_type:
                蒸馏类型，可选：
                    - 'none' : 不使用蒸馏
                    - 'soft' : 软蒸馏，使用教师输出的概率分布
                    - 'hard' : 硬蒸馏，使用教师输出的预测类别

            alpha:
                蒸馏损失所占权重。
                总损失中，base_loss 权重为 (1 - alpha)，distillation_loss 权重为 alpha。

            tau:
                蒸馏温度系数（temperature）。
                仅在 soft distillation 中使用，用于平滑概率分布。
        """
        super().__init__()

        self.base_criterion = base_criterion
        self.teacher_model = teacher_model

        # 限定蒸馏模式只能是三种之一
        assert distillation_type in ['none', 'soft', 'hard']

        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        前向计算总损失。

        参数：
            inputs:
                原始输入样本。
                教师模型会使用它来生成 teacher_outputs。

            outputs:
                学生模型的输出。
                有两种可能：
                    1. Tensor
                       表示模型只返回一个输出（普通分类输出）
                    2. Tuple[Tensor, Tensor]
                       表示模型返回两个输出：
                       - 第一个：主分类输出 outputs
                       - 第二个：蒸馏分支输出 outputs_kd

            labels:
                真实标签，用于计算基础监督损失。

        返回：
            loss:
                最终总损失。
        """
        # ------------------------------------------------------------
        # Step 1. 解析学生模型输出
        # ------------------------------------------------------------
        # 默认没有蒸馏分支输出
        outputs_kd = None

        if not isinstance(outputs, torch.Tensor):
            # 约定：
            # 如果模型返回的是 tuple，则格式应为 [主输出, 蒸馏输出]
            outputs, outputs_kd = outputs

        # ------------------------------------------------------------
        # Step 2. 计算基础监督损失
        # ------------------------------------------------------------
        # 这是最普通的分类损失，如交叉熵
        base_loss = self.base_criterion(outputs, labels)

        # 如果当前不启用蒸馏，直接返回基础损失
        if self.distillation_type == 'none':
            return base_loss

        # ------------------------------------------------------------
        # Step 3. 检查蒸馏输出是否合法
        # ------------------------------------------------------------
        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )

        # ------------------------------------------------------------
        # Step 4. 获取教师模型输出
        # ------------------------------------------------------------
        # 注意：教师模型只提供监督，不参与反向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # ------------------------------------------------------------
        # Step 5. 根据蒸馏类型计算 distillation_loss
        # ------------------------------------------------------------
        if self.distillation_type == 'soft':
            # ========== 软蒸馏 ==========
            # 思想：
            #   让学生模型的蒸馏分支输出分布，去逼近教师模型输出分布
            # 方法：
            #   使用带温度 T 的 KL 散度

            T = self.tau

            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                # 这里给 teacher 提供的是 log probability
                # 并设置 log_target=True
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()

            # 说明：
            # 1. 乘以 T^2 是蒸馏中的常见做法，用于温度缩放后的梯度校正
            # 2. 除以 outputs_kd.numel() 是为了保持旧版 PyTorch 行为

        elif self.distillation_type == 'hard':
            # ========== 硬蒸馏 ==========
            # 思想：
            #   直接把教师模型预测概率最大的类别，当作学生蒸馏分支的监督标签
            distillation_loss = F.cross_entropy(
                outputs_kd,
                teacher_outputs.argmax(dim=1)
            )

        # ------------------------------------------------------------
        # Step 6. 融合基础损失与蒸馏损失
        # ------------------------------------------------------------
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss
