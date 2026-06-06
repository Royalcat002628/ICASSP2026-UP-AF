# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             classification_report, average_precision_score)



def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for samples, targets in tqdm(metric_logger.log_every(data_loader, print_freq, header),
                                     desc=f"Epoch {epoch}", ncols=100, leave=False):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}






def evaluate_test(data_loader, model, device, result_dir):
    """
    对测试集数据进行评估，计算各项指标，
    同时将混淆矩阵图和评估报告保存到 result_dir 目录下。
    """
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    model.eval()  # 设置评估模式
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

            # 通过 softmax 转换为概率矩阵
            probs = torch.softmax(outputs, dim=1)  # 转换为概率
            preds = outputs.argmax(dim=1)  # 获取类别索引

            # 打印 y_pred 的形状（检查是否是一个概率矩阵）
            print(f"Shape of y_pred (probs): {probs.shape}")  # 输出形状检查 (batch_size, num_classes)

            # 将目标标签、预测标签和预测概率添加到列表
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # **检查 y_true 的标签范围**
    print("Unique labels in y_true:", np.unique(all_targets))  # 打印所有唯一的标签值
    if np.max(all_targets) >= 6:
        print("Warning: Some labels are out of bounds!")
        all_targets = all_targets[all_targets < 6]  # 只保留有效标签
        all_probs = all_probs[all_targets < 6]    # 确保 y_pred 和 y_true 保持一致

    # 计算指标（多分类采用 weighted 平均）
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    conf_mat = confusion_matrix(all_targets, all_preds)

    try:
        # 计算 ROC AUC，multi_class='ovr' 是多分类时使用的选项
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except Exception as e:
        roc_auc = None
        print("ROC AUC calculation error:", e)

    # 计算平均精度，multi_class='ovr' 是多分类时使用的选项
    avg_precision = average_precision_score(all_targets, all_probs, average='weighted')

    # 生成分类报告
    class_report = classification_report(all_targets, all_preds)

    # 绘制混淆矩阵图并保存到 result_dir 目录
    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(all_targets)))
    plt.xticks(tick_marks, np.unique(all_targets))
    plt.yticks(tick_marks, np.unique(all_targets))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # 保存评估报告文本文件到 result_dir 目录
    report_path = os.path.join(result_dir, "test_evaluation.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_mat))
        f.write("\n")
        f.write(f"ROC AUC: {roc_auc}\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write(f"\nAverage Precision: {avg_precision:.4f}\n")

    # 返回所有指标和文件路径
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_mat.tolist(),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "classification_report": class_report,
        "cm_path": cm_path,
        "report_path": report_path
    }

    return metrics



