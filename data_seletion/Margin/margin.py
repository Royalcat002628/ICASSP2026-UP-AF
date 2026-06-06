#!/usr/bin/env python3
"""
tri_laf_auto_tau.py

功能说明：
    本脚本基于 TriLAF 的 Stage 3 选择逻辑，实现：
    1. 读取已有 Stage1 / Stage2 样本索引；
    2. 在剩余未标注样本中，根据不确定性评分方法进行排序；
    3. 支持自动根据全量 margin 分布的某个百分位选择 tau；
    4. 支持动态更新 rho；
    5. 支持去重（dedupe）；
    6. 最终输出新的样本子集 JSON。

额外说明：
    该版本还包含各阶段耗时统计，便于分析运行瓶颈。
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time  # 用于时间统计


def train_linear_head(core_feats, core_labels, num_classes, epochs, lr, device):
    """
    训练一个简单线性分类头。

    作用：
        使用已标注核心样本（core_feats, core_labels）训练一个线性探头，
        作为后续 margin、EGL 等方法的不确定性估计基础模型。

    参数：
        core_feats:   已标注样本特征，形状 [n_core, d]
        core_labels:  已标注样本标签，形状 [n_core]
        num_classes:  类别数
        epochs:       训练轮数
        lr:           学习率
        device:       运行设备（cpu / cuda）

    返回：
        训练好的 nn.Linear 模型
    """
    feats = torch.tensor(core_feats, dtype=torch.float32).to(device)
    labs = torch.tensor(core_labels, dtype=torch.long).to(device)

    model = nn.Linear(feats.size(1), num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()

    return model


def compute_single_margin(model, features, device):
    """
    计算单个线性头下的 margin。

    定义：
        margin = top1_logit - top2_logit

    含义：
        - margin 越大，模型越自信
        - margin 越小，样本越不确定

    参数：
        model:    已训练好的线性分类头
        features: 待评估特征，形状 [n, d]
        device:   运行设备

    返回：
        margins: shape = [n] 的 numpy 数组
    """
    model.eval()
    with torch.no_grad():
        feats = torch.tensor(features, dtype=torch.float32).to(device)
        logits = model(feats)  # [n, C]

        # 取每个样本最大的两个 logit
        top2 = torch.topk(logits, 2, dim=1).values  # [n, 2]

        # margin = 最大 logit - 第二大 logit
        margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()

    return margins


def compute_ensemble_margin(core_feats, core_labels, features, num_classes,
                            epochs, lr, device, num_heads):
    """
    通过多个随机初始化线性头组成 ensemble，计算 margin 的均值与方差。

    思想：
        - 每个 head 独立训练
        - 对同一个样本，多个 head 会给出不同的 margin
        - 用均值和方差一起衡量不确定性

    参数：
        core_feats, core_labels: 训练集特征与标签
        features:                全部待评估特征
        num_classes:             类别数
        epochs, lr, device:      训练参数
        num_heads:               线性头个数

    返回：
        mean_margin: [n]，各样本 margin 的均值
        var_margin:  [n]，各样本 margin 的方差
    """
    all_margins = []

    for h in range(num_heads):
        # 让每个 head 采用不同随机初始化
        torch.manual_seed(1234 + h)

        head = train_linear_head(
            core_feats, core_labels, num_classes,
            epochs, lr, device
        )

        marg = compute_single_margin(head, features, device)
        all_margins.append(marg)

    all_margins = np.stack(all_margins, axis=0)  # [num_heads, n]

    mean_margin = all_margins.mean(axis=0)
    var_margin = all_margins.var(axis=0)

    return mean_margin, var_margin


def compute_mcdrop_margins_and_probs(core_feats, core_labels, features,
                                     num_classes, epochs, lr, device,
                                     dropout_p, mc_iters):
    """
    使用 MC-Dropout 估计 margin 和概率分布。

    流程：
        1. 训练一个带 Dropout 的线性分类头
        2. 在测试时重复前向 mc_iters 次
        3. 每次记录：
           - margin
           - softmax 概率
        4. 统计 margin 的均值和方差

    参数：
        core_feats, core_labels: 已标注训练样本
        features:                待评估样本特征
        num_classes:             类别数
        epochs, lr, device:      训练参数
        dropout_p:               Dropout 概率
        mc_iters:                MC 前向次数

    返回：
        mean_margin: [n]
        var_margin:  [n]
        probs_mc:    [T, n, C]
    """
    class DropoutLinear(nn.Module):
        """
        带 Dropout 的简单线性分类器。
        """
        def __init__(self, in_dim, out_dim, p):
            super().__init__()
            self.dropout = nn.Dropout(p)
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            x = self.dropout(x)
            return self.fc(x)

    feats_t = torch.tensor(core_feats, dtype=torch.float32).to(device)
    labs_t = torch.tensor(core_labels, dtype=torch.long).to(device)

    model = DropoutLinear(core_feats.shape[1], num_classes, dropout_p).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 先正常训练
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(feats_t)
        loss = criterion(outputs, labs_t)
        loss.backward()
        optimizer.step()

    # 再做多次随机前向
    model.eval()
    margins_mc = []
    probs_mc = []

    with torch.no_grad():
        feats_all = torch.tensor(features, dtype=torch.float32).to(device)

        for _ in range(mc_iters):
            logits = model(feats_all)  # [n, C]
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # [n, C]
            top2 = torch.topk(logits, 2, dim=1).values
            margin_iter = (top2[:, 0] - top2[:, 1]).cpu().numpy()

            margins_mc.append(margin_iter)
            probs_mc.append(probs)

    margins_mc = np.stack(margins_mc, axis=0)  # [T, n]
    probs_mc = np.stack(probs_mc, axis=0)      # [T, n, C]

    mean_margin = margins_mc.mean(axis=0)
    var_margin = margins_mc.var(axis=0)

    return mean_margin, var_margin, probs_mc


def compute_egl_scores(model, features, device):
    """
    计算 Expected Gradient Length (EGL) 分数。

    含义：
        EGL 衡量“如果给这个样本加上标签进行训练，预期会带来多大的梯度更新”。
        分数越大，说明样本越值得标注。

    公式直观理解：
        EGL(x) = ||f(x)|| * Σ_y [ p(y|x) * ||p(·|x) - one_hot(y)|| ]

    参数：
        model:    训练好的线性分类头
        features: 待评估特征 [n, d]
        device:   运行设备

    返回：
        egl_scores: [n]，值越大越不确定
    """
    model.eval()
    with torch.no_grad():
        feats = torch.tensor(features, dtype=torch.float32).to(device)
        logits = model(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # [n, C]

    features_np = features
    n, C = probs.shape

    # 特征范数
    f_norm = np.linalg.norm(features_np, axis=1)  # [n]

    egl_scores = np.zeros(n, dtype=float)
    eyeC = np.eye(C)

    for i in range(n):
        p = probs[i]
        sum_term = 0.0
        for y in range(C):
            v = p - eyeC[y]
            sum_term += p[y] * np.linalg.norm(v)
        egl_scores[i] = f_norm[i] * sum_term

    return egl_scores


def compute_bald_scores(probs_mc):
    """
    计算 BALD 分数（Bayesian Active Learning by Disagreement）。

    含义：
        BALD 衡量模型参数不确定性，即：
            I[y, ω | x] = H(E[p(y|x,ω)]) - E[H(p(y|x,ω)])]

        分数越大，说明不同采样前向之间分歧越大，样本越值得标注。

    参数：
        probs_mc: [T, n, C]，MC-Dropout 下多次前向得到的概率分布

    返回：
        bald_scores: [n]
    """
    mean_probs = probs_mc.mean(axis=0)  # [n, C]
    eps = 1e-12

    # 平均概率分布的熵
    H_mean = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

    # 每次前向概率分布的熵
    H_each = -np.sum(probs_mc * np.log(probs_mc + eps), axis=2)  # [T, n]

    # 熵的期望
    H_expected = H_each.mean(axis=0)

    # 互信息 = 总熵 - 期望熵
    bald_scores = H_mean - H_expected
    return bald_scores


def main():
    """
    主流程函数。

    整体流程：
        Step A. 解析命令行参数
        Step B. 加载特征、标签、Stage1/Stage2 索引
        Step C. （若需要）训练一个线性探头
        Step D. 计算全量 margin 分布，并可自动确定 tau
        Step E. 根据 method 计算不确定性 scores
        Step F. 基于 m_L 和 tau 动态更新 rho
        Step G. 从未标注样本中选出最不确定的 B 个
        Step H. 可选去重
        Step I. 与 Stage1/Stage2 合并，保存结果
        Step J. 打印耗时统计
    """
    start_total = time.perf_counter()

    # ==================================================
    # 1. 参数解析
    # ==================================================
    parser = argparse.ArgumentParser("TriLAF with Auto-Tau + Timing")

    parser.add_argument('--features_inputs', type=str, required=True,
                        help='.npy: features (d cols) + labels (last col)')
    parser.add_argument('--stage1_indices', type=str, required=True,
                        help='Stage1 (core) indices JSON')
    parser.add_argument('--stage2_indices', type=str, required=True,
                        help='Stage2 (boundary) indices JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where to save final JSON')

    parser.add_argument('--method', type=str, required=True,
                        choices=['margin', 'ensemble', 'mcdrop', 'egl', 'bald'],
                        help='Uncertainty metric for Stage3')

    parser.add_argument('--rho', type=float, required=True,
                        help='Initial fraction of U to sample in Stage3')

    parser.add_argument('--tau', type=float, required=False,
                        help='Target minimal margin threshold (if not auto_tau)')
    parser.add_argument('--auto_tau', action='store_true',
                        help='If set, compute tau from specified percentile of full margin distribution')
    parser.add_argument('--tau_percentile', type=float, default=None,
                        help='Which percentile to use as tau when --auto_tau is set')

    parser.add_argument('--decay', type=float, default=0.8,
                        help='Decay factor when m_L > tau')
    parser.add_argument('--grow', type=float, default=1.2,
                        help='Grow factor when m_L ≤ tau')
    parser.add_argument('--min_rho', type=float, default=0.01,
                        help='Lower bound for rho when using clip')
    parser.add_argument('--max_rho', type=float, default=0.5,
                        help='Upper bound for rho when using clip')

    parser.add_argument('--percentiles', type=float, nargs='+', default=[5, 10, 20],
                        help='Percentiles to print from full margin distribution')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs for linear head training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for linear head')

    parser.add_argument('--dedupe_thresh', type=float, default=1e-3,
                        help='Deduplication distance threshold')

    parser.add_argument('--device', type=str,
                        default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Torch device: cuda or cpu')

    parser.add_argument('--use_dynamic', action='store_true',
                        help='Enable dynamic rho update (requires tau)')
    parser.add_argument('--use_dedupe', action='store_true',
                        help='Enable deduplication in Stage3')
    parser.add_argument('--use_percentile', action='store_true',
                        help='Print specified percentiles of full margin distribution')
    parser.add_argument('--use_clip', action='store_true',
                        help='Enable clipping rho to [min_rho, max_rho]')

    parser.add_argument('--num_heads', type=int, default=5,
                        help='Number of heads for ensemble')
    parser.add_argument('--dropout_p', type=float, default=0.5,
                        help='Dropout p for mcdrop / bald')
    parser.add_argument('--mc_iters', type=int, default=10,
                        help='MC-Dropout iterations for mcdrop / bald')

    args = parser.parse_args()

    # ==================================================
    # 2. Load 阶段：读取数据与索引
    # ==================================================
    t0 = time.perf_counter()

    # 读取特征和标签
    data = np.load(args.features_inputs)
    features = data[:, :-1]
    labels = data[:, -1].astype(int)

    total_n = features.shape[0]
    num_classes = int(labels.max()) + 1

    # 读取 Stage1 / Stage2 的索引文件
    with open(args.stage1_indices) as f:
        s1 = json.load(f)
    with open(args.stage2_indices) as f:
        s2 = json.load(f)

    set1, set2 = set(s1), set(s2)

    # L = 已选中的样本集合（Stage1 ∪ Stage2）
    L = sorted(set1 | set2)

    # U = 未被选择的候选样本集合
    U = [i for i in range(total_n) if i not in L]

    # core_feats/core_labels 用于训练探头
    core_feats = features[L]
    core_labels = labels[L]

    device = torch.device(args.device)

    t1 = time.perf_counter()
    load_time = t1 - t0

    # ==================================================
    # 3. Step0：训练基础线性头（如有需要）
    # ==================================================
    t2 = time.perf_counter()

    base_head = None

    # 若 method 依赖 linear head，或需要 auto_tau / percentile，则必须训练
    if args.method in ['margin', 'ensemble', 'egl'] or args.auto_tau or args.use_percentile:
        base_head = train_linear_head(
            core_feats, core_labels, num_classes,
            epochs=args.epochs, lr=args.lr, device=device
        )

    t3 = time.perf_counter()
    step0_time = t3 - t2

    # ==================================================
    # 4. Step1：计算全量 margin / 不确定性分数
    # ==================================================
    t4 = time.perf_counter()

    full_margin = None

    # 若需要自动选择 tau，或打印 margin 分布百分位，则先算全量 single-head margin
    if args.auto_tau or args.use_percentile:
        full_margin = compute_single_margin(base_head, features, device)

        # 打印指定百分位
        if args.use_percentile:
            p_vals = np.percentile(full_margin, args.percentiles)
            pct_str = ', '.join(f'{int(p)}th={v:.4f}' for p, v in zip(args.percentiles, p_vals))
            print(f'Full margin distribution percentiles: {pct_str}')

        # 自动根据某个百分位确定 tau
        if args.auto_tau:
            if args.tau_percentile is None:
                raise ValueError("--auto_tau requires --tau_percentile to be set.")
            tau_val = np.percentile(full_margin, args.tau_percentile)
            args.tau = float(tau_val)
            print(f'Auto-selected tau = {args.tau:.4f} (the {args.tau_percentile}th percentile)')

    # 若未启用 auto_tau，则必须手动给 tau
    if not args.auto_tau and args.tau is None:
        raise ValueError("Either --tau <value> or --auto_tau --tau_percentile <p> must be provided.")

    # --------------------------------------------------
    # 根据 method 计算 scores
    # 分数越高，表示越“不确定”，越值得选入 Stage3
    # --------------------------------------------------
    if args.method == 'margin':
        margin_vals = full_margin if full_margin is not None else compute_single_margin(base_head, features, device)
        scores = -margin_vals  # margin 越小，score 越大

    elif args.method == 'ensemble':
        mean_m, var_m = compute_ensemble_margin(
            core_feats, core_labels,
            features, num_classes,
            args.epochs, args.lr,
            device, args.num_heads
        )
        # 一个简单组合：均值小 + 方差大 => 不确定
        scores = -mean_m + var_m

    elif args.method == 'mcdrop':
        mean_m, var_m, _ = compute_mcdrop_margins_and_probs(
            core_feats, core_labels,
            features, num_classes,
            args.epochs, args.lr,
            device, args.dropout_p,
            args.mc_iters
        )
        scores = -mean_m + var_m

    elif args.method == 'bald':
        _, _, probs_mc = compute_mcdrop_margins_and_probs(
            core_feats, core_labels,
            features, num_classes,
            args.epochs, args.lr,
            device, args.dropout_p,
            args.mc_iters
        )
        scores = compute_bald_scores(probs_mc)

    elif args.method == 'egl':
        scores = compute_egl_scores(base_head, features, device)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    t5 = time.perf_counter()
    step1_time = t5 - t4

    # ==================================================
    # 5. Step2：动态更新 rho + 选样 + 去重 + 合并
    # ==================================================
    t6 = time.perf_counter()

    # 计算当前已标注集合 L 中的最小 margin：m_L
    # 它表示当前已选样本集中“最危险 / 最不确定”的那个 margin
    if base_head is not None:
        labeled_margin = full_margin[L] if full_margin is not None \
            else compute_single_margin(base_head, features, device)[L]
        m_L = float(np.min(labeled_margin))
    else:
        m_L = 0.0

    print(f'Current minimal labeled-set margin m_L = {m_L:.4f}')

    # 初始 rho
    rho = args.rho

    # 若启用动态更新：
    #   - 若 m_L > tau，说明当前已标注集整体较稳，可以减少本轮采样量
    #   - 若 m_L ≤ tau，说明当前边界还不够稳，需要增加采样量
    if args.use_dynamic:
        if m_L > args.tau:
            rho = rho * args.decay
        else:
            rho = rho * args.grow

        # 是否启用范围裁剪
        if args.use_clip:
            rho = float(np.clip(rho, args.min_rho, args.max_rho))

        print(f'Updated rho = {rho:.4f} (tau={args.tau:.4f}, decay={args.decay}, grow={args.grow}, clip={args.use_clip})')
    else:
        print(f'Fixed rho = {rho:.4f} (dynamic disabled)')

    # 计算本轮要选多少个 margin 样本
    B = int(np.ceil(rho * len(U)))
    if B <= 0:
        raise ValueError(f"Computed margin budget B = {B}, must be > 0.")
    print(f'B_margin = {B}')

    # 按 score 从高到低排序，优先选最不确定样本
    ranked = sorted(U, key=lambda i: scores[i], reverse=True)

    selected_margin = []

    # 可选去重逻辑
    if args.use_dedupe:
        for idx in ranked:
            if len(selected_margin) >= B:
                break

            # 与已选样本的特征距离都足够大，才保留
            if all(np.linalg.norm(features[idx] - features[j]) > args.dedupe_thresh
                   for j in selected_margin):
                selected_margin.append(idx)
    else:
        selected_margin = ranked[:B]

    # 最终集合 = Stage1 ∪ Stage2 ∪ 本轮 margin 样本
    final_set = sorted(set(set1) | set(set2) | set(selected_margin))

    t7 = time.perf_counter()
    step2_time = t7 - t6

    # ==================================================
    # 6. Step3：写输出文件
    # ==================================================
    t8 = time.perf_counter()

    os.makedirs(args.output_dir, exist_ok=True)

    N1, N2, M = len(set1), len(set2), len(selected_margin)
    T = len(final_set)

    # 输出文件名中带上集合组成信息，便于管理
    fname = f"Total_{T}_Core{N1}_Bdy{N2}_Mgn{M}_rho{rho:.3f}_{args.method}.json"
    outpath = os.path.join(args.output_dir, fname)

    with open(outpath, 'w') as f_out:
        json.dump(final_set, f_out)

    t9 = time.perf_counter()
    step3_time = t9 - t8

    total_time = t9 - start_total

    # ==================================================
    # 7. 打印耗时统计
    # ==================================================
    print("\n=== Timing Report ===")
    print(f"Load time:    {load_time: .4f} sec")
    print(f"Step0 time:   {step0_time: .4f} sec  # 训练 linear head")
    print(f"Step1 time:   {step1_time: .4f} sec  # 计算 margin/不确定度 分数")
    print(f"Step2 time:   {step2_time: .4f} sec  # 更新 rho + 排序 + 去重 + 合并")
    print(f"Step3 time:   {step3_time: .4f} sec  # 写输出 JSON")
    print(f"Total time:   {total_time: .4f} sec")
    print("=====================\n")

    print(f"Final subset size = {T} (Core={N1}, Boundary={N2}, Margin={M}), saved to:\n  {outpath}")


if __name__ == '__main__':
    """
    程序入口。

    作用：
        直接调用 main()，执行完整的 TriLAF Stage3 选择流程。
    """
    main()
