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
杂项工具函数文件（util.py）

文件作用：
    该文件集中放置训练 / 评估 / 分布式 / 权重加载 / 指标统计 / PCA / 检索评估等常用工具。

大致可以分成以下几类：
    1. 数据增强小工具
    2. 预训练权重加载工具
    3. 训练辅助工具（梯度裁剪、冻结层、scheduler 等）
    4. 日志与统计工具（SmoothedValue / MetricLogger）
    5. 分布式训练工具
    6. 初始化 / 优化器 / 参数分组工具
    7. 检索任务评估工具（mAP / AP）
    8. 多尺度特征提取工具
"""

import os
import sys
import time
import math
import random
import datetime
import subprocess
import argparse
import warnings
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps


# ============================================================
# 1. 数据增强相关小工具
# ============================================================
class GaussianBlur(object):
    """
    对 PIL 图像应用高斯模糊。

    作用：
        以一定概率对图像做 Gaussian Blur，
        常用于自监督学习或强数据增强场景。

    参数：
        p          : 触发该增强的概率
        radius_min : 模糊半径最小值
        radius_max : 模糊半径最大值
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        """
        输入一张 PIL 图像，按概率决定是否执行高斯模糊。
        """
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    对 PIL 图像应用 Solarization（色调反转）增强。

    作用：
        以一定概率对图像做 solarize 处理，
        也是常见的数据增强方式之一。
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        按概率决定是否对图像执行 solarize。
        """
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


# ============================================================
# 2. 预训练权重加载相关
# ============================================================
def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    """
    加载模型的预训练权重。

    逻辑：
        1. 如果用户提供了本地权重文件，则优先加载本地权重
        2. 如果没有提供本地文件，则尝试根据模型类型自动下载 DINO 官方预训练权重
        3. 如果没有对应官方权重，则使用随机初始化权重

    参数：
        model              : 待加载权重的模型
        pretrained_weights : 本地权重路径
        checkpoint_key     : 若 checkpoint 是字典，则从哪个 key 中取权重
        model_name         : 模型名称（如 vit_small / vit_base / resnet50）
        patch_size         : patch 大小，用于匹配对应预训练权重
    """
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        # 去掉多卡训练可能带来的前缀 "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 去掉 multicrop 包装可能带来的前缀 "backbone."
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None

        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"

        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    """
    加载线性分类器（linear classifier）的预训练权重。

    用途：
        常用于线性评估（linear evaluation）场景。
    """
    url = None

    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"

    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")


# ============================================================
# 3. 训练辅助函数
# ============================================================
def clip_gradients(model, clip):
    """
    对模型梯度做裁剪。

    作用：
        避免梯度过大导致训练不稳定。

    参数：
        model : 模型
        clip  : 裁剪阈值

    返回：
        norms : 每个参数梯度范数列表
    """
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())

            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)

    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    """
    在训练早期冻结最后一层的梯度。

    作用：
        某些自监督训练策略中，会在前若干个 epoch 冻结最后一层，
        以提高训练稳定性。

    参数：
        epoch             : 当前 epoch
        model             : 模型
        freeze_last_layer : 冻结到第几个 epoch 为止
    """
    if epoch >= freeze_last_layer:
        return

    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    从 checkpoint 恢复训练状态。

    功能：
        - 加载 checkpoint 文件
        - 根据传入的 key -> object 关系恢复模型/优化器等状态
        - 恢复 run_variables 中记录的训练变量

    参数：
        ckp_path      : checkpoint 路径
        run_variables : 需要恢复的运行变量字典
        **kwargs      : 例如 model=model, optimizer=optimizer
    """
    if not os.path.isfile(ckp_path):
        return

    print("Found checkpoint at {}".format(ckp_path))
    checkpoint = torch.load(ckp_path, map_location="cpu")

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """
    构造 cosine 学习率调度序列。

    功能：
        - 支持 warmup
        - warmup 后按 cosine 规律从 base_value 衰减到 final_value

    返回：
        schedule : 长度为 epochs * niter_per_ep 的 numpy 数组
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep

    return schedule


def bool_flag(s):
    """
    将命令行中的字符串解析为布尔值。

    支持：
        True  : "on", "true", "1"
        False : "off", "false", "0"
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}

    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    固定随机种子，增强实验可复现性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================
# 4. 日志与平滑统计工具
# ============================================================
class SmoothedValue(object):
    """
    用于跟踪一组数值，并提供平滑统计结果。

    常见用途：
        - 记录 loss
        - 记录时间
        - 记录准确率等训练指标

    可提供：
        - median
        - avg
        - global_avg
        - max
        - 当前值 value
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        更新统计值。
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在分布式训练中，同步 count 和 total。

        注意：
            不会同步 deque 中的窗口内容。
        """
        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()

        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        """
        将当前统计结果格式化输出。
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


def reduce_dict(input_dict, average=True):
    """
    在分布式场景下，对字典中的 tensor 值做 all_reduce。

    参数：
        input_dict : 待规约的字典
        average    : True 则取平均，False 则取和

    返回：
        reduced_dict : 规约后的字典
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


class MetricLogger(object):
    """
    指标日志记录器。

    作用：
        - 管理多个 SmoothedValue
        - 统一更新和打印训练过程中的统计信息
        - 支持分布式同步
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        批量更新多个指标。
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        支持通过 logger.loss / logger.acc1 这类方式访问指标。
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        将所有记录指标拼接成可读字符串。
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        同步所有指标。
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        手动添加一个 meter。
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        迭代器包装器：按固定频率打印日志。

        功能：
            - 统计 data_time / iter_time
            - 打印 ETA
            - 打印当前指标
            - 若有 GPU，则额外打印显存占用
        """
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')

        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable))
        )


# ============================================================
# 5. Git 与分布式环境辅助函数
# ============================================================
def get_sha():
    """
    获取当前代码仓库的 git 信息。

    返回内容包括：
        - 当前提交 sha
        - 工作区是否有未提交修改
        - 当前分支名
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'

    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass

    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    """
    判断 torch.distributed 是否可用且已初始化。
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取分布式训练总进程数。
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程序号。
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    判断当前进程是否为主进程。
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    仅在主进程中保存文件。
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    在非主进程中禁用 print，避免多进程重复输出。

    参数：
        is_master : 当前进程是否为主进程
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """
    初始化分布式训练环境。

    支持三种启动方式：
        1. torch.distributed.launch / torchrun
        2. slurm 集群
        3. 单机单卡（若 GPU 可用）

    若没有 GPU，则直接退出。
    """
    # ------------------------------------------------------------
    # 1. torchrun / launch 启动方式
    # ------------------------------------------------------------
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # ------------------------------------------------------------
    # 2. slurm 启动方式
    # ------------------------------------------------------------
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    # ------------------------------------------------------------
    # 3. 单机单卡方式
    # ------------------------------------------------------------
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)

    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    dist.barrier()
    setup_for_distributed(args.rank == 0)


# ============================================================
# 6. 指标与初始化相关函数
# ============================================================
def accuracy(output, target, topk=(1,)):
    """
    计算 Top-k 准确率。

    参数：
        output : 模型输出 logits
        target : 真实标签
        topk   : 需要计算的 top-k 列表，如 (1,) 或 (1,5)

    返回：
        每个 k 对应的准确率列表
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [
        correct[:k].reshape(-1).float().sum(0) * 100. / batch_size
        for k in topk
    ]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    截断正态分布初始化的底层实现。

    说明：
        该实现参考了 PyTorch 官方版本。
    """
    def norm_cdf(x):
        """
        计算标准正态分布的累积分布函数。
        """
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    截断正态分布初始化接口函数。
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================
# 7. 优化器：LARS
# ============================================================
class LARS(torch.optim.Optimizer):
    """
    LARS 优化器。

    说明：
        基本实现参考 Barlow Twins 官方代码。
        常用于大 batch 训练场景。
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        执行一次参数更新。
        """
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                # 对非偏置 / 非一维参数添加权重衰减
                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                # LARS 自适应缩放
                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)

                    q = torch.where(
                        param_norm > 0.,
                        torch.where(update_norm > 0,
                                    (g['eta'] * param_norm / update_norm), one),
                        one
                    )
                    dp = dp.mul(q)

                # 动量项
                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)

                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                # 参数更新
                p.add_(mu, alpha=-g['lr'])


# ============================================================
# 8. 多尺度输入包装器
# ============================================================
class MultiCropWrapper(nn.Module):
    """
    多尺度输入包装器。

    作用：
        对不同分辨率输入分别做前向传播，
        再将它们的输出拼接，最后统一送入 head。

    常见于：
        多 crop 自监督训练（例如 DINO）中。
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()

        # 去掉 backbone 自带分类头，只保留特征提取部分
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """
        前向传播。

        输入：
            x 可以是单个 tensor，也可以是由不同分辨率 crop 组成的 list
        """
        if not isinstance(x, list):
            x = [x]

        # 按输入分辨率分组
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0
        )

        start_idx, output = 0, torch.empty(0).to(x[0].device)

        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))

            # XCiT 模型可能返回 tuple
            if isinstance(_out, tuple):
                _out = _out[0]

            output = torch.cat((output, _out))
            start_idx = end_idx

        return self.head(output)


# ============================================================
# 9. 参数分组与 BatchNorm 判断
# ============================================================
def get_params_groups(model):
    """
    将模型参数分成两组：
        1. regularized     : 参与 weight decay
        2. not_regularized : 不参与 weight decay（如 bias / norm 参数）

    返回：
        可直接传给优化器的参数组列表
    """
    regularized = []
    not_regularized = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    return [
        {'params': regularized},
        {'params': not_regularized, 'weight_decay': 0.}
    ]


def has_batchnorms(model):
    """
    判断模型中是否包含 BatchNorm。
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True

    return False


# ============================================================
# 10. PCA 工具
# ============================================================
class PCA():
    """
    PCA 降维与白化工具类。

    功能：
        - 根据协方差矩阵训练 PCA 变换
        - 对 numpy / torch 数据应用 PCA 变换
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        根据协方差矩阵训练 PCA。

        参数：
            cov : 协方差矩阵（numpy 数组）
        """
        d, v = np.linalg.eigh(cov)

        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()

        if n_0 > 0:
            d[d < eps] = eps

        totenergy = d.sum()

        # 按特征值从大到小排序，并保留前 self.dim 个主成分
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # 白化矩阵
        d = np.diag(1. / d ** self.whit)

        # PCA 投影矩阵
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        """
        对输入数据应用 PCA 变换。

        支持：
            - numpy 数组
            - GPU 上的 torch tensor
            - CPU 上的 torch tensor
        """
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(
                torch.cuda.FloatTensor(self.dvt),
                x.transpose(0, 1)
            ).transpose(0, 1)

        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)

        return torch.mm(
            torch.FloatTensor(self.dvt),
            x.transpose(0, 1)
        ).transpose(0, 1)


# ============================================================
# 11. 检索任务评估：AP / mAP
# ============================================================
def compute_ap(ranks, nres):
    """
    根据正样本排名计算 average precision（AP）。

    参数：
        ranks : 正样本的零基排名位置
        nres  : 正样本总数

    返回：
        ap    : average precision
    """
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    计算检索任务中的 mAP。

    参数：
        ranks  : 检索结果排名，形状为 [db_size, #queries]
        gnd    : ground truth 列表
        kappas : 需要计算 precision@k 的 k 列表

    返回：
        map, aps, pr, prs
    """
    map = 0.
    nq = len(gnd)
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # 若该 query 没有正样本，则跳过
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # 找到正样本和 junk 样本在排序中的位置
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0

        if len(junk):
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # precision @ k
        pos += 1
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq

        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


# ============================================================
# 12. 多尺度特征提取
# ============================================================
def multi_scale(samples, model):
    """
    使用多尺度输入提取特征，并对结果求平均。

    流程：
        1. 将输入图像缩放到多个尺度
        2. 分别送入模型提取特征
        3. 对多尺度特征求平均
        4. 最后做归一化

    参数：
        samples : 输入样本
        model   : 特征提取模型

    返回：
        v : 多尺度融合后的特征向量
    """
    v = None

    for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(
                samples,
                scale_factor=s,
                mode='bilinear',
                align_corners=False
            )

        feats = model(inp).clone()

        if v is None:
            v = feats
        else:
            v += feats

    v /= 3
    v /= v.norm()

    return v
