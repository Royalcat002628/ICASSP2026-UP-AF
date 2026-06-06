# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
工具函数文件（utils.py）

文件作用：
    该文件主要提供两类工具：

    1. 日志与指标统计工具
       - SmoothedValue
       - MetricLogger

    2. 分布式训练辅助工具
       - 分布式环境初始化
       - 仅主进程保存文件
       - 控制非主进程不打印日志
       - 加载 EMA checkpoint 的辅助函数

整体来看，这个文件通常会被训练主流程（如 main.py、engine.py）调用，
本身不直接承担训练逻辑，而是作为底层支持模块存在。
"""

import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


# ============================================================
# 1. 平滑统计类：SmoothedValue
# ============================================================
class SmoothedValue(object):
    """
    用于跟踪一组数值，并提供平滑后的统计结果。

    作用：
        在训练过程中，某些指标（如 loss、iter time、data time）会不断变化，
        直接打印瞬时值不够稳定，因此这里通过一个滑动窗口来记录最近若干次的值，
        并提供如下统计量：

        - median      : 窗口内中位数
        - avg         : 窗口内平均值
        - global_avg  : 从开始累计到现在的全局平均值
        - max         : 窗口内最大值
        - value       : 当前最新值

    常见用途：
        - 训练 loss 统计
        - 数据加载耗时统计
        - 迭代耗时统计
        - 准确率等指标统计
    """

    def __init__(self, window_size=20, fmt=None):
        """
        初始化平滑统计对象。

        参数：
            window_size : 滑动窗口大小，默认记录最近 20 个值
            fmt         : 输出格式字符串
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        更新统计值。

        参数：
            value : 新加入的数值
            n     : 该数值对应的样本数 / 次数权重
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在分布式训练中，同步 total 和 count。

        注意：
            这里只同步累计统计量，不同步 deque 中窗口内的具体历史值。
            因此同步后的 global_avg 是跨进程一致的，但 median / avg 仍是本地窗口统计。
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
        """
        返回滑动窗口内的中位数。
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        返回滑动窗口内的平均值。
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        返回从开始累计到现在的全局平均值。
        """
        return self.total / self.count

    @property
    def max(self):
        """
        返回滑动窗口内的最大值。
        """
        return max(self.deque)

    @property
    def value(self):
        """
        返回当前最新值。
        """
        return self.deque[-1]

    def __str__(self):
        """
        将统计值格式化为字符串，便于日志打印。
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


# ============================================================
# 2. 指标日志记录器：MetricLogger
# ============================================================
class MetricLogger(object):
    """
    指标日志记录器。

    作用：
        统一管理多个 SmoothedValue 对象，并提供：
        - update(): 更新指标
        - __str__(): 转成可打印字符串
        - synchronize_between_processes(): 分布式同步
        - log_every(): 包装 iterable，定期打印训练日志

    常见场景：
        在 train_one_epoch / evaluate 中记录：
        - loss
        - lr
        - acc1
        - acc5
        - iter time
        - data time
    """

    def __init__(self, delimiter="\t"):
        """
        初始化日志记录器。

        参数：
            delimiter : 多个指标之间的分隔符
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        批量更新多个指标。

        用法示例：
            logger.update(loss=0.23, lr=1e-4)

        说明：
            若传入的是 tensor，会自动转成标量。
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """
        支持通过 logger.loss / logger.acc1 直接访问对应指标对象。
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        将所有指标拼接成可打印字符串。
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        同步所有指标对象。
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        手动添加一个指标对象。

        参数：
            name  : 指标名称
            meter : SmoothedValue 实例
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        包装一个可迭代对象，并按照固定频率打印日志。

        功能：
            - 统计 data_time（读取数据耗时）
            - 统计 iter_time（单轮迭代耗时）
            - 估计剩余时间 ETA
            - 打印当前各项指标
            - 若有 CUDA，则额外打印显存占用

        参数：
            iterable   : 可迭代对象，如 DataLoader
            print_freq : 每隔多少步打印一次
            header     : 日志前缀，如 "Epoch: [1]" 或 "Val:"
        """
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # 根据 iterable 长度动态设置步数显示宽度
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            # 统计数据读取时间
            data_time.update(time.time() - end)

            yield obj

            # 统计本轮总耗时
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable))
        )


# ============================================================
# 3. EMA checkpoint 加载辅助函数
# ============================================================
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    为 ModelEma 提供 checkpoint 加载兼容函数。

    作用：
        某些情况下，ModelEma 的内部加载函数希望接收“文件对象”而不是已经加载好的 Python 对象。
        这里通过 BytesIO 在内存中构造一个临时文件，绕过这个限制。

    参数：
        model_ema   : timm.utils.ModelEma 实例
        checkpoint  : 已经加载到内存中的 checkpoint 对象
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


# ============================================================
# 4. 分布式打印控制
# ============================================================
def setup_for_distributed(is_master):
    """
    在分布式训练中控制打印行为。

    作用：
        只有主进程（master process）默认允许打印，
        其他进程的 print 会被静默掉，避免多卡训练时日志重复刷屏。

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


# ============================================================
# 5. 分布式环境状态查询
# ============================================================
def is_dist_avail_and_initialized():
    """
    判断 torch.distributed 是否可用且已经初始化。
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    获取分布式训练中的总进程数。

    若未启用分布式，则返回 1。
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    获取当前进程的 rank。

    若未启用分布式，则返回 0。
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
    仅在主进程中执行 torch.save。

    作用：
        避免多卡训练时每个进程都重复保存同一个 checkpoint。
    """
    if is_main_process():
        torch.save(*args, **kwargs)


# ============================================================
# 6. 分布式训练初始化
# ============================================================
def init_distributed_mode(args):
    """
    初始化分布式训练环境。

    支持两种常见启动方式：
        1. torchrun / torch.distributed.launch
           通过环境变量 RANK / WORLD_SIZE / LOCAL_RANK 获取信息
        2. SLURM 集群
           通过环境变量 SLURM_PROCID 获取 rank

    若检测不到分布式环境变量，则默认关闭分布式模式。

    参数：
        args : 命令行参数对象，会在函数内补充/修改以下字段：
               - rank
               - world_size
               - gpu
               - distributed
               - dist_backend
    """
    # ------------------------------------------------------------
    # 方式 1：torchrun / torch.distributed.launch
    # ------------------------------------------------------------
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # ------------------------------------------------------------
    # 方式 2：SLURM 集群
    # ------------------------------------------------------------
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    # ------------------------------------------------------------
    # 未检测到分布式环境
    # ------------------------------------------------------------
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(args.gpu)

    # 指定分布式后端
    args.dist_backend = 'nccl'

    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    # 初始化进程组
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    # 等待所有进程同步
    torch.distributed.barrier()

    # 设置非主进程不打印
    setup_for_distributed(args.rank == 0)
