import argparse
import csv
import json
from pathlib import Path

import torch
import torch.serialization

# PyTorch 2.6+ 默认 weights_only=True。
# 如果 checkpoint 中保存了 argparse.Namespace（例如训练时把 args 一起保存），
# 直接 torch.load 可能会被安全机制拦截。
# 这里把 argparse.Namespace 加入 allowlist，保证此类 checkpoint 可以正常加载。
torch.serialization.add_safe_globals([argparse.Namespace])

from torch.utils.data import DataLoader, SequentialSampler

from datasets import build_dataset
from models import DistilledVisionTransformer


def try_get_image_path(dataset, idx):
    """
    尝试从数据集中获取指定样本的原始图像路径。

    作用：
        由于不同数据集/包装器的内部结构不完全一致，
        这个函数做一个“兼容性获取”：
        - 如果是 torchvision 常见数据集（如 ImageFolder），优先从 .samples / .imgs 中取路径
        - 如果外层是自定义包装器，就递归进入其 .dataset
        - 如果最终还是拿不到路径，就退化为返回字符串 "index:编号"

    参数：
        dataset : 数据集对象
        idx     : 样本索引

    返回：
        str : 图像路径；若无法获取，则返回类似 "index:123"
    """
    # ------------------------------------------------------------
    # 1. 优先尝试 torchvision 常见字段：samples / imgs
    # ------------------------------------------------------------
    for attr in ["samples", "imgs"]:
        if hasattr(dataset, attr):
            items = getattr(dataset, attr)
            if items and isinstance(items[idx], (tuple, list)) and len(items[idx]) >= 1:
                return str(items[idx][0])

    # ------------------------------------------------------------
    # 2. 如果当前 dataset 是包装器，则递归进入底层 dataset
    # ------------------------------------------------------------
    if hasattr(dataset, "dataset"):
        return try_get_image_path(dataset.dataset, idx)

    # ------------------------------------------------------------
    # 3. 实在拿不到路径时，返回索引占位符
    # ------------------------------------------------------------
    return f"index:{idx}"


@torch.inference_mode()
def main():
    """
    主函数：使用训练好的 DeiT / ViT 模型对整个数据集做批量预测，并保存结果。

    整体流程：
        1. 解析命令行参数
        2. 构建待预测数据集和 DataLoader
        3. 构建模型结构
        4. 加载 checkpoint 权重
        5. 对所有样本进行前向预测
        6. 将结果保存为 CSV 或 JSONL 文件

    输出结果中默认包含：
        - image_path : 图像路径
        - pred_idx   : 预测类别编号
        - pred_label : 预测类别名称（若数据集提供 classes）
        - pred_prob  : 预测类别概率（可选）
    """
    # ============================================================
    # 一、命令行参数定义
    # ============================================================
    parser = argparse.ArgumentParser("Predict all images with a trained DeiT/ViT model")

    # -----------------------------
    # 1. 数据与模型相关参数
    # -----------------------------
    parser.add_argument("--data-path", type=str, required=True,
                        help="数据集根目录（需与训练时保持一致）")
    parser.add_argument(
        "--data-set",
        type=str,
        default="IMNETSUBSET",
        choices=[
            "CIFAR10", "CIFAR100", "IMNET", "INAT", "INAT19",
            "CIFAR10SUBSET", "CIFAR100SUBSET", "IMNETSUBSET"
        ],
        help="数据集类型（应与训练时保持一致）",
    )
    parser.add_argument("--input-size", type=int, default=224,
                        help="输入图像尺寸")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="预测时 batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 的 worker 数")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备，如 cuda 或 cpu")

    # -----------------------------
    # 2. 权重相关参数
    # -----------------------------
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint 路径（.pth），支持 {'model':...} 或直接 state_dict")
    parser.add_argument("--use-ema", action="store_true",
                        help="若 checkpoint 中包含 model_ema，则优先加载 EMA 权重")

    # -----------------------------
    # 3. 输出相关参数
    # -----------------------------
    parser.add_argument("--out", type=str, default="predictions.csv",
                        help="输出文件路径，支持 .csv 或 .jsonl")
    parser.add_argument("--save-prob", action="store_true",
                        help="是否同时保存预测类别的概率")

    args = parser.parse_args()

    # ============================================================
    # 二、选择运行设备
    # ============================================================
    # 如果用户要求 cuda，但当前环境不可用，则自动退回 cpu
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # ============================================================
    # 三、构建待预测数据集
    # ============================================================
    # 这里默认使用 is_train=False，与验证/测试阶段一致
    dataset, nb_classes = build_dataset(is_train=False, args=args)

    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),  # 按固定顺序遍历，保证输出顺序稳定
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ============================================================
    # 四、构建模型
    # ============================================================
    # 这里直接采用与你训练脚本中一致的 DistilledVisionTransformer 配置
    model = DistilledVisionTransformer(
        img_size=[args.input_size],
        patch_size=16,
        in_chans=3,
        num_classes=nb_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=torch.nn.LayerNorm,
    ).to(device)

    model.eval()

    # ============================================================
    # 五、加载 checkpoint 权重
    # ============================================================
    ckpt = None

    # 兼容不同版本 PyTorch 的 torch.load 行为：
    # - 新版本可传 weights_only=True
    # - 老版本可能不支持该参数
    # - 若仍失败，则回退到 weights_only=False（仅适用于可信来源）
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        # 老版本 PyTorch 没有 weights_only 参数
        ckpt = torch.load(args.ckpt, map_location="cpu")
    except Exception as e:
        print("⚠️ weights_only=True 加载失败，尝试使用 weights_only=False（仅在你信任 ckpt 来源时安全）：", repr(e))
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # ------------------------------------------------------------
    # 支持三种常见 checkpoint 格式：
    # 1. {'model': ...}
    # 2. {'model_ema': ...}
    # 3. 直接就是 state_dict
    # ------------------------------------------------------------
    if args.use_ema and isinstance(ckpt, dict) and "model_ema" in ckpt:
        state = ckpt["model_ema"]

        # 有些 EMA 结果可能再包一层 state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]

    else:
        state = ckpt

    # strict=False：允许部分 key 对不上，便于兼容不同 checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print("⚠️ Missing keys (ignored):", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("⚠️ Unexpected keys (ignored):", unexpected[:10], "..." if len(unexpected) > 10 else "")

    # ============================================================
    # 六、准备输出文件
    # ============================================================
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_jsonl = out_path.suffix.lower() == ".jsonl"
    is_csv = out_path.suffix.lower() == ".csv"

    if not (is_jsonl or is_csv):
        raise ValueError("out file must end with .csv or .jsonl")

    # 尝试获取类别名称（若数据集带有 classes 属性）
    class_names = getattr(dataset, "classes", None)

    def pred_to_label(pred_idx: int):
        """
        将预测类别编号转换为类别名称。

        若数据集提供了 class_names，则返回对应类别名；
        否则退化为返回数字字符串。
        """
        if class_names and 0 <= pred_idx < len(class_names):
            return class_names[pred_idx]
        return str(pred_idx)

    total = len(dataset)
    seen = 0

    # ============================================================
    # 七、批量预测并保存结果
    # ============================================================
    if is_csv:
        # --------------------------------------------------------
        # 1. 保存为 CSV 格式
        # --------------------------------------------------------
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 写入表头
            header = ["image_path", "pred_idx", "pred_label"]
            if args.save_prob:
                header.append("pred_prob")
            writer.writerow(header)

            # 逐 batch 推理
            for batch_idx, batch in enumerate(loader):
                # 最常见情况：batch = (images, targets)
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.to(device, non_blocking=True)

                # 前向得到 logits
                logits = model(images)

                # softmax 转概率
                probs = torch.softmax(logits, dim=-1)

                # 取最大概率及其类别编号
                pred_prob, pred_idx = torch.max(probs, dim=-1)

                bs = pred_idx.shape[0]
                for i in range(bs):
                    ds_idx = seen + i
                    img_path = try_get_image_path(dataset, ds_idx)
                    pidx = int(pred_idx[i].item())

                    row = [img_path, pidx, pred_to_label(pidx)]
                    if args.save_prob:
                        row.append(float(pred_prob[i].item()))

                    writer.writerow(row)

                seen += bs

                # 定期打印进度
                if seen % max(1, min(2000, total)) == 0:
                    print(f"Progress: {seen}/{total}")

    else:
        # --------------------------------------------------------
        # 2. 保存为 JSONL 格式
        # --------------------------------------------------------
        with out_path.open("w", encoding="utf-8") as f:
            for batch_idx, batch in enumerate(loader):
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.to(device, non_blocking=True)

                logits = model(images)
                probs = torch.softmax(logits, dim=-1)
                pred_prob, pred_idx = torch.max(probs, dim=-1)

                bs = pred_idx.shape[0]
                for i in range(bs):
                    ds_idx = seen + i
                    img_path = try_get_image_path(dataset, ds_idx)
                    pidx = int(pred_idx[i].item())

                    obj = {
                        "image_path": img_path,
                        "pred_idx": pidx,
                        "pred_label": pred_to_label(pidx),
                    }

                    if args.save_prob:
                        obj["pred_prob"] = float(pred_prob[i].item())

                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

                seen += bs

                if seen % max(1, min(2000, total)) == 0:
                    print(f"Progress: {seen}/{total}")

    # ============================================================
    # 八、输出完成提示
    # ============================================================
    print(f"✅ Saved predictions to: {out_path.resolve()}")


if __name__ == "__main__":
    """
    程序入口。

    作用：
        直接调用 main()，完成：
        参数解析 → 数据集构建 → 模型加载 → 批量预测 → 结果保存
    """
    main()
