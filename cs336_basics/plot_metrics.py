import matplotlib.pyplot as plt
from cs336_basics import config
import csv
from pathlib import Path

def plot_metrics(metrics_path,output_path):
    with open(metrics_path,newline = '',encoding = 'utf-8') as f:
        rows = list(csv.DictReader(f))

    steps = [int(row['step']) for row in rows]
    elapsed = [float(row['elapsed_s']) for row in rows]
    train_losses = [float(row['train_loss']) for row in rows]
    val_losses = [float(row['val_loss']) for row in rows]

    # 画图
    fig,axes = plt.subplots(1,2,figsize = (12,4))


    # 4. 左图：loss 随训练步数变化
    axes[0].plot(steps, train_losses, label="Train", marker="o")
    axes[0].plot(steps, val_losses, label="Validation", marker="o")

    axes[0].set_title("Loss vs. training steps")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 5. 右图：loss 随实际耗时变化
    axes[1].plot(elapsed, train_losses, label="Train", marker="o")
    axes[1].plot(elapsed, val_losses, label="Validation", marker="o")

    axes[1].set_title("Loss vs. elapsed time")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 6. 调整布局，避免文字挤在一起
    fig.tight_layout()

    # 7. 创建输出目录并保存图片
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    # 8. 释放这张图占用的资源
    plt.close(fig)

def plot_lr_comparison(metrics_paths, output_path):
    """
    metrics_paths：字典，键为图例名称，值为 CSV 路径。
    output_path：输出图片路径。
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for label, csv_path in metrics_paths.items():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"日志没有数据：{csv_path}")

        # CSV 读出的值是字符串，需要转换为数值。
        steps = [int(row["step"]) for row in rows]
        elapsed = [float(row["elapsed_s"]) for row in rows]
        train_losses = [float(row["train_loss"]) for row in rows]
        val_losses = [float(row["val_loss"]) for row in rows]

        axes[0].plot(steps, train_losses, label=label)
        axes[1].plot(steps, val_losses, label=label)
        axes[2].plot(elapsed, val_losses, label=label)

    titles = [
        "Training loss vs. steps",
        "Validation loss vs. steps",
        "Validation loss vs. elapsed time",
    ]
    xlabels = ["Step", "Step", "Elapsed time (s)"]

    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# 新增：批大小对比单独绘图，保留原来的学习率对比函数。
def plot_batch_comparison(metrics_paths, output_path, context_length):
    """metrics_paths 的键是各组训练 batch size，值是对应的 CSV 路径。

    context_length 必须与这几组日志训练时使用的上下文长度一致。
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for batch_size, csv_path in metrics_paths.items():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"日志没有数据：{csv_path}")

        # 新增：每份日志使用自己的训练 batch size；CSV 的 step 已经加过 1。
        tokens_seen = [
            int(row["step"]) * batch_size * context_length
            for row in rows
        ]
        elapsed = [float(row["elapsed_s"]) for row in rows]
        train_losses = [float(row["train_loss"]) for row in rows]
        val_losses = [float(row["val_loss"]) for row in rows]
        label = f"batch={batch_size}"

        axes[0].plot(tokens_seen, train_losses, label=label)
        axes[1].plot(tokens_seen, val_losses, label=label)
        axes[2].plot(elapsed, val_losses, label=label)

    titles = [
        "Training loss vs. training tokens",
        "Validation loss vs. training tokens",
        "Validation loss vs. elapsed time",
    ]
    xlabels = [
        "Training tokens processed",
        "Training tokens processed",
        "Elapsed time (s)",
    ]

    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_metrics(
        config.METRICS_PATH,
        config.PLOT_PATH,
    )

    # metrics_paths = {
    #     "lr=1e-4": config.ROOT_DIR / "logs" / "ts_lr_1e4.csv",
    #     "lr=3e-4": config.ROOT_DIR / "logs" / "ts_lr_3e4.csv",
    #     "lr=1e-3": config.ROOT_DIR / "logs" / "ts_lr_1e3.csv",
    #     "lr=3e-3": config.ROOT_DIR / "logs" / "ts_lr_3e3.csv",
    #     "lr=1e-2": config.ROOT_DIR / "logs" / "ts_lr_1e2.csv",
    #     "lr=3e-2": config.ROOT_DIR / "logs" / "ts_lr_3e2.csv",
    #     "lr=1e-1": config.ROOT_DIR / "logs" / "ts_lr_1e1.csv",
    # }
    #
    # plot_lr_comparison(
    #     metrics_paths,
    #     config.ROOT_DIR / "plots" / "lr_comparison.png",
    # )

    # 修改：字典键直接保存各组训练 batch size，供新函数计算训练 token 数。
    # metrics_paths_batch = {
    #     1: config.ROOT_DIR / "logs" / "ts_batch_1.csv",
    #     4: config.ROOT_DIR / "logs" / "ts_batch_4.csv",
    #     16: config.ROOT_DIR / "logs" / "ts_batch_16.csv",
    # }
    #
    # # 修改：批大小实验调用新函数，三个实验使用相同的训练上下文长度。
    # plot_batch_comparison(
    #     metrics_paths_batch,
    #     config.ROOT_DIR / "plots" / "batch_comparison.png",
    #     context_length=config.CONTEXT_LENGTH,
    # )

    # metrics_paths = {
    #     "lr=3e-3": config.ROOT_DIR / "logs" / "ts_batch_16.csv",
    #     "lr=1e-3": config.ROOT_DIR / "logs" / "ts_batch_16_lr_1e3.csv",
    #     "lr=1e-2": config.ROOT_DIR / "logs" / "ts_batch_16_lr_1e2.csv",
    # }
    #
    # plot_lr_comparison(
    #     metrics_paths,
    #     config.ROOT_DIR / "plots" / "batch16_lr_comparison.png",
    # )
