import pickle
import numpy as np
import torch
from pathlib import Path
from cs336_basics.checkpoint import save_checkpoint,load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.optim import AdamW, get_lr_cosine_schedule,gradient_clipping
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.data import get_batch
from cs336_basics import config
from cs336_basics.metrics import append_metrics
import time
import math
# 新增：使用已有的 tqdm 依赖显示训练进度。
from tqdm.auto import tqdm

# 固定随机数种子
np.random.seed(42)
torch.manual_seed(42)


special_tokens = ['<|endoftext|>']


def train(model,optimizer,
          train_data,test_data,
          num_steps,batch_size,
          context_length,
          resume_path,
          device):
    model.train()

    # 判断是否从头开始训练，调用检查点
    if resume_path is not None:
        start_step = load_checkpoint(resume_path,model,optimizer)
        if start_step > num_steps:
            raise ValueError('检查点已完成步数超过目标训练步数')
    else:
        start_step = 0
        # 清空上一周目的日志 或 重新创建日志：
        config.METRICS_PATH.parent.mkdir(parents = True,exist_ok = True)
        with open(config.METRICS_PATH,'w',encoding = 'utf-8') as f:
            pass

    # 创建字典，用于输出表格、记录相关数据
    metrics = {}

    # 开始计时
    start_time = time.perf_counter()
    # 新增：显示总步数、百分比和剩余时间；续训时从已完成步数开始显示。
    progress = tqdm(
        range(start_step, num_steps),
        total=num_steps,
        initial=start_step,
        desc="训练进度",
        unit="step",
        dynamic_ncols=True,
    )
    for step in progress:


        # 设置当前学习率:
        # 实际上的学习率完全由此处设置，覆盖掉原先交给optimizer保存的学习率
        lr = get_lr_cosine_schedule(step,config.MAX_LR,config.MIN_LR,config.WARMUP_STEPS,num_steps)
        # lr = get_lr_cosine_schedule(step,3e-4,3e-5,config.WARMUP_STEPS,num_steps)


        for group in optimizer.param_groups:
            group['lr'] = lr

        # 每轮随机选取一定训练集和答案
        input, target = get_batch(train_data, batch_size=batch_size,
                                  context_length=context_length,
                                  device=device)
        model_output = model(input)
        loss = cross_entropy(model_output,target)

        # 出现NaN 或 无穷大 时停止，避免继续影响权重
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f'第{step + 1}步 loss 非有限值：{loss.item()}'
            )


        loss.backward()
        gradient_clipping(model.parameters(),config.MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        # 新增：显示当前训练 loss 和学习率，由进度条按刷新间隔更新屏幕。
        progress.set_postfix(
            train_loss=f"{loss.item():.4f}",
            lr=f"{lr:.2e}",
            refresh=False,
        )

        # 统计结束时间
        elapsed_time = time.perf_counter() - start_time

        if (
                step == 0
                or (step + 1) % config.EVAL_STEPS == 0
                or step + 1 == num_steps
        ):
            # 打印数据
            val_loss = evaluate(model, test_data, config.EVAL_BATCH_SIZE, context_length, device,config.EVAL_BATCHES)
            # 修改：使用 tqdm.write 输出验证日志，避免覆盖正在显示的进度条。
            tqdm.write(
                f"step={step + 1} "
                f"train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f}"
            )

            # 保存断点 checkpoint
            save_checkpoint(model, optimizer, step+1, config.CHECKPOINT_PATH)

            try:
                val_ppl = math.exp(val_loss)
            except OverflowError:
                val_ppl = float('inf')

            # 记录表格相关数据
            metrics = {
                'step': step + 1,
                'elapsed_s' : elapsed_time,
                'lr' : lr,
                'train_loss' : loss.item(),
                'val_loss' : val_loss,
                'val_ppl' : val_ppl,
            }
            append_metrics(config.METRICS_PATH,metrics)
            # 修改：保存提示也通过 tqdm.write 输出，保持进度条显示完整。
            tqdm.write('记录已保存')

    # 结束后保存 checkpoint
    save_checkpoint(model, optimizer, num_steps, config.CHECKPOINT_PATH)

def evaluate(model,val_data,batch_size,context_length,device,num_batches = 10):
    # 固定随机状态
    rng_state = np.random.get_state()
    np.random.seed(123)


    model.eval()

    try:
        total_loss = 0
        for batch in range(num_batches):
            input, target = get_batch(val_data, batch_size=batch_size, context_length=context_length, device=device)
            with torch.no_grad():
                loss = cross_entropy(model(input), target)
                total_loss += loss.item()
    finally:
        np.random.set_state(rng_state)
        model.train()


    model.train()
    return total_loss / num_batches



if __name__ == '__main__':
    # 初始化 tokenizer 分词器 （使用类方法直接定义）
    tokenizer = Tokenizer.from_files(config.VOCAB_TOTAL_PATH, config.MERGES_TOTAL_PATH, special_tokens)
    vocab_size = len(tokenizer.vocab)

    # 创建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=config.CONTEXT_LENGTH,
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        rope_theta=config.ROPE_THETA,
        use_rmsnorm=config.USE_RMSNORM,
        device=config.DEVICE,
        dtype=config.DTYPE,
    )

    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=config.MAX_LR,
                      betas=config.BETAS, eps=config.EPS,
                      weight_decay=config.WEIGHT_DECAY)

    train_data = np.memmap(config.TRAIN_TOTAL_PATH, dtype=np.uint16, mode="r")
    valid_data = np.load(config.VALID_TOTAL_PATH, mmap_mode="r")


    train(
        model,
        optimizer,
        train_data,
        valid_data,
        config.NUM_STEPS,
        config.BATCH_SIZE,
        config.CONTEXT_LENGTH,
        config.RESUME_PATH,
        config.DEVICE
    )


