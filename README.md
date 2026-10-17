# Transformer Language Model from Scratch

<!-- 2026-10-17：精简项目首页，保留核心实现、实测结果与运行入口；详细过程见 experiment_log.md。 -->

基于 **Stanford CS336 Spring 2025 Assignment 1** 的个人学习项目，目标是理解 Transformer 各部件、熟悉 PyTorch，并练习性能分析与数据处理。这是一个用于学习的小型语言模型。

使用 PyTorch 张量运算和自动微分，实现从 **BPE 分词、数据编码、模型训练到文本生成** 的完整流程。

## 已实现

| 模块 | 内容 | 代码 |
|---|---|---|
| 分词器 | 字节级 BPE、特殊 token、预分词、编解码与迭代编码 | [bpe.py](cs336_basics/bpe.py)、[tokenizer.py](cs336_basics/tokenizer.py) |
| Transformer | Linear、Embedding、RMSNorm、SwiGLU、RoPE、因果多头注意力与 Pre-norm Block | [layers.py](cs336_basics/layers.py) |
| 训练 | AdamW、warmup / 余弦学习率、梯度裁剪、验证、checkpoint 与指标记录 | [optim.py](cs336_basics/optim.py)、[train.py](cs336_basics/train.py) |
| 数据与生成 | 按文档边界分块编码、`uint16` 存储、内存映射读取、温度 / top-p 采样 | [编码脚本](cs336_basics/encode_train-stream.py)、[generate.py](cs336_basics/generate.py) |

核心层和优化器为自定义实现，未直接使用 `nn.Linear`、`nn.Embedding`、内置多头注意力或内置 AdamW。

## 实验结果

环境：**MacBook Air M5、Python 3.11、PyTorch 2.6.0**；模型训练使用 MPS / FP32。

**模型训练与消融。**基线约 **2.99M 参数**：2 层、`d_model=128`、4 个头、`d_ff=384`、上下文 64、词表 10,000。每轮从完整 TinyStories 训练集随机采样，预算为 `40,000 × 16 × 64 = 40,960,000` 个 token。

| 实验 | 最大学习率 | 最终验证 loss | 最终验证 PPL |
|---|---:|---:|---:|
| 基线：RMSNorm + RoPE + SwiGLU | 1e-2 | **2.1614** | **8.6835** |
| 无 RMSNorm，降低学习率 | 1e-3 | 2.2834 | 9.8100 |
| NoPE，保留 RMSNorm | 1e-2 | 2.2751 | 9.7287 |

无 RMSNorm 在学习率 1e-2 时训练失稳，第 4,000 步中断；降低十倍学习率后完成训练。以上为单次固定种子实验，每次验证固定采样 2,560 个预测位置，并非完整验证集均值；低学习率无 RMSNorm 实验同时改变了学习率。

![基线训练与验证曲线](plots/ts_full_baseline.png)

**BPE 优化。**通过增量 pair 计数、倒排索引、重复预分词去重编码和堆选择减少重复计算。同一约 50 MiB 输入、10,000 词表上，后两项优化将端到端耗时从 **7.62 秒降至 4.56 秒，约 1.67 倍加速**；峰值 RSS 从 385.75 MiB 增至 447.14 MiB。前者为单次记录，后者耗时取三次中位数、内存取三次最大值，均未开启 profiler；不代表全量语料性能。

**数据管线。**将约 **2.23 GB** 训练文本分块编码为 **541,229,347 个 `uint16` token ID**，通过 `np.memmap` 按批次读取，避免一次性构造全量 Python 整数列表。验证集为 5,465,883 个 token，使用同一套分词器。

详细过程见 [实验记录](experiment_log.md)；原始结果见 [基线](logs/ts_full_baseline.csv)、[无 RMSNorm](logs/ts_full_no_rmsnorm_lr1e3.csv)、[NoPE](logs/ts_full_nope.csv)。

## 当前限制与后续计划

- BPE 训练仍整文件读取；分块处理用于语料编码阶段。
- 生成每步重算上下文，尚未实现 KV Cache；计划验证增量前向的一致性并测量速度、缓存占用。
- checkpoint 尚未绑定模型配置、分词器标识和随机数状态，后续补充自动校验与恢复。

## 来源

课程框架、测试和作业说明来自 [Stanford CS336 Spring 2025 Assignment 1](https://github.com/stanford-cs336/assignment1-basics/tree/spring2025)。本仓库是个人学习实现，保留原始 [LICENSE](LICENSE)。
