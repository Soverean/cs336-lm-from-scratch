from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent

# 存储词表 和 组合顺序 的路径
# VOCAB_PATH = ROOT_DIR / 'vocab.pkl'
# MERGES_PATH = ROOT_DIR / 'merges.pkl'
# TRAIN_PATH = ROOT_DIR / 'cs336_basics' / 'data' / 'train_data.npy'
# TEST_PATH = ROOT_DIR / 'cs336_basics' / 'data' / 'test_data.npy'
# CHECKPOINT_PATH = ROOT_DIR / 'checkpoint_small.pt'

# 小词组 词表 和 组合顺序 的路径
# TINY_VOCAB_PATH = ROOT_DIR / 'artifacts' / 'tinystories_pilot' / 'vocab.pkl'
# TINY_MERGES_PATH = ROOT_DIR / 'artifacts' / 'tinystories_pilot' / 'merges.pkl'

# 统一tiny路径
TS_DIR = ROOT_DIR / 'artifacts' / 'tinystories_clean'

VOCAB_PATH =  TS_DIR / 'vocab.pkl'
MERGES_PATH = TS_DIR / 'merges.pkl'
TRAIN_PATH = TS_DIR / 'train_data.npy'
TEST_PATH = TS_DIR / 'test_data.npy'

# 检查点路径
CHECKPOINT_PATH = ROOT_DIR / "artifacts/tinystories/ts_full_nope.pt"
# csv数据路径
METRICS_PATH = ROOT_DIR / "logs/ts_full_nope.csv"
# 图像路径
PLOT_PATH = ROOT_DIR / "plots/ts_full_nope.png"

# 小语料路径
CORPUS_EN_PATH = ROOT_DIR /'tests' / 'fixtures' / 'corpus.en'

# 50M语料路径
TS_50_PATH = ROOT_DIR / 'artifacts' / 'bpe_scale'
TRAIN_50M_PATH = ROOT_DIR / 'data' / 'tinystories' / 'train_50MiB.txt'

# 完整语料样本
TOTAL_PATH = ROOT_DIR / 'data' / 'tinystories' / 'TinyStoriesV2-GPT4-train.txt'

VOCAB_TOTAL_PATH = ROOT_DIR / 'artifacts' / 'tinystories' / 'vocab.pkl'
MERGES_TOTAL_PATH = ROOT_DIR / 'artifacts' / 'tinystories' / 'merges.pkl'
TRAIN_TOTAL_PATH = ROOT_DIR / 'artifacts' / 'tinystories' / 'train.bin'
VALID_TOTAL_PATH = ROOT_DIR / 'artifacts' / 'tinystories' / 'valid.npy'

# 可调整为 checkpoint
RESUME_PATH = None

# 初始化各个初始配置
CONTEXT_LENGTH = 64
BATCH_SIZE = 16

D_MODEL = 128
NUM_LAYERS = 2
NUM_HEADS = 4
D_FF = 384
DEVICE = 'mps'
DTYPE = torch.float32

# 训练轮数200轮
NUM_STEPS = 40000
WARMUP_STEPS = 4000
# 每20步验证一次
EVAL_STEPS = 1000


MAX_LR = 1e-2
MIN_LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

# 每次验证抽取多少个batch
EVAL_BATCHES = 10
# 每个验证batch有多少个序列
EVAL_BATCH_SIZE = 4


# 温度和top_p
TEMPERATURE = 0.8
TOP_P = 0.9

# 消融实验
USE_RMSNORM = True
ROPE_THETA = None


