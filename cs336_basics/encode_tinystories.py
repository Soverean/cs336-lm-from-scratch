import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics import config

tokenizer = Tokenizer.from_files(
    config.VOCAB_TOTAL_PATH,
    config.MERGES_TOTAL_PATH,
    special_tokens = ['<|endoftext|>']
)

input_path = config.ROOT_DIR / "data/tinystories/TinyStoriesV2-GPT4-valid.txt"

# 1.读取文件：input_path 是地址，text是文件中的字符串
with open(input_path,'r',encoding = 'utf-8') as f:
    text = f.read()

# 2.编码并计时：把返回的token ID保存到ids
start_time = time.perf_counter()
ids = tokenizer.encode(text)
total_time = time.perf_counter() - start_time

# 3.统计数据：这里统一
byte_count = len(text.encode("utf-8"))
bytes_per_token = byte_count / len(ids)
throughput = byte_count / total_time / (1024 * 1024)  # MiB/s

print("token 数量：", len(ids))
print(f"编码耗时：{total_time:.2f} 秒")
print(f"平均每个 token 对应字节数：{bytes_per_token:.2f}")
print(f"编码吞吐量：{throughput:.2f} MiB/s")

# 4. 验证编码结果：可以还原原文，且所有 ID 都在词表范围内。
assert tokenizer.decode(ids) == text, "解码后与原文不一致"
assert min(ids) >= 0 and max(ids) < len(tokenizer.vocab), "ID 超出词表范围"
print("验证通过")

# 5. 转成 NumPy 数组并保存，供模型训练时读取。
data = np.asarray(ids, dtype=np.uint16)

output_path = config.ROOT_DIR / "artifacts/tinystories/valid.npy"
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, data)

print("保存位置：", output_path)
print("数组形状：", data.shape)
print("数组类型：", data.dtype)
