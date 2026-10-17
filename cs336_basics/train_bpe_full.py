import pickle

from cs336_basics import config

from cs336_basics.bpe import train_bpe



input_path = config.ROOT_DIR / "data/tinystories/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

config.VOCAB_TOTAL_PATH.parent.mkdir(parents=True, exist_ok=True)


with open(config.VOCAB_TOTAL_PATH, 'wb')as f:
    pickle.dump(vocab, f)

with open(config.MERGES_TOTAL_PATH, 'wb')as f:
    pickle.dump(merges, f)

print("词表大小：", len(vocab))
print("合并次数：", len(merges))

longest_token = max(vocab.values(), key=len)
print("最长 token：", longest_token)
print("字节长度：", len(longest_token))


