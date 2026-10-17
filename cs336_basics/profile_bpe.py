from cs336_basics import config
from cs336_basics.bpe import train_bpe

input_path = config.TS_DIR / 'train.txt'
vocab_size = 10000
special_tokens = ['<|endoftext|>']


if __name__ == '__main__':
    vocab,merges = train_bpe(config.TRAIN_50M_PATH,vocab_size,special_tokens)

    longest_token = max(vocab.values(),key = len)
    print(longest_token,len(longest_token))

