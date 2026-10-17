import pickle
from cs336_basics.bpe import train_bpe
from cs336_basics import config
from time import perf_counter
from cs336_basics.tokenizer import Tokenizer

def train_5M_vocab(vocab_path, merges_path, src_path, vocab_size, special_tokens):

    start_time = perf_counter()

    vocab, merges = train_bpe(src_path, vocab_size, special_tokens)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)

    time = perf_counter() - start_time
    print(f'Tokenization took {time} seconds')
    print(len(merges))


if __name__ == '__main__':
    #新方法会先划分训练/测试文本，再使用训练集进行BPE
    out_dir = config.ROOT_DIR / 'artifacts' / 'tinystories_clean'
    out_dir.mkdir(parents = True,exist_ok = True)

    src_path =  config.ROOT_DIR / 'tests' / 'fixtures' / 'tinystories_sample_5M.txt'
    vocab_size = 10000

    with open(src_path,'r',encoding = 'utf-8') as f:
        text = f.read()

    eos = "<|endoftext|>"

    document = text.split(eos)

    documents = [doc for doc in document if doc.strip()]

    cut = int(len(documents) * 0.9)

    train_documents = documents[:cut]
    test_documents = documents[cut:]

    train_text = eos.join(train_documents) + eos
    test_text = eos.join(test_documents) + eos

    train_path = out_dir / 'train.txt'
    test_path = out_dir / 'test.txt'

    train_path.write_text(train_text,encoding = 'utf-8')
    test_path.write_text(test_text,encoding = 'utf-8')


    special_tokens = ['<|endoftext|>']

    # 训练bpe
    train_5M_vocab(config.VOCAB_PATH,config.MERGES_PATH,train_path,vocab_size,special_tokens)

    tokenizer = Tokenizer.from_files(config.VOCAB_PATH,config.MERGES_PATH,special_tokens)
    train_data = tokenizer.encode(train_text)
    print(train_data)
    train_revocer = tokenizer.decode(train_data)
    print(train_revocer)