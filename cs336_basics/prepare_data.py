import numpy as np
import torch
from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.optim import AdamW, get_lr_cosine_schedule,gradient_clipping
from cs336_basics.layers import TransformerLM, cross_entropy
from cs336_basics.data import get_batch
from pathlib import Path
from cs336_basics import config

special_tokens = ['<|endoftext|>']
vocab_filepath = '/Users/soverean/cs336-a1-2025/vocab.pkl'
merges_filepath = '/Users/soverean/cs336-a1-2025/merges.pkl'
corpus_en_filepath = '/Users/soverean/cs336-a1-2025/tests/fixtures/corpus.en'
data_dir = Path('/Users/soverean/cs336-a1-2025/cs336_basics/data')


# 数据预处理，得到token ID，分别得到测试集、训练集
def prepare_data(tokenizer,input_path,data_dir):
    # 处理初始训练语料
    with open(input_path,'r',encoding = 'utf-8') as f:
        text = f.read()
    eos = '<|endoftext|>'
    documents = [doc for doc in text.split(eos) if doc.strip()]
    cut = int(len(documents)*0.9)

    train_text = eos.join(documents[:cut]) + eos
    test_text = eos.join(documents[cut:]) + eos

    train_token_id = tokenizer.encode(train_text)
    test_token_id = tokenizer.encode(test_text)

    train_data = np.asarray(train_token_id,dtype = np.uint16)
    test_data = np.asarray(test_token_id,dtype = np.uint16)

    np.save(data_dir / 'train_data',train_data)
    np.save(data_dir / 'test_data',test_data)

def encode_file(tokenizer,input_path,output_path):
    text = input_path.read_text(encoding = 'utf-8')
    ids = tokenizer.encode(text)
    np_id = np.asarray(ids,dtype = np.uint16)
    np.save(output_path,np_id)




if __name__ == '__main__':
    # tinystories的路径
    src_path =  config.ROOT_DIR / 'tests' / 'fixtures' / 'tinystories_sample_5M.txt'


    # 初始化 tokenizer 分词器 （使用类方法直接定义）
    tokenizer = Tokenizer.from_files(config.VOCAB_PATH,config.MERGES_PATH,special_tokens)

    encode_file(tokenizer,config.TS_DIR / 'train.txt',config.TRAIN_PATH)
    encode_file(tokenizer,config.TS_DIR / 'test.txt',config.TEST_PATH)
