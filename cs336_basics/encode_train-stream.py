import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics import config

def iter_text_chunks(input_path,tokenizer,chunk_size = 1024*1024):
    """
    功能：
        把train.txt文件提取，一次提取部分，防止内存溢出等问题
        然后转换为np数组存储

        实用buffer记录每次需要yield的值

    """
    buffer = ''
    eos = '<|endoftext|>'

    with open(input_path,'r',encoding = 'utf-8') as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            buffer += chunk
            position = buffer.rfind(eos)

            if position == -1:
                continue

            position = position + len(eos)
            text_to_data = buffer[:position]
            buffer = buffer[position:]

            ids = tokenizer.encode(text_to_data)

            yield np.asarray(ids,dtype = np.uint16)

        if buffer:
            ids = tokenizer.encode(buffer)
            yield np.asarray(ids,dtype = np.uint16)



if __name__ == '__main__':

    tokenizer = Tokenizer.from_files(config.VOCAB_TOTAL_PATH,
                          config.MERGES_TOTAL_PATH,
                          special_tokens= ['<|endoftext|>'])

    with open(config.TRAIN_TOTAL_PATH,'wb') as output_file:
        for data in iter_text_chunks(config.TOTAL_PATH,tokenizer):
            data.tofile(output_file)