import regex
import pickle
from collections.abc import Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self,vocab,merges,special_tokens = None):
        super().__init__()

        self.vocab = vocab.copy()
        self.merges = merges

        self.word2index = {token: index for index,token in vocab.items()}

        # 判断是否存在特殊token列表，若有需要提前进行utf-8编码，并且判断对应表中是否存在特token
        if special_tokens is not None:
            self.special_tokens = list(special_tokens)
            for token in special_tokens:
                token_data = token.encode('utf-8')
                if token_data not in self.word2index:
                    new_id = max(self.vocab,default = -1) + 1

                    self.vocab[new_id] = token_data
                    self.word2index[token_data] = new_id

        else:
            self.special_tokens = []


        self.merges_dict = {value: index for index,value in enumerate(self.merges)}



    # 根据特殊token 将初始语料 切分为 语段列表
    def split_text(self,text:str)->list[str]:

        if not self.special_tokens:
            return [text]

        special_tokens_list = []
        # 先匹配更长的特殊 token，避免它被较短的重叠 token 提前拆开。
        for token in sorted(self.special_tokens, key=len, reverse=True):
            token = regex.escape(token)
            special_tokens_list.append(token)

        special_tokens_str = '|'.join(special_tokens_list)
        pattern = "(" + special_tokens_str + ")"
        text_list = regex.split(pattern,text)

        return text_list

    def _apply_merges(self,token:list[bytes])->list[bytes]:
        """
        encode预分词后每个词都转字节utf-8然后调用_apply_merges,
        目的是开始组合相邻tokens
        通过反复遍历，找到按merges 优先级最高 的 预分词中的 相邻token对进行组合
        组合完毕后再返回给 encode 进行查表转换为id
        """
        while(True):
            merge_tokens_id = None
            merge_tokens_priority = len(self.merges_dict)
            for i in range(len(token)-1):
                merge_tokens = tuple(token[i:i+2])
                if merge_tokens in self.merges_dict:
                    priority = self.merges_dict[merge_tokens]
                    if priority < merge_tokens_priority:
                        merge_tokens_id = i
                        merge_tokens_priority = priority

            if merge_tokens_id is None:
                break
            else:
                merge_tokens = token[merge_tokens_id] + token[merge_tokens_id + 1]
                token = token[0:merge_tokens_id] + [merge_tokens] + token[merge_tokens_id+2:]
        return token

    # 编码环节：将输入的语料 编码为 token ID列表
    def encode(self,text:str) -> list[int]:

        # 调用特殊字符切分，将长语料根据特殊字符切分为语段列表
        text_list = self.split_text(text)

        ids = []
        for text in text_list:
            if text in self.special_tokens:
                # 让完整token去找词表
                text_data = text.encode('utf-8')
                text_id = self.word2index[text_data]
                ids.append(text_id)
            else:
                # 正常分词单字节
                for match in regex.finditer(PAT,text):
                    tokens = match.group()
                    tokens_data = tokens.encode('utf-8')
                    tokens_list = []
                    for value in tokens_data:
                        tokens_list.append(bytes([value]))
                    tokens = self._apply_merges(tokens_list)
                    for token in tokens:
                        ids.append(self.word2index[token])


        return ids


    def encode_iterable(self,iterable) ->Iterator[int]:
        """
            实现分批读取文本、分批编码、节省内存

        """
        for text in iterable:
            ids = self.encode(text)

            for token_id in ids:
                yield token_id





    def decode(self,ids:list[int]) -> str:
        data_list = []
        for i in ids:
            token = self.vocab[i]
            data_list.append(token)

        data = b''.join(data_list)
        text = data.decode('utf-8',errors='replace')
        return text

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens = None):
        # 从 vocab_filepath 读取vocab
        with open(vocab_filepath,'rb')as f:
            vocab = pickle.load(f)

        # 从merges_filepath 读取merges
        with open(merges_filepath,'rb')as f:
            merges = pickle.load(f)

        return cls(vocab,merges,special_tokens)





if __name__ == '__main__':
    vocab_filepath = "/Users/soverean/cs336-a1-2025/vocab.pkl"
    merges_filepath = "/Users/soverean/cs336-a1-2025/merges.pkl"

    tokenizer = Tokenizer.from_files(vocab_filepath,
                                    merges_filepath,
                                    special_tokens=['<|endoftext|>'])

    text = '你好,cat!<|endoftext|>'
    ids = tokenizer.encode(text)
    recovered_text = tokenizer.decode(ids)

    print("词表大小：", len(tokenizer.vocab))
    print("合并次数：", len(tokenizer.merges))
    print("编码结果：", ids)
    print("解码结果：", recovered_text)
    print("是否还原成功：", recovered_text == text)
