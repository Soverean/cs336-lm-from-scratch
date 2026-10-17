import heapq
import pickle
import regex
from mpmath.math2 import INF
from sympy.codegen.ast import continue_
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 根据特殊字符分词
def split_on_special_tokens(text:str,special_tokens:list[str]):

    if len(special_tokens) == 0:
        return [text]

    special_list = []
    for token in special_tokens:
        token = regex.escape(token)
        special_list.append(token)
    special_str = '|'.join(special_list)

    text_list = regex.split(special_str,text)

    return text_list

# 优化5:不再一开始进行utf-8编码，统一最后进行编码
# 预分词
def count_pretokens(text:str,special_tokens:list[str]):
    # 调用特殊token分词，提前剔除特殊token
    fragments = split_on_special_tokens(text,special_tokens)

    counts = {}
    data_count = {}
    # 遍历各个分词，进行预分词
    for frag in fragments:
        # 使用正则化进行预分词:
        for match in regex.finditer(PAT,frag):
            pre_token = match.group()
            counts[pre_token] = counts.get(pre_token,0) + 1


    for key,count in counts.items():
        key_data = key.encode('utf-8')
        key_list = []
        for value_data in key_data:
            key_list.append(bytes([value_data]))
        key_tuple = tuple(key_list)
        data_count[key_tuple] = count

    return data_count


# 统计相邻token对的频次
def count_pairs(pretoken_counts):
    """
    功能:
        将提前 按预分词分好的 tokens列表，统计他们出现次数
    输入:
        预分词:frequency 的 字典
    输出：
        带有 frequency 的 分词列表token_pairs
    """
    token_pairs = {}

    for token,frequency in pretoken_counts.items():

        for i in range(len(token) - 1):
            pair = (token[i],token[i+1])

            token_pairs[pair] = token_pairs.get(pair,0) + frequency

    return token_pairs


# 选择合并对，选择最多的
def select_pair(pair_counts,heap):
    """
    功能:
        根据 tokens列表的出现次数 和 frequency，选取出现次数最高的tokens对，返回元组
    输入:
        token对元组：频率 的 字典
        heap:维护pair_counts的最大堆
    输出:
        出现最多的 token对元组
    优化：
        根据堆优化，来选取最高频次的pair
    """
    while(heap):
        candidate = heapq.heappop(heap)

        current_frequency = pair_counts.get(candidate.pair,0)

        if candidate.frequency != current_frequency:
            continue

        return candidate.pair


# 实际上做合并操作 输入 预分词 和选出的 token对
# 优化版
def merge_pair(pretoken_counts,pair_counts,pair,pair_to_word,heap):
    """
    功能：
        根据输入的 合并对 ，在总tokens表中寻找可以合并对token对
        同时需要维护pair_count和pair_to_word等字典内容
    输入：
        预分词，选出的合并对
        pair_counts: tokens对 计数
        pair_to_word:tokens对 对应 愿tokens序列的 索引字典
    result:
        合并后的词对
    优化：
        原版每次合并，需要计算len、遍历所有count对耗时最大
        1、将len(tokens)计算集中，减少大量len函数
        2、提前删除不包含 当前组合pair中任一元素 的tokens
    """
    left,right = pair
    merged_token = left + right

    affected_word = pair_to_word[pair].copy()
    changes = []

    #循环遍历

    for tokens in affected_word:
        new_tokens = []
        i = 0

        # 优化1，len只进行一次
        n = len(tokens)

        frequency = pretoken_counts[tokens]

        while i < n:
            if(
                i + 1 < n
                and tokens[i] == left
                and tokens[i+1] == right
            ):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        key = tuple(new_tokens)
        # change记录改变了什么，tokens是原分词列表、key是新分词元组，frequency是当前分词频率
        changes.append((tokens,key,frequency))


    # 遍历changes，修改旧序列
    for tokens,key,frequency in changes:
        # 第一步更新 pair_counts，删除旧版本的
        update_pair_counts(pair_counts,tokens,frequency,-1,heap)

        # 第二步修改 字典 pair_to_word，同样删除
        for token_pair in set(zip(tokens,tokens[1:])):
            pair_to_word[token_pair].discard(tokens)

            if not pair_to_word[token_pair]:
                del pair_to_word[token_pair]

        # 第三步修改 pretoken_counts
        pretoken_counts.pop(tokens,None)

    # 遍历changes，加入新序列
    for tokens,key,frequency in changes:
        pretoken_counts[key] = pretoken_counts.get(key,0) + frequency

        update_pair_counts(pair_counts,key,frequency,1,heap)

        for token_pair in set(zip(key,key[1:])):
            pair_to_word.setdefault(token_pair, set()).add(key)



    return pretoken_counts


# 优化3:增量维护pair_counts
def update_pair_counts(pair_counts,tokens,frequency,sign,heap):
    """
    功能:
        维护pair_counts，根据输入的当前tokens，去判断其中是否存在需要合并的 token对，
        根据当前tokens的frequency来增改此token中所有相邻token对的pair_counts
    输入:
        pair_counts：token对计数
        tokens：预分词后的 分词
        frequency：当前预分词的频率
        sign：增删符号
    """
    for pair in zip(tokens,tokens[1:]):
        new_frequency = pair_counts.get(pair,0) + frequency * sign

        if new_frequency > 0:
            pair_counts[pair] = new_frequency
            heapq.heappush(
                heap,
                PairCandidate(new_frequency,pair),
            )
        elif new_frequency == 0:
            pair_counts.pop(pair,None)
        else:
            raise ValueError(f'pair 计数变为负数:{pair}')

# 优化四：快速找出哪些词包含选中的相邻pair
def build_pair_index(pretoken_counts):
    """
    功能:
        创建倒排字典，通过token对索引需要修改的完整token序列，不再进行遍历

    """
    pair_to_word = {}
    for tokens,frequency in pretoken_counts.items():
        for pair in zip(tokens,tokens[1:]):
            pair_to_word.setdefault(pair, set()).add(tokens)

    return pair_to_word


# 优化6:
class PairCandidate:
    def __init__(self,frequency,pair):
        self.frequency = frequency
        self.pair = pair

    def __lt__(self,other):
        # 让频率更大的被认为其实小
        if self.frequency != other.frequency:
            return self.frequency > other.frequency
        # 频率相同时，让字典序更大的被视为更小
        return self.pair > other.pair



# 组装完整的BPE训练函数
def train_bpe(input_path,vocab_size,special_tokens):
    """
        输入：
            初始语料路径、词组大小、特殊token列表
        return：
            没有自动保存到文件中，需要手动保存
            词表（list）、merges（元组list）

    """
    vocab = {i:bytes([i])  for i in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    merges = []

    with open(input_path,'r',encoding = 'utf-8') as f:
        text = f.read()

    # 提前进行预分词
    pretoken_counts = count_pretokens(text, special_tokens)

    # 构建token对 和 预分词的字典
    pair_to_word = build_pair_index(pretoken_counts)

    # 优化3:将统计 token对次数 的函数移出循环外，不再依赖函数多次统计合并后新对排行

    # 计算相邻token对的次数
    pair_counts = count_pairs(pretoken_counts)

    # 优化6:构建最小堆，优化select_pair
    heap = []

    # 由于heap是最小堆，所以取负之后 实现取频率最高的
    for pair,frequency in pair_counts.items():
        heap.append(PairCandidate(frequency,pair))

    heapq.heapify(heap)

    # 每次合并新增一个词表项，因此这是预计需要完成的合并次数
    total_merges = max(0, vocab_size - len(vocab))

    with tqdm(total=total_merges, desc="BPE 合并", unit="次") as progress:
        while(len(vocab) < vocab_size):

            # 选取最大 相邻token对
            max_tuple = select_pair(pair_counts,heap)

            # # 优化3:由于不在循环中使用counts_pairs，需新增函数管理token对排行字典
            # update_pair_counts(pair_counts,max_tuple,frequency,sign)


            # 若max_tuple为None，则要break
            if max_tuple is None:
                break

            merges.append(max_tuple)
            left,right= max_tuple

            vocab[len(vocab)] = left + right

            # 更新预分词结果
            pretoken_counts = merge_pair(pretoken_counts, pair_counts,max_tuple,pair_to_word,heap)
            progress.update(1)


    return vocab,merges













if __name__ == "__main__":
    input_path = ('/Users/soverean/cs336-a1-2025/tests/fixtures/corpus.en')
    vocab_size = 500
    special_tokens = ['<|endoftext|>']
    vocab_filepath = '/Users/soverean/cs336-a1-2025/vocab.pkl'
    merges_filepath = '/Users/soverean/cs336-a1-2025/merges.pkl'
    vocab = {}
    merges = []

    vocab,merges = train_bpe(input_path,vocab_size,special_tokens)

    with open(vocab_filepath,'wb') as f:
        pickle.dump(vocab,f)

    with open(merges_filepath,'wb') as f:
        pickle.dump(merges,f)