import torch

from cs336_basics import config
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.layers import TransformerLM, softmax, RMSNorm


# 预测函数
def generate(model,input_ids,max_new_tokens,context_length,temperature,top_p,eos_token_id = None):
    """
        input_ids: [1,T]的整数张量，表示token_id
        max_new_tokens: 最多新增多少个token
        context_length：每次最多给模型多少个tokn，跟模型配置保持一致
        eos_token_id：结束符ID；生成该ID时停止，None表示只按长度停止
    """
    model.eval()
    new_tokens = 0
    with torch.no_grad():
        while new_tokens < max_new_tokens:
            model_input = input_ids[:,-context_length:]
            model_output = model(model_input)
            last_logits = model_output[:,-1,:]

            # 选择并追加token
            next_id = sample_next_token(last_logits,temperature,top_p)
            input_ids = torch.cat([input_ids,next_id],dim = -1)
            new_tokens += 1

            if eos_token_id is not None:
                if next_id == eos_token_id:
                    break
    input_ids_list = input_ids[0].tolist()
    print(f'input_ids_list:{input_ids_list}')
    return input_ids_list

# 温度和top-p采样
def sample_next_token(logits,temperature = 1.0,top_p = 0.5):
    """
        输入(1,vocab_size)的logits,输出(1,1)的token ID

    """

    if temperature <= 0:
        raise ValueError('temperature should be a positive')
    if not 0 < top_p <= 1.0:
        raise ValueError('top_p should be a value between 0 and 1')

    probs = softmax(logits / temperature)
    # 降序得到sort_probs和新张量元素对应原张量的索引表sort_indices
    sort_probs,sort_indices = torch.sort(probs,dim = -1,descending = True)

    total_probs = torch.cumsum(sort_probs,dim = -1)

    zero_column = torch.zeros_like(total_probs[:,:1])
    previous = torch.cat([zero_column,total_probs[:,:-1]],dim = -1)

    if top_p < 1.0:
        result = sort_probs.masked_fill(previous >= top_p,0.0)
    else:
        result = sort_probs

    # 归一化:
    total = result.sum(dim = -1,keepdim = True)
    sampling_probs = result / total

    target = torch.multinomial(
        sampling_probs,
        num_samples = 1,
        replacement = True,
    )

    result_index = sort_indices.gather(dim = -1,index = target)
    return result_index



if __name__ == '__main__':
    special_tokens = ['<|endoftext|>']
    # 初始化 tokenizer 分词器 （使用类方法直接定义）
    tokenizer = Tokenizer.from_files(config.VOCAB_TOTAL_PATH, config.MERGES_TOTAL_PATH, special_tokens)
    vocab_size = len(tokenizer.vocab)

    # 创建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=config.CONTEXT_LENGTH,
        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        rope_theta=config.ROPE_THETA,
        use_rmsnorm=config.USE_RMSNORM,
        device=config.DEVICE,
        dtype=config.DTYPE,
    )
    checkpoint = torch.load(config.CHECKPOINT_PATH,map_location = config.DEVICE)
    model.load_state_dict(checkpoint['model'])

    prompt = "There was a pig named zhu,"

    input_ids = torch.tensor([tokenizer.encode(prompt)],
                             dtype=torch.long,
                             device=config.DEVICE)

    output_id_list = generate(model,input_ids,256,config.CONTEXT_LENGTH,config.TEMPERATURE,config.TOP_P,
                              eos_token_id = tokenizer.word2index[b'<|endoftext|>'])

    output = tokenizer.decode(output_id_list)
    print(output)

