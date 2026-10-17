import math
import torch
from torch import nn

# 创建 激活函数：SiLU
def silu(x:torch.Tensor)->torch.Tensor:
    denom = torch.sigmoid(x)
    return x * denom


# 新建层归一化，规范输入输出
class RMSNorm(nn.Module):
    def __init__(self,d_model,eps = 1e-5,device =None,dtype = None):
        super().__init__()
        self.eps = eps
        # dtype 指定权重的存储类型；输出类型由每次传入的 x 决定。
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))


    def forward(self, x):
        # 在转换前记录输入类型，供归一化计算结束后恢复。
        original_dtype = x.dtype
        x = x.to(torch.float32)
        x_2 = x*x
        # average.shape: [batch_size,seq_len,1]
        average = x_2.mean(dim = -1,keepdim = True)

        denom = torch.sqrt(average + self.eps)

        result = x / denom * self.weight
        # 恢复本次输入的类型，而不是构造函数中指定的权重类型。
        result = result.to(dtype = original_dtype)

        return result


# 新建嵌入层
class Embedding(nn.Module):
    # num_embeddings: 词数  embedding_dim:词向量维度
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))

        nn.init.trunc_normal_(self.weight,
                              mean = 0.0,
                              std = 1.0,
                              a = -3.0,
                              b = 3.0)

    def forward(self,token_ids):
        return self.weight[token_ids]


# 构建基础线性层
class Linear(nn.Module):
    def __init__(self,in_features,out_features,device = None,dtype = None,):
        super().__init__()

        # 创建并保存权重
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features,in_features,device=device,dtype=dtype,)

        # 将权重注册为模型的可学习参数
        self.weight = nn.Parameter(self.weight)

        std = math.sqrt(2 / (in_features + out_features))

        # 初始化权重
        nn.init.trunc_normal_(self.weight,
                              mean = 0.0,
                              std = std,
                              a = -3 * std,
                              b = 3 * std)


    def forward(self,x):
        # 使用权重对x进行线性变换
        result = x @ self.weight.T
        return result

# 带门控的前馈网络，其中有输出函数SiLU
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device = None,dtype = None):
        super().__init__()

        self.linear_w1 = Linear(d_model,d_ff,
                                device = device,
                                dtype = dtype)

        self.linear_w2 = Linear(d_ff,d_model,
                                device = device,
                                dtype = dtype)

        self.linear_w3 = Linear(d_model,d_ff,
                                device = device,
                                dtype = dtype)


    def forward(self,x):
        w1 = self.linear_w1(x)
        w1_silu = silu(w1)
        w3 = self.linear_w3(x)

        w1_w3 = w1_silu * w3

        w2 = self.linear_w2(w1_w3)

        return w2

# softmax激活函数
def softmax(x:torch.Tensor,dim:int = -1)->torch.Tensor:
    maximum = torch.amax(x,dim = dim,keepdim = True)

    shifted = x - maximum
    exp_values = torch.exp(shifted)

    sum_exp = torch.sum(exp_values,dim = dim,keepdim = True)

    probabilities = exp_values / sum_exp

    return probabilities


# 缩放点积注意力
def scaled_dot_product_attention(Q,K,V,mask = None):

    # K和Q的最后一维维度相同，根据矩阵运算，把最后一维和倒数二维交换，即可相乘
    # Q:[2,3,4] K:[2,5,4] K_T:[2,4,5]
    K_T = K.transpose(-2,-1)
    # Q_K:[2,3,5]
    Q_K = Q @ K_T

    d_model = Q.shape[-1]
    d_model_tensor = torch.tensor(d_model)

    denom = torch.sqrt(d_model_tensor)

    # probabilities.shape: [2,3,5]
    scoers = Q_K / denom
    if mask is not None:
        scoers = scoers.masked_fill(~mask,float('-inf'))

    attention_weights = softmax(scoers,dim = -1)

    result = attention_weights @ V

    return result


# 旋转位置编码，定义一个类
class RotaryPositionalEmbedding(nn.Module):
    """
    功能：
        定义旋转位置编码，模仿时钟时针分针秒针的逻辑，对于一个token的词向量
        旋转角度根据向量（最后一维的）维度决定：高维度每次的旋转较慢，低纬度较快，以此来区分不同token之间的位置关系
        旋转次数根据token位置决定，处于句子开始的不旋转，每向后一个位置就旋转一次
    参数：
        创建时需输入：
            参数维度（后分为多头注意力时，可输入每个头的维度）
            句子最大长度，为了算出cos，sin表（根据位置计算旋转次数），参数要放入buffer
            设置device，需要把维度下标送入GPU
        forward:
            x：需要旋转的向量
            token_position：该向量的位置参数（形状跟x类似，没有最后一维）
        return：
            旋转后的向量

    """
    def __init__(self,theta,d_k,max_seq_len,device = None):
        super().__init__()

        # 计算并保存sin、cos

        # 计算维度下标[0，2，4，6]
        dim_indices = torch.arange(0,d_k,2,
                                   device = device,
                                   dtype = torch.float32)

        # exponents随着维度越靠后，越接近1，加了负号就是越靠后越接近 -2
        exponents = dim_indices / d_k

        # 频率从1 一直到万分之一
        frequencies = theta ** (-exponents)

        # 计算每个token位置的旋转角度
        positions = torch.arange(max_seq_len,
                                  device = device,
                                  dtype = torch.float32)
        angels = torch.outer(positions,frequencies)

        cos_values = torch.cos(angels)
        sin_values = torch.sin(angels)

        self.register_buffer('cos_cached',cos_values,persistent = False)
        self.register_buffer('sin_cached',sin_values,persistent = False)





    def forward(self,x,token_positions):
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        output = torch.empty_like(x)

        output[...,0::2] = rotated_even
        output[...,1::2] = rotated_odd

        return output


# 多头自注意力（包含了RoPE）
class MultiHeadSelfAttention(nn.Module):
    """
        类：多头自注意力
        功能：
            将输入的向量，通过维度高低分给多个头，分别进行注意力机制qkv计算
            计算完后返回同样维度的向量
        参数：
            构建时：
                模型维度：向量的维度，用于调用旋转位置编码、计算头维度等
                头数：分给多个头，后续还需调整向量形状，以头、token、维度的顺序运算
                device：设备
                dtype
                theta：旋转角度，需要传给RoPE，决定基础旋转大小
                max_seq_len：最大token个数 = 最大句子长度。传给RoPE，方便计算
            forward：
                x：需要运算的张量
                token_position：位置信息，跟x的形状类似，但是没有词向量维度
                return：修改后的x
    """

    def __init__(self,d_model,num_heads,device = None,dtype = None,
                 theta = None,max_seq_len = None):
        super().__init__()

        # 创建四个投影层
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 一次性生成多头的qkv
        self.q_proj = Linear(d_model,d_model,
                                device = device,
                                dtype = dtype
                                )
        self.k_proj = Linear(d_model,d_model,
                                device = device,
                                dtype = dtype
                                )
        self.v_proj = Linear(d_model,d_model,
                                device = device,
                                dtype = dtype
                                )

        self.out_proj = Linear(d_model,d_model,
                                  device = device,
                                  dtype = dtype)


        # 判断是否需要旋转，需要旋转需要调用RoPE类，方便后续使用
        self.rope = None

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta,self.d_head,max_seq_len,
                                                  device = device)



    def forward(self,x,token_positions = None):

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # q.shape:[2,3,8] -> [2,3,2,4] 按注意力头切分
        q = q.reshape(*q.shape[:-1],self.num_heads,self.d_head)

        # q.shape:[2,3,2,4] -> [2,2,3,4] 整体顺序由注意力头决定，一个头里包含不同token
        q = q.transpose(-3,-2)

        k = k.reshape(*k.shape[:-1],self.num_heads,self.d_head)
        k = k.transpose(-3,-2)

        v = v.reshape(*v.shape[:-1],self.num_heads,self.d_head)
        v = v.transpose(-3,-2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    x.shape[-2],
                    device = x.device,
                    dtype = torch.long
                )
            positions_for_heads = token_positions.unsqueeze(-2)

            q = self.rope(q,positions_for_heads)
            k = self.rope(k,positions_for_heads)

        # 构建自注意力掩码
        mask = torch.tril(
            torch.ones(x.shape[-2],x.shape[-2],
                       device = x.device,
                       dtype = torch.bool)
        )

        # result.shape:[2,2,3,4]
        result = scaled_dot_product_attention(q,k,v,mask)

        result = result.transpose(-3,-2)
        result = result.reshape(*result.shape[:-2],self.d_model)

        return self.out_proj(result)


# 组装Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,
                 theta = None,max_seq_len = None,
                 use_rmsnorm = True,
                 device = None,dtype = None):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm

        # 分别设计两个层归一化，一个用在自注意力层前，一个用在前馈神经前
        self.ln1 = RMSNorm(d_model,
                           device = device,
                           dtype = dtype)
        self.ln2 = RMSNorm(d_model,
                           device = device,
                           dtype = dtype)
        # 设计自注意力机制，方便残差神经网络构建
        self.attn = MultiHeadSelfAttention(d_model =d_model,
                                           num_heads =num_heads,
                                           device = device,
                                           dtype = dtype,
                                           theta = theta,
                                           max_seq_len=max_seq_len)
        # 调用SwiGLU，建立前馈神经网络
        self.ffn = SwiGLU(d_model,d_ff,
                          device = device,
                          dtype = dtype)

    def forward(self,x):
        # 消融实验1:是否使用RMSNorm
        x_norm = self.ln1(x) if self.use_rmsnorm else x
        h = x + self.attn(x_norm)
        # 消融实验1
        h_norm = self.ln2(h) if self.use_rmsnorm else h
        result = h + self.ffn(h_norm)
        return result

# 定义模型类
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,
                 num_layers,num_heads,d_ff,rope_theta,
                 use_rmsnorm = True,
                 device = None,dtype = None):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size,d_model,
                                          device = device,
                                          dtype = dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,
                             rope_theta,context_length,
                             self.use_rmsnorm,
                             device = device,
                             dtype = dtype) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model,device = device,dtype = dtype)

        self.lm_head = Linear(d_model,vocab_size,device = device,dtype = dtype)




    def forward(self,token_ids):
        # 把token ID转化为词向量
        result = self.token_embeddings(token_ids)

        # 进入TransformerBlock循环
        for layer in self.layers:
            result = layer(result)

        # 归一化RMSNorm
        result = self.ln_final(result) if self.use_rmsnorm else result

        # 最后进入线性层，映射Vocab
        result = self.lm_head(result)
        return result


# 交叉熵损失函数
def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:

    maximum = torch.amax(logits,dim = -1,keepdim = True)
    shift = logits - maximum

    exp = torch.exp(shift)

    exp_sum = torch.sum(exp,dim = -1,keepdim = True)

    log = torch.log(exp_sum)

    index = targets.unsqueeze(-1)

    correct_score = shift.gather(dim = -1,index = index)

    loss = log - correct_score

    avg_loss = torch.mean(loss)

    return avg_loss





if __name__ == '__main__':
    token_ids = torch.LongTensor([[0,1,2,3,4,5,6,7,8,9]])
    model = TransformerLM(20,20,32,3,4,
                          64,10000,device = 'mps',dtype = torch.float32)
    print(model(token_ids).shape)