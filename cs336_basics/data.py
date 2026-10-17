import numpy as np
import torch

# 建立普通函数，完成数据批次处理
def get_batch(dataset,batch_size,context_length,device):
    """
        把分词器产生的一长串token ID，整理成模型每次训练需要的输入和目标
        初始预料随机选择多个起点，从每个起点开始构建batch，和对应的targets
    """

    starts = np.random.randint(
        low = 0,
        high = len(dataset) - context_length,
        size = batch_size,
    )
    x = []
    y = []
    for start in starts:
        token_list = dataset[start:start + context_length]
        target = dataset[start + 1:start + context_length + 1]
        x.append(token_list)
        y.append(target)
    x = np.array(x)
    y = np.array(y)
    x = torch.tensor(x,dtype = torch.long,device = device)
    y = torch.tensor(y,dtype = torch.long,device = device)
    return x,y



if __name__ == '__main__':
    dataset = np.random.randint(0,10,10)
    print(dataset)

    print(get_batch(dataset,4,3,device = 'cpu'))