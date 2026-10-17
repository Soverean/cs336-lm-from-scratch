import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3,
                 betas=(0.9,0.999),eps = 1e-8,weight_decay = 0.01):
        defaults = {
            'lr':lr,
            'betas':betas,
            'eps':eps,
            'weight_decay':weight_decay
        }

        super().__init__(params,defaults)

    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            beta1,beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 若state为空，给几个数赋初值
                if not state:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                # 开始梯度下降
                t = state['t'] + 1
                m = state['m']
                v = state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2

                step_size = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 更新权重矩阵,此时无需记录梯度
                with torch.no_grad():
                    # p -= 和 p = p - 不一样，pytorch张量的 -= 会进行原地修改，这里需要原地修改
                    p -= step_size * m / (torch.sqrt(v) + eps)

                    # L2正则化(AdamW版)
                    p -= weight_decay * p * lr


                # 保存参数
                state['t'] = t
                state['m'] = m
                state['v'] = v

# 加入学习率预热、裁剪功能，动态更新学习率
def get_lr_cosine_schedule(it,max_learning,min_learning,warmup_iters,cosine_cycle_iters):
    """
        实际控制学习率的函数，会覆盖掉其他任何地方传给optimizer的学习率
        更换为预热、余弦裁剪的学习率
    """
    if it < warmup_iters:
        result = it / warmup_iters * max_learning
    elif it < cosine_cycle_iters:
        process = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        result = min_learning + (1 + math.cos(process * math.pi)) * (max_learning - min_learning) / 2
    else:
        result = min_learning
    return result

# 梯度裁剪，当整体梯度同时过大时，按相同比例缩小梯度，限制当前步更新
def gradient_clipping(parameters,max_l2_norm):
    parameters_list = [p for p in parameters if p.grad is not None]

    if not parameters_list:
        return

    sum = torch.tensor(0.0,
                       device = parameters_list[0].grad.device,
                       dtype = torch.float32)
    for p in parameters_list:
        sum += p.grad.float().square().sum()
    G = torch.sqrt(sum)
    if G > max_l2_norm:
        result = max_l2_norm / (G + 1e-6)
        with torch.no_grad():
            for p in parameters_list:
                p.grad.mul_(result)

if __name__ == '__main__':
    lr = 1e-3
    get_lr_cosine_schedule(1,1e-4,1e-5,10,10)