import torch
from jinja2 import optimizer


# 保存检查点
def save_checkpoint(model,optimizer,iteration,out):
    """
        需要保存 模型状态、优化器状态、训练步数

    """
    checkpoint = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'iteration':iteration,
    }
    # 把文件写入 out 制定的文件
    torch.save(checkpoint,out)



# 加载检查点
def load_checkpoint(src,model,optimizer):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']