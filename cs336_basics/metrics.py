from cs336_basics import config
from pathlib import Path
import csv


def append_metrics(path,metrics):
    # 创建路径的根目录
    path.parent.mkdir(parents = True,exist_ok = True)

    with open(path,'a',newline = '',encoding = 'utf-8') as f:
        # 创建写入器、判断是否写表头、写入数据
        fieldnames = [
            'step','elapsed_s','lr',
            'train_loss','val_loss','val_ppl'
        ]

        writer = csv.DictWriter(f,fieldnames = fieldnames)

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(metrics)






if '__main__' == __name__:
    pass
