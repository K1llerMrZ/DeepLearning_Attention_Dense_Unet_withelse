import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 用于进行可视化
import tensorboard
from torchviz import make_dot

from Model.Attention_Unet_plus import AttU_Net_plus

if __name__ == "__main__":
    # 首先来搭建一个模型
    modelviz = AttU_Net_plus()
    # 创建输入
    sampledata = torch.rand(1, 1, 64, 64)
    # 看看输出结果对不对
    out = modelviz(sampledata)

    # 1. 来用tensorflow进行可视化
    with SummaryWriter("./log", comment="sample_model_visualization") as sw:
        sw.add_graph(modelviz, sampledata)

    # 2. 保存成pt文件后进行可视化
    # torch.save(modelviz, "./log/modelviz.pt")
    #
    # # 3. 使用graphviz进行可视化
    # out = modelviz(sampledata)
    # g = make_dot(out)
    # g.render('modelviz', view=False)  # 这种方式会生成一个pdf文件
