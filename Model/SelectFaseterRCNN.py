import torch
import torch.nn as nn
import torchvision

import torchvision.models.detection as vmd
import matplotlib.pyplot as plt

import numpy as np
from torchvision.models.detection.anchor_utils import AnchorGenerator


class ManyConv(nn.Module):
    def __init__(self, in_ch, out_ch, num=2):
        super(ManyConv, self).__init__()

        self.Ws = nn.ModuleList()

        self.Wi = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        if num > 2:
            for i in range(num - 2):
                self.Ws.append(nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))

        self.Wo = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
        )

        self.relu = nn.ReLU(inplace=True)
        self.cov1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Wi(x)

        for n in self.Ws:
            x1 = n(x1)

        out = self.Wo(x1)

        out = self.relu(out + self.cov1x1(x))
        return out


class Conv13(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv13, self).__init__()

        self.out_channels = out_ch

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.convs = nn.ModuleList()
        self.convs.append(ManyConv(in_ch, 16, 2))
        self.convs.append(ManyConv(16, 32, 2))
        self.convs.append(ManyConv(32, 64, 3))
        self.convs.append(ManyConv(64, 128, 3))

        self.outConv = ManyConv(128, out_ch, 3)

    def forward(self, x):
        x1 = x
        for net in self.convs:
            x1 = net(x1)
            x1 = self.maxpool(x1)

        out = self.outConv(x1)
        return out


def SelectFasterRCNN(in_ch, out_ch, device="cpu"):
    model = Conv13(in_ch, out_ch)
    model.to(device)
    return vmd.FasterRCNN(model,
                          num_classes=2,
                          # transform parameters
                          min_size=512,
                          max_size=1024,
                          image_mean=None,
                          image_std=None,
                          # RPN parameters
                          rpn_anchor_generator=AnchorGenerator(((4,), (8,), (16,), (32,), (64,))),
                          rpn_head=None,
                          rpn_pre_nms_top_n_train=200,
                          rpn_pre_nms_top_n_test=100,
                          rpn_post_nms_top_n_train=200,
                          rpn_post_nms_top_n_test=100,
                          rpn_nms_thresh=0.7,
                          rpn_fg_iou_thresh=0.7,
                          rpn_bg_iou_thresh=0.3,
                          rpn_batch_size_per_image=16,
                          rpn_positive_fraction=0.5,
                          rpn_score_thresh=0.0,
                          # Box parameters
                          box_roi_pool=None,
                          box_head=None,
                          box_predictor=None,
                          box_score_thresh=0.05,
                          box_nms_thresh=0.5,
                          box_detections_per_img=100,
                          box_fg_iou_thresh=0.5,
                          box_bg_iou_thresh=0.5,
                          box_batch_size_per_image=16,
                          box_positive_fraction=0.25,
                          bbox_reg_weights=None,
                          )


if __name__ == "__main__":
    # net = Conv13(1,1)
    # net.to("cuda:0")
    #
    # x = torch.randn(1, 1, 512, 512).to("cuda:0")
    #
    # out = net(x)

    net = SelectFasterRCNN(3, 1, "cuda:0")
    net = net.cuda()
    net.train()

    boxes = torch.LongTensor([
        [1, 2, 3, 4]
    ]).to("cuda:0")

    labels = torch.LongTensor([
        1
    ]).to("cuda:0")

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels

    t = [target]

    x = torch.randn(1, 3, 512, 512).to("cuda:0")

    loss = net.forward(x, t)

    for i in loss:
        print(i, "is", loss[i].item())
        # loss[i].backward()

    losses = None
    for i in loss:
        if losses is None:
            losses = loss[i]
        else:
            losses += loss[i]

    losses.backward()
    print(loss)
    print(losses, losses.item())
