import numpy as np
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

import data_reader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import *
from Model.SelectFaseterRCNN import SelectFasterRCNN
from command import *

from function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = data_reader.CTData_roi(r"..\..\Medical Image\divided data", True)
test_dataset = data_reader.CTData_roi(r"..\..\Medical Image\divided data", False)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=4,
                         shuffle=False,
                         num_workers=4,
                         collate_fn=collate_fn)

net = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights. DEFAULT)
num_classes = 2
in_features = net.roi_heads.box_predictor.cls_score.in_features
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# net = SelectFasterRCNN(3, 1, device)

opt = optim.SGD(net.parameters(), momentum=0.9, lr=0.005, weight_decay=0.0005)
# opt = optim.Adam(net.parameters(), lr=0.01, eps=1e-8)

def training_rcnn_main(load=False):
    net.to(device)

    epoch = 0
    losses = ([], [])
    if load:
        epoch, losses = load_net(net, opt, "Data")

    train_loss, test_loss = train_fasterRCnn(net, train_loader, test_loader, opt,
                                             begin_epoch=epoch, epoch_num=15, save_num=-10, print_num=50, device=device,
                                             test_device=device, losses=losses)

    train_losses = {}
    for i in train_loss:
        for k, v in i.items():
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].append(v)

    test_losses = {}
    for i in test_loss:
        for k, v in i.items():
            if k not in test_losses:
                test_losses[k] = []
            test_losses[k].append(v)

    plt.subplot(121)

    for k, v in train_losses.items():
        plt.plot(v, label=k)

    plt.subplot(122)

    for k, v in test_losses.items():
        plt.plot(v, label=k)

    plt.show()


def show_rcnn_main():
    net.to(device)
    net.train(False)

    epoch = 0
    losses = ([], [])
    epoch, losses = load_net(net, opt, "Data")

    train_loss, test_loss = losses

    data_loader = train_loader
    idx = 0
    for data in data_loader:
        images, _t = data
        images = list(image.to(device) for image in images)
        targets = net(images)
        for target, image, t in zip(targets, images, _t):
            boxes = target["boxes"].cpu().detach().numpy()
            image = image.cpu().detach().numpy()
            orboxes = t["boxes"]
            show_box(image, boxes, orboxes)
            idx += 1
        if idx > 100:
            break

    train_losses = {}
    for i in train_loss:
        for k, v in i.items():
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].append(v)

    train_losses_np = {}
    for k in train_losses.keys():
        lar = np.array(train_losses[k])
        a = np.min(lar)
        k = np.max(lar) - a
        lar = (lar - a) / k
        train_losses_np[k] = lar

    test_losses = {}
    for i in test_loss:
        for k, v in i.items():
            if k not in test_losses:
                test_losses[k] = []
            test_losses[k].append(v)

    test_losses_np = {}
    for k in test_losses.keys():
        lar = np.array(test_losses[k])
        a = np.min(lar)
        k = np.max(lar) - a
        lar = (lar - a) / k
        test_losses_np[k] = lar

    plt.subplot(121)

    for k, v in train_losses_np.items():
        plt.plot(v, label=k)

    plt.subplot(122)

    for k, v in test_losses_np.items():
        plt.plot(v, label=k)

    plt.show()


if __name__ == "__main__":
    # print(net)
    training_rcnn_main()
    # show_rcnn_main()
