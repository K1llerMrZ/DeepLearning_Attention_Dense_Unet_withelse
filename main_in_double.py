import numpy as np
import torch.utils.data

import data_reader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import *
from Model.Attention_Unet import AttU_Net
from Model.Attention_Unet_plus import AttU_Net_plus
from Model.ResidualUNet import ResidualUNet
from Model.IdentifyNet import IDentifyNet
from command import *
from function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = data_reader.CTData_forU_identy(r"..\..\Medical Image\divided data\cut", True)
test_dataset = data_reader.CTData_forU_identy(r"..\..\Medical Image\divided data\cut", False)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

net = IDentifyNet(1, 1)
# opt = optim.Adam(net.parameters(), lr=1e-3)  # learning rate
opt = optim.SGD(net.parameters(), momentum=0.9, lr=5e-3, weight_decay=0.0005)
lossFc = nn.CrossEntropyLoss()


def training_main(load=False):
    net.to(device)

    epoch = 0
    losses = ([], [])
    if load:
        epoch, losses = load_net(net, opt, "Data")

    t1, t2 = train_in_double(net, train_loader, test_loader, opt, lossFc,
                             begin_epoch=epoch, epoch_num=8, save_num=-2, print_num=25,
                             device=device, test_device=device, losses=losses)
    train_loss, test_loss = np.array(t1), np.array(t2)

    plt.plot(train_loss, c='r', label="train loss")
    plt.plot(test_loss, c='b', label="test loss")

    plt.ylabel("loss")

    plt.show()


def testing_main():
    net.to(device)

    load_net(net, opt, "Data")

    def get_loss(dataloader: torch.utils.data.DataLoader):
        for idx, data in enumerate(dataloader, 0):
            x, t = data
            x, t = x.to(device), t

            y = net(x)
            y = y.to("cpu")

            y, t = y.detach().numpy()[0][0], t.numpy()[0][0]

            loss = t - y


def show_main():
    noml = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    epoch, te = load_net(net, opt, "Data")

    net.train(False)

    threshold = 0.6
    idx = 0
    while idx < 10:
        data, lab, tag = test_dataset[idx]
        data, lab = torch.from_numpy(data).unsqueeze(0), torch.from_numpy(lab).unsqueeze(0)

        output, feature = net(data, lab)

        x, y, t = data[0][0].numpy(), output[0], lab[0][0].numpy()

        plt.subplot(231)
        plt.title("Input")
        plt.axis('off')
        plt.imshow(x, cmap="gray")

        plt.subplot(232)
        plt.title("Input")
        plt.axis('off')
        plt.imshow(t, cmap="gray")

        print(tag*5, y*5)

        # # TN
        # plt.subplot(337)
        # plt.title("TN")
        # img = np.zeros_like(loss)
        # img[t < 0.5] = (-(y - 1))[t < 0.5]
        # plt.imshow(img)

        plt.show()
        idx += 1

    # for i in range(16):
    #     data, lab = test_dataset[i]
    #     data, lab = torch.from_numpy(data).unsqueeze(0), torch.from_numpy(lab).unsqueeze(0)
    #
    #     output = net(data)
    #
    #     x, y, t = data[0][0].numpy(), output[0][0].detach().numpy(), lab[0][0].numpy()
    #     y = noml(y)
    #     y[y < threshold] = 0
    #     y[y > threshold] = 1
    #     t[t > 0] = 1
    #
    #     m1 = get_statistic(y)
    #     m2 = get_statistic(t)
    #
    #     plt.subplot(4, 4, i + 1)
    #     plt.plot(list(m1.values()), c="r")
    #     plt.plot(list(m2.values()), c="b")
    #     # plt.axis('off')
    #     # plt.xticks(list(m1.keys()))

    plt.show()

    t1, t2 = te
    train_loss, test_loss = np.array(t1), np.array(t2)

    plt.plot(test_loss, c='b', label="test loss")
    plt.plot(train_loss, c='r', label="train loss")
    plt.legend()
    plt.ylabel("loss")

    plt.show()


if __name__ == "__main__":
    # show_main()
    # testing_main()
    training_main()
