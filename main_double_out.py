import numpy as np
import torch.utils.data

import data_reader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import *
from Model.Attention_Unet import AttU_Net
from Model.Attention_Unet_plus import AttU_Net_plus
from Model.Attention_Unet_plus_with_identify import AttU_Net_plus_with_identify
from Model.ResidualUNet import ResidualUNet
from command import *
from function import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = data_reader.CTData_forU_identy(r"..\..\Medical Image\divided data\cut", True)
test_dataset = data_reader.CTData_forU_identy(r"..\..\Medical Image\divided data\cut", False)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

net = AttU_Net_plus_with_identify(1, 1)
opt = optim.Adam(net.parameters(), lr=2e-4)  # learning rate
lossFc1 = SoftDiceLoss()
lossFc2 = nn.CrossEntropyLoss()


def training_main(load=False):
    net.to(device)

    epoch = 0
    losses = ([], [])
    if load:
        epoch, losses = load_net(net, opt, "Data")

    t1, t2 = train_double_out(net, train_loader, test_loader, opt, lossFc1, lossFc2,
                              begin_epoch=epoch, epoch_num=100, save_num=-2, print_num=25,
                              device=device, test_device=device, losses=losses)
    train_loss, test_loss = np.array(t1), np.array(t2)

    train_loss1 = [i for i, j in train_loss]
    train_loss2 = [j for i, j in train_loss]
    test_loss1 = [i for i, j in test_loss]
    test_loss2 = [j for i, j in test_loss]

    plt.subplot(121)
    plt.plot(train_loss1, c='r', label="train loss")
    plt.plot(test_loss1, c='b', label="test loss")
    plt.xlabel("loss")

    plt.subplot(122)
    plt.plot(train_loss2, c='r', label="train loss")
    plt.plot(test_loss2, c='b', label="test loss")
    plt.xlabel("loss")

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

    threshold = 0.6
    idx = 0
    while idx < 10:
        data, lab, tsub = test_dataset[idx]
        data, lab = torch.from_numpy(data).unsqueeze(0), torch.from_numpy(lab).unsqueeze(0)

        output, sub, tab = net(data)

        x, y, t = data[0][0].numpy(), output[0][0].detach().numpy(), lab[0][0].numpy()
        y = noml(y)
        y[y < threshold] = 0
        y[y > threshold] = 1
        t[t > 0] = 1

        loss = y - t

        print(tab)
        plt.subplot(231)
        plt.title("Input")
        plt.axis('off')
        plt.imshow(x, cmap="gray")

        plt.subplot(232)
        plt.title("Output")
        plt.axis('off')
        plt.imshow(y)

        plt.subplot(233)
        plt.title("Label")
        plt.axis('off')
        plt.imshow(t)

        # FN
        plt.subplot(234)
        plt.title("FN")
        img = np.zeros_like(loss)
        img[loss < 0] = 1
        plt.axis('off')
        plt.imshow(img)

        # FP
        plt.subplot(235)
        plt.title("FP")
        img = np.zeros_like(loss)
        img[loss > 0] = 1
        plt.axis('off')
        plt.imshow(img)

        # TP
        plt.subplot(236)
        plt.title("TP")
        img = np.zeros_like(loss)
        img[t > 0.5] = y[t > 0.5]
        plt.axis('off')
        plt.imshow(img)


        plt.show()
        idx += 1

    t1, t2 = te
    train_loss, test_loss = t1, t2

    train_loss1 = [i for i, j in train_loss]
    train_loss2 = [j for i, j in train_loss]
    test_loss1 = [i for i, j in test_loss]
    test_loss2 = [j for i, j in test_loss]

    plt.subplot(121)
    plt.plot(train_loss1, c='r', label="train loss")
    plt.plot(test_loss1, c='b', label="test loss")
    plt.ylabel("loss")

    plt.subplot(122)
    plt.plot(train_loss2, c='r', label="train loss")
    plt.plot(test_loss2, c='b', label="test loss")
    plt.ylabel("loss")

    plt.show()


if __name__ == "__main__":
    show_main()
    # testing_main()
    # training_main()
