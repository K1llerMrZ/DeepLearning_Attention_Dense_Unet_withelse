import math
import os
from typing import Dict, List, Tuple
from function import save_net

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import time

import torchvision
from torchvision.models.detection import FasterRCNN


def train(net: nn.Module, trainloader: dataloader.DataLoader, testloader: dataloader.DataLoader, opt: optim.Optimizer,
          loss_function=nn.CrossEntropyLoss(), begin_epoch=0, epoch_num=5, save_num=0, print_num=10, device="cpu",
          test_device="cpu", losses=([], [])):
    start = time.time()
    net.train()
    train_loss_list = losses[0]
    test_loss_list = losses[1]
    data_folder = "Data"
    for _epoch in range(epoch_num):
        epoch = _epoch + begin_epoch
        print(epoch + 1, "epoch running on", device)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 清空梯度缓存
            opt.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, labels)

            # class DiceLoss(nn.Module):
            #     def __init__(self, smooth=1e-6):
            #         super(DiceLoss, self).__init__()
            #         self.smooth = smooth
            #
            #     def forward(self, pred, target):
            #         pred = torch.sigmoid(pred)  # 确保概率范围
            #         intersection = (pred * target).sum()
            #         return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
            #
            # class BCEDiceLoss(nn.Module):
            #     def __init__(self, bce_weight=0.5):
            #         super(BCEDiceLoss, self).__init__()
            #         self.bce = nn.BCEWithLogitsLoss()
            #         self.dice = DiceLoss()
            #         self.w = bce_weight
            #
            #     def forward(self, pred, target):
            #         bce_loss = self.bce(pred, target)
            #         dice_loss = self.dice(pred, target)
            #         return self.w * bce_loss + (1 - self.w) * dice_loss

            loss.backward()
            opt.step()

            # 打印统计信息
            running_loss += loss.item()

            if (i + 1) % print_num == 0:
                inputs, labels = iter(testloader).__next__()
                inputs, labels = inputs.to(test_device), labels.to(test_device)
                net.train(False)
                if test_device != device:
                    net.to(test_device)
                outputs = net(inputs)
                loss = loss_function(outputs, labels)

                print('[%d, %5d] train loss: %.16f' % (epoch + 1, i + 1, running_loss / print_num),
                      'test loss: %.16f' % loss.item())
                train_loss_list.append(running_loss / print_num)
                test_loss_list.append(loss.item())
                running_loss = 0.0
                if test_device != device:
                    net.to(device)
                net.train()

            if save_num > 0 and (i + 1) % save_num == 0:
                save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)

        if save_num < 0 and (epoch + 1) % save_num == 0:
            save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)

    print('Finished Training! Total cost time: ', time.time() - start)
    return train_loss_list, test_loss_list


def train_fasterRCnn(net: FasterRCNN, trainloader: dataloader.DataLoader,
                     testloader: dataloader.DataLoader, opt: optim.Optimizer,
                     begin_epoch=0, epoch_num=5, save_num=0, print_num=10, device="cpu", test_device="cpu",
                     losses=([], [])):
    start = time.time()
    train_loss_list = losses[0]
    test_loss_list = losses[1]
    data_folder = "Data"
    if_save = False

    net.train()
    for _epoch in range(epoch_num):
        epoch = _epoch + begin_epoch
        if_save = False
        print(epoch + 1, "epoch running on", device)

        running_loss = {}
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # print(labels)
            if type(targets) != list:
                raise TypeError("labels should be list, now is %s" % type(targets).__name__)

            # 清空梯度缓存
            opt.zero_grad()

            try:
                outputs: Dict[torch.Tensor] = net(images, targets)
            except Exception as e:
                print("Wring! Throw Error:", e)
                continue

            loss = None
            for k, lo in outputs.items():
                if loss is None:
                    loss = lo
                else:
                    loss += lo

                if math.isnan(lo.item()):
                    save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch_num,
                             file=f"[{time.ctime()}]error-data")
                    with open("error.log", "a") as file:
                        file.write(f"[{time.ctime()}] {k} is nan at epoch {epoch + 1} batch {i + 1}\n")
                    # continue
                    raise ValueError(f"{k} is nan at epoch {epoch + 1} batch {i + 1}, please check your data.")
            loss.backward()
            opt.step()

            # 打印统计信息
            for item in outputs:
                if item not in running_loss:
                    running_loss[item] = outputs[item].item()
                else:
                    running_loss[item] += outputs[item].item()

            if (i + 1) % print_num == 0:
                # 获取输入数据
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # print(labels)
                if type(targets) != list:
                    raise TypeError("labels should be list, now is %s" % type(targets).__name__)

                outputs: Dict[torch.Tensor] = net(images, targets)

                print("===============[%d epoch, %5d mini batch]================" % (epoch + 1, i + 1))
                for x in outputs:
                    print('train %s: %.16f' % (x, running_loss[x] / print_num),
                          'test %s: %.16f' % (x, outputs[x].item()))
                train_loss_list.append(running_loss)
                test_loss = {}
                for item in outputs:
                    test_loss[item] = outputs[item].item()
                test_loss_list.append(test_loss)
                running_loss = {}
                if test_device != device:
                    net.to(device)

            if save_num > 0 and (i + 1) % save_num == 0:
                save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)
                if_save = True

        if save_num < 0 and (epoch + 1) % -save_num == 0 and not if_save:
            save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)
            if_save = True

    if not if_save:
        save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch_num)
    print('Finished Training! Total cost time: ', time.time() - start)
    return train_loss_list, test_loss_list


def train_double_out(net: nn.Module, trainloader: dataloader.DataLoader, testloader: dataloader.DataLoader,
                     opt: optim.Optimizer,
                     loss_function1=nn.CrossEntropyLoss(), loss_function2=nn.MSELoss(),
                     begin_epoch=0, epoch_num=5, save_num=0, print_num=10, device="cpu",
                     test_device="cpu", losses=([], [])):
    start = time.time()
    net.train()
    train_loss_list = losses[0]
    test_loss_list = losses[1]
    data_folder = "Data"
    for _epoch in range(epoch_num):
        epoch = _epoch + begin_epoch
        print(epoch + 1, "epoch running on", device)

        running_loss1 = 0.0
        running_loss2 = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels, subtlety = data
            inputs, labels, subtlety = inputs.to(device), labels.to(device), subtlety.to(device).float()
            # 清空梯度缓存
            opt.zero_grad()

            outputs1, outputs2, dis = net(inputs)
            loss1 = loss_function1(outputs1, labels)
            loss2 = loss_function2(outputs2, subtlety)
            loss = loss2 + loss1
            loss.backward()
            opt.step()

            # 打印统计信息
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()

            if (i + 1) % print_num == 0:
                inputs, labels, subtlety = iter(testloader).__next__()
                inputs, labels, subtlety = inputs.to(test_device), labels.to(test_device), subtlety.to(device).float()
                net.train(False)
                if test_device != device:
                    net.to(test_device)
                outputs1, outputs2, dis = net(inputs)
                loss1 = loss_function1(outputs1, labels)
                loss2 = loss_function2(outputs2, subtlety)

                print('[%d, %5d] train loss 1: %.16f' % (epoch + 1, i + 1, running_loss1 / print_num),
                      'test loss 1: %.16f' % loss1.item(),
                      'train loss 2: %.16f' % (running_loss2 / print_num),
                      'test loss 2: %.16f' % loss2.item()
                      )
                train_loss_list.append((running_loss1 / print_num, running_loss2 / print_num))
                test_loss_list.append((loss1.item(), loss2.item()))
                running_loss1 = 0.0
                running_loss2 = 0.0
                if test_device != device:
                    net.to(device)
                net.train()

            if save_num > 0 and (i + 1) % save_num == 0:
                save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)

        if save_num < 0 and (epoch + 1) % save_num == 0:
            save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)

    print('Finished Training! Total cost time: ', time.time() - start)
    return train_loss_list, test_loss_list


def train_in_double(net: nn.Module, trainloader: dataloader.DataLoader, testloader: dataloader.DataLoader,
                    opt: optim.Optimizer, loss_function=nn.CrossEntropyLoss(),
                    begin_epoch=0, epoch_num=5, save_num=0, print_num=10, device="cpu",
                    test_device="cpu", losses=([], [])):
    start = time.time()
    net.train()
    train_loss_list = losses[0]
    test_loss_list = losses[1]
    data_folder = "Data"
    for _epoch in range(epoch_num):
        epoch = _epoch + begin_epoch
        print(epoch + 1, "epoch running on", device)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels, tag = data
            inputs, labels, tag = inputs.to(device), labels.to(device), tag.to(device).float()
            # 清空梯度缓存
            opt.zero_grad()

            outputs, _ = net(inputs, labels)
            loss = loss_function(outputs, tag)
            loss.backward()
            opt.step()

            # 打印统计信息
            running_loss += loss.item()

            if (i + 1) % print_num == 0:
                data = iter(testloader).__next__()

                net.train(False)
                if test_device != device:
                    net.to(test_device)

                inputs, labels, tag = data
                inputs, labels, tag = inputs.to(device), labels.to(device), tag.to(device).float()
                # 清空梯度缓存
                opt.zero_grad()

                outputs, _ = net(inputs, labels)
                loss = loss_function(outputs, tag)

                print('[%d, %5d] train loss: %.16f' % (epoch + 1, i + 1, running_loss / print_num),
                      'test loss: %.16f' % loss.item())
                train_loss_list.append(running_loss / print_num)
                test_loss_list.append(loss.item())
                running_loss = 0.0
                if test_device != device:
                    net.to(device)
                net.train()

            if save_num > 0 and (i + 1) % save_num == 0:
                save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)

        if save_num < 0 and (epoch + 1) % save_num == 0:
            save_net(net, opt, (train_loss_list, test_loss_list), data_folder, epoch)

    print('Finished Training! Total cost time: ', time.time() - start)
    return train_loss_list, test_loss_list
