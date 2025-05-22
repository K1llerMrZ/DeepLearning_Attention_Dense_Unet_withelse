import copy
import math
import os
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class CTData(Dataset):

    def __init__(self, root, train=True, mod="noml"):
        super(CTData, self).__init__()
        self.train = train
        self.mod = mod

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_root = root + r"\train"
        else:
            file_root = root + r"\test"

        self.folder = file_root + "\\"

        filenames = os.listdir(file_root)

        self.images = []
        self.labels = {}

        for file in filenames:
            ct_name = os.path.splitext(file)[0]
            if os.path.splitext(file)[1] == ".png":
                self.images.append((file, ct_name))
            elif os.path.splitext(file)[1] == ".csv":
                t = []
                with open(self.folder + "\\" + file, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    for line in reader:
                        t.append([int(x) for x in line])
                self.labels[ct_name] = t

    def __getitem__(self, index):
        img_name = self.folder + self.images[index][0]

        label = self.labels[self.images[index][1]]

        img = np.array(plt.imread(img_name))[:, :, :1]

        label_img = np.zeros_like(img)

        for x, y, wid in label:
            for i in range(x - wid, x + wid):
                for j in range(y - wid, y + wid):
                    far = math.sqrt(math.fabs(i - x) ** 2 + math.fabs(i - y) ** 2)
                    label_img[j, i, 0] = 1 / (1 + far)

        return img.transpose(2, 0, 1), label_img.transpose(2, 0, 1)

    def __len__(self):
        return len(self.images)

    def whether_nodules(self, index) -> bool:
        label = self.labels[self.images[index][1]]
        return len(label) != 0

    def find_nodules(self, begin=0) -> int:
        for i in range(begin + 1, len(self)):
            if self.whether_nodules(i):
                return i
        return -1

    def set_mod(self, mod):
        self.mod = mod


class CTData_roi(Dataset):

    def __init__(self, root, train=True):
        super(CTData_roi, self).__init__()
        self.train = train

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_root = root + r"\train"
        else:
            file_root = root + r"\test"

        self.folder = file_root + "\\"

        self.datas = []
        with open(self.folder + "annotations.csv") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if len(line) != 0:
                    self.datas.append(tuple(line))

    def __getitem__(self, index):
        imgname, x, y, wid = self.datas[index]
        imgname = self.folder + imgname + ".png"

        x, y, wid = int(x), int(y), int(wid)

        or_img = np.array(plt.imread(imgname))[:, :, :3]
        img = copy.deepcopy(or_img)

        boxes = []
        labels = []
        area = []

        boxes = [[x - wid, y - wid, x + wid, y + wid]]

        trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.OneOf([
            #     A.GaussianBlur(),  # 将高斯噪声添加到输入图像
            #     A.GaussNoise(),  # 将高斯噪声应用于输入图像。
            # ], p=0.2),  # 应用选定变换的概率
            A.OneOf([
                A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
        ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']), )

        # try:
        #     trimgs = trans(image=img, bboxes=boxes, category_ids=[1])
        #     pass
        # except ValueError as e:
        #     # print("Wring! Throw Error:", e)
        #     pass
        # else:
        #     img = trimgs["image"]
        #     boxes = trimgs["bboxes"]

        if len(boxes) == 0:
            img = or_img
            boxes = [[x - wid, y - wid, x + wid, y + wid]]

        area.append((boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]))
        labels.append(1)

        area = torch.as_tensor(area, dtype=torch.float32)
        # there is only one class
        boxes = torch.LongTensor(boxes)
        labels = torch.LongTensor(labels)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': area}

        return torch.from_numpy(img.transpose(2, 0, 1)), target

    def __len__(self):
        return len(self.datas)


class CTData_forU(Dataset):

    def __init__(self, root, train=True, tranform=None):
        super(CTData_forU, self).__init__()
        self.train = train

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_root = root + r"\train"
        else:
            file_root = root + r"\test"

        self.folder = file_root + "\\"

        mask_files = os.listdir(os.path.join(file_root, "mask"))
        image_files = os.listdir(os.path.join(file_root, "image"))

        self.filenames = list(zip(image_files, mask_files))
        random.shuffle(self.filenames)

        if tranform is None:
            self.trans = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.OneOf([
                #     A.GaussianBlur(),  # 将高斯噪声添加到输入图像
                #     A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                # ], p=0.2),  # 应用选定变换的概率
                A.OneOf([
                    A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                    A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                    A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                # 随机应用仿射变换：平移，缩放和旋转输入
                A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
            ], p=0.5)
        else:
            self.trans = tranform

    def __getitem__(self, index):
        i, m = self.filenames[index]

        img_name = self.folder + "image\\" + i
        mask_name = self.folder + "mask\\" + m

        img = np.array(plt.imread(img_name))[:, :, :1]
        mask = np.array(plt.imread(mask_name))[:, :, :1]

        # plt.subplot(221)
        # plt.imshow(img, cmap="gray")
        # plt.subplot(222)
        # plt.imshow(mask)

        if self.train:
            dicts = self.trans(image=img, mask=mask)
            img = dicts["image"]
            mask = dicts["mask"]

        # plt.subplot(223)
        # plt.imshow(img, cmap="gray")
        # plt.subplot(224)
        # plt.imshow(mask)
        #
        # plt.show()

        return img.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def __len__(self):
        return len(self.filenames)


class CTData_forU_identy(Dataset):

    def __init__(self, root, train=True, tranform=None):
        super(CTData_forU_identy, self).__init__()
        self.train = train

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_root = root + r"\train"
        else:
            file_root = root + r"\test"

        sub = {}
        sub_file = root + r"\annotations.csv"
        with open(sub_file) as sf:
            reader = csv.reader(sf)
            for row in reader:
                if row:
                    sub[int(row[0])] = int(row[1])

        self.subtlety = sub

        self.folder = file_root + "\\"

        mask_files = os.listdir(os.path.join(file_root, "mask"))
        image_files = os.listdir(os.path.join(file_root, "image"))

        self.filenames = list(zip(image_files, mask_files))
        random.shuffle(self.filenames)

        if tranform is None:
            self.trans = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.OneOf([
                #     A.GaussianBlur(),  # 将高斯噪声添加到输入图像
                #     A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                # ], p=0.2),  # 应用选定变换的概率
                A.OneOf([
                    A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                    A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                    A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                # 随机应用仿射变换：平移，缩放和旋转输入
                A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
            ], p=0.5)
        else:
            self.trans = tranform

    def __getitem__(self, index):
        i, m = self.filenames[index]

        img_name = self.folder + "image\\" + i
        mask_name = self.folder + "mask\\" + m

        img = np.array(plt.imread(img_name))[:, :, :1]
        mask = np.array(plt.imread(mask_name))[:, :, :1]

        if self.train:
            dicts = self.trans(image=img, mask=mask)
            img = dicts["image"]
            mask = dicts["mask"]

        tag = np.zeros((5,))
        tag[self.subtlety[int(i[:6])] - 1] = 1

        return img.transpose(2, 0, 1), mask.transpose(2, 0, 1), tag

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    data = CTData_forU(r"F:\Medical Image\divided data\cut")
    for i in range(len(data)):
        img, lab = data.__getitem__(i)
        plt.subplot(131)
        plt.imshow(img, cmap="gray")
        plt.subplot(132)
        plt.imshow(lab)
        break

    plt.show()
