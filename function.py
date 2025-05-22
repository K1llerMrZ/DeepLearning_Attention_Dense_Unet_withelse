import csv
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import time
import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection import FasterRCNN
from xml.etree.ElementTree import parse
import SimpleITK as sitk


def findAllMhd(base):
    list = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            if os.path.splitext(f)[1] == ".mhd":
                fullname = os.path.join(root, f)
                list.append(fullname)
    return list


def load_net(net: nn.Module, opt: optim.Optimizer, folder="Data", file=None, tag="pt"):
    netname = net.__class__.__name__
    if type(net) is FasterRCNN:
        netname += "(" + net.backbone.__class__.__name__ + ")"
    file_folder = folder + "\\" + netname + "-" + opt.__class__.__name__
    # file_folder = folder + "\\" + net.__class__.__name__ + "-" + opt.__class__.__name__

    if file is None:
        lis = []
        for i in os.listdir(file_folder):
            lis.append(time.strptime(os.path.splitext(i)[0], "%Y-%m-%d-%H%M%S"))
        lis.sort(reverse=True)
        file = time.strftime("%Y-%m-%d-%H%M%S", lis[0])

    filename = file_folder + "\\" + file + "." + tag

    parm = torch.load(filename)
    print("succeed load", filename)

    net.load_state_dict(parm["net"])
    opt.load_state_dict(parm["opt"])

    return parm["epoch"], parm["loss"]


def save_net(net: nn.Module, opt: optim.Optimizer, losses: Tuple[List, List],
             folder="Data", epoch=0, file=None, tag="pt"):
    netname = net.__class__.__name__
    if type(net) is FasterRCNN:
        netname += "(" + net.backbone.__class__.__name__ + ")"
    file_folder = folder + "\\" + netname + "-" + opt.__class__.__name__

    if not os.path.exists(file_folder):
        os.mkdir(file_folder)

    if file is None:
        file = time.strftime("%Y-%m-%d-%H%M%S", time.gmtime())

    filename = file_folder + "\\" + file + "." + tag

    try:
        net_parm = net.state_dict()
        opt_parm = opt.state_dict()

        parm = {"net": net_parm, "opt": opt_parm, "epoch": epoch, "loss": losses}
        torch.save(parm, filename)
        print("succeed save in file", filename)

    except RuntimeError:
        print("RuntimeError: open file", filename, "unsucess.")

    except Exception as e:
        print(e, "\nsave file unsucceed.")


def show_box(image, boxes, orboxes=None):
    plt.imshow(image[0], cmap="gray")
    ax = plt.gca()

    for box in boxes:
        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], color="red", fill=False, linewidth=1))

    if orboxes is not None:
        for box in orboxes:
            ax.add_patch(
                plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                              color="blue", fill=False, linewidth=1))

    plt.show()


def get_xml(path):
    print(f"Opening file {path}.")

    reader = sitk.ImageSeriesReader()

    try:
        dicom = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom)
        image = reader.Execute()
    except RuntimeError as e:
        print(e)
        return

    Origin = image.GetOrigin()
    Spacing = image.GetSpacing()

    img_array = sitk.GetArrayFromImage(image)

    tranz = lambda z: int((z - Origin[2]) / Spacing[2])

    xml = None
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == ".xml":
            xml = file
            break

    if xml is None:
        print(RuntimeError("no xml!"))
        return

    with open(os.path.join(path, xml)) as f:
        et = parse(f)

        root = et.getroot()
        xmlns = "http://www.nih.gov"
        sessions = root.findall("{%s}readingSession" % xmlns)
        for session in sessions:
            rad_id = session.find("{%s}servicingRadiologistID" % xmlns).text
            nodules = session.findall("{%s}unblindedReadNodule" % xmlns)

            for nodule in nodules:
                nodule_id = nodule.find("{%s}noduleID" % xmlns).text
                print(f"Dealing nodule {nodule_id} from {rad_id}.")

                char = nodule.find("{%s}characteristics" % xmlns)
                if char is not None:
                    subtlety = char.find("{%s}malignancy" % xmlns)
                    if subtlety is None:
                        continue
                    else:
                        subtlety = int(subtlety.text)
                else:
                    continue

                rois = nodule.findall("{%s}roi" % xmlns)
                images = []
                masks = []

                xsum = 0
                ysum = 0
                count = 0
                for roi in rois:
                    if roi is None or len(roi.findall("{%s}edgeMap" % xmlns)) <= 1:
                        break

                    z = roi.find("{%s}imageZposition" % xmlns).text
                    z = tranz(float(z))
                    image = np.zeros((512, 512))

                    minx, miny, maxx, maxy = 512, 512, 0, 0
                    for edge in roi.findall("{%s}edgeMap" % xmlns):
                        x = int(edge.find("{%s}xCoord" % xmlns).text)
                        y = int(edge.find("{%s}yCoord" % xmlns).text)
                        image[y, x] = 1
                        xsum += y
                        ysum += x
                        count += 1
                        if maxx < x: maxx = x
                        if maxy < y: maxy = y
                        if minx > x: minx = x
                        if miny > y: miny = y

                    mask = np.zeros((512, 512))
                    for i in range(miny, maxy + 1):
                        for j in range(minx, maxx + 1):
                            if image[i, j] == 1:
                                for k in range(maxx, minx - 1, -1):
                                    if image[i, k] == 1:
                                        for p in range(j, k):
                                            mask[i, p] = 1

                    for i in range(minx, maxx + 1):
                        for j in range(miny, maxy + 1):
                            if image[j, i] == 1:
                                for k in range(maxy, miny - 1, -1):
                                    if image[k, i] == 1:
                                        for p in range(j, k):
                                            mask[p, i] = 1 if mask[p, i] == 1 else 0

                    images.append(img_array[z])
                    masks.append(mask)

                else:
                    point = int(xsum / count), int(ysum / count)
                    yield images, masks, point, subtlety

            # plt.subplot(122)
            # plt.imshow(mask, cmap="gray")
            # plt.show()


def cut_image(images, masks, point, cut=32):
    if point[0] <= 32: point = (33, point[1])
    if point[1] <= 32: point = (point[0], 33)
    if point[0] >= 512 - 32: point = (512 - 33, point[1])
    if point[1] >= 512 - 32: point = (point[0], 512 - 33)

    for i in range(len(images)):
        images[i] = images[i][point[0] - cut:point[0] + cut, point[1] - cut:point[1] + cut]
        masks[i] = masks[i][point[0] - cut:point[0] + cut, point[1] - cut:point[1] + cut]

    return images, masks


def get_statistic(image, door=0.7):
    ps = {}
    point = image.shape[0] // 2, image.shape[1] // 2

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            le = i
            # le = int(math.sqrt((point[0] - i) ** 2 + (point[1] - j) ** 2))
            # le = (i + j) // 10
            if le not in ps:
                ps[le] = 0
            if image[i, j] > door:
                ps[le] += 1

    return ps


def get_malignancy(root, out_path):
    filenames = os.listdir(root)

    with open(os.path.join(out_path, "annotations.csv"), "w") as fs:
        writer = csv.writer(fs)
        count = 0
        for idx, filess in enumerate(filenames, 0):
            for files in os.listdir(os.path.join(root, filess)):
                for file in os.listdir(os.path.join(root, filess, files)):
                    path = os.path.join(root, filess, files, file)
                    print(f"Opening file {path}.")

                    xml = None
                    for f in os.listdir(path):
                        if os.path.splitext(f)[1] == ".xml":
                            xml = f
                            break

                    if xml is None:
                        print(RuntimeError("no xml!"))
                        continue

                    with open(os.path.join(path, xml)) as f:
                        et = parse(f)

                        base = et.getroot()
                        xmlns = "http://www.nih.gov"
                        sessions = base.findall("{%s}readingSession" % xmlns)
                        for session in sessions:
                            rad_id = session.find("{%s}servicingRadiologistID" % xmlns).text
                            nodules = session.findall("{%s}unblindedReadNodule" % xmlns)

                            for nodule in nodules:
                                nodule_id = nodule.find("{%s}noduleID" % xmlns).text
                                print(f"Dealing nodule {nodule_id} from {rad_id}.")

                                char = nodule.find("{%s}characteristics" % xmlns)
                                if char is not None:
                                    subtlety = char.find("{%s}malignancy" % xmlns)
                                    if subtlety is None:
                                        continue
                                    else:
                                        subtlety = int(subtlety.text)
                                else:
                                    continue

                                rois = nodule.findall("{%s}roi" % xmlns)

                                if rois is None:
                                    continue

                                for roi in rois:
                                    if len(roi.findall("{%s}edgeMap" % xmlns)) <= 1:
                                        continue

                                    writer.writerow([count, subtlety])
                                    count += 1


if __name__ == "__main__":
    # for ims, mks, p, sub in get_xml(
    #         r"E:\pythonProject\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"):
    #
    #     images, masks = cut_image(ims, mks, p)
    #
    #     for i, m in zip(images, masks):
    #         plt.subplot(121)
    #         plt.imshow(i, cmap="gray")
    #         plt.subplot(122)
    #         plt.imshow(m, cmap="gray")
    #         plt.show()

    # get_malignancy("D:\Data_Pro", "D:\Medical Image\divided data\cut")
    pass