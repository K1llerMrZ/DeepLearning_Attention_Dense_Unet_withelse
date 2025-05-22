import os
import random

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
# from lungmask import mask
from function import *


def divide_ct(ct_root, nodules_file, out_root, out_empty=4, test=1 / 6):
    filenames = findAllMhd(ct_root)
    test_begin = int(len(filenames) - len(filenames) * test)

    nodules_origin_datas = {}
    with open(nodules_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            if row[0] not in nodules_origin_datas:
                nodules_origin_datas[row[0]] = []
            nodules_origin_datas[row[0]].append([float(x) for x in (row[1], row[2], row[3], row[4])])
    print("begin divide")

    for idx, file in enumerate(filenames, 0):
        ct_name = os.path.splitext(os.path.split(file)[1])[0]  # 获取ct编号

        if ct_name not in nodules_origin_datas:
            continue

        itkimage = sitk.ReadImage(file)  # 读取.mhd文件
        img_mask = mask.LMInferer().apply(itkimage)

        out_path = os.path.join(out_root, "train" if idx < test_begin else "test")

        Origin = itkimage.GetOrigin()
        Spacing = itkimage.GetSpacing()
        ct_sacn = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取

        img_mask[img_mask == 2] = 1

        a = np.min(ct_sacn[img_mask > 0])
        k = np.max(ct_sacn[img_mask > 0]) - a

        ct_sacn = ((ct_sacn - a) / k) * img_mask

        nod_counter = 0
        nodules_datas = {}

        if ct_name in nodules_origin_datas:
            nodules = nodules_origin_datas[ct_name]
            for nod in nodules:
                x, y, z = int((nod[0] - Origin[0]) / Spacing[0]), int((nod[1] - Origin[1]) / Spacing[1]), \
                          int((nod[2] - Origin[2]) / Spacing[2])
                radius = int(nod[3] / Spacing[0] / 2)

                if z not in nodules_datas:
                    nodules_datas[z] = []
                nodules_datas[z].append([x, y, radius])

            for i in nodules_datas:
                save_name = os.path.join(out_path, ct_name + "-" + str(i))

                with open(save_name + ".csv", "w", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(nodules_datas[i])

                plt.imsave(save_name + ".png", ct_sacn[i], cmap='gray')
                print("deal %.4f (%4d in %4d)" % (idx / len(filenames), idx, len(filenames)), "write %5d" % i, "in",
                      ct_name, "with nodule(s) succeed!")
                nod_counter += 1

            pages = list(range(len(ct_sacn)))
            for i in nodules_datas:
                pages.remove(i)

            random.shuffle(pages)
            for i in pages[:nod_counter * out_empty]:
                save_name = os.path.join(out_path, ct_name + "-" + str(i))
                with open(save_name + ".csv", "w", newline=''): pass
                plt.imsave(save_name + ".png", ct_sacn[i], cmap='gray')
                print("deal %.4f (%4d in %4d)" % (idx / len(filenames), idx, len(filenames)), "write %5d" % i, "in",
                      ct_name, "with out nodule(s) succeed!")


def divide_ct_3(ct_root, nodules_file, out_root, out_empty=4, test=1 / 6):
    filenames = findAllMhd(ct_root)
    test_begin = int(len(filenames) - len(filenames) * test)

    nodules_origin_datas = {}
    with open(nodules_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            if row[0] not in nodules_origin_datas:
                nodules_origin_datas[row[0]] = []
            nodules_origin_datas[row[0]].append([float(x) for x in (row[1], row[2], row[3], row[4])])
    print("begin divide")

    for idx, file in enumerate(filenames, 0):
        ct_name = os.path.splitext(os.path.split(file)[1])[0]  # 获取ct编号

        if ct_name not in nodules_origin_datas:
            continue

        itkimage = sitk.ReadImage(file)  # 读取.mhd文件
        img_mask = mask.LMInferer().apply(itkimage)

        out_path = os.path.join(out_root, "train" if idx < test_begin else "test")
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        Origin = itkimage.GetOrigin()
        Spacing = itkimage.GetSpacing()
        ct_sacn = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取

        img_mask[img_mask == 2] = 1

        a = np.min(ct_sacn[img_mask > 0])
        k = np.max(ct_sacn[img_mask > 0]) - a

        ct_sacn = ((ct_sacn - a) / k) * img_mask

        nod_counter = 0
        nodules_datas = {}

        if ct_name in nodules_origin_datas:
            nodules = nodules_origin_datas[ct_name]
            for nod in nodules:
                x, y, z = int((nod[0] - Origin[0]) / Spacing[0]), int((nod[1] - Origin[1]) / Spacing[1]), \
                          int((nod[2] - Origin[2]) / Spacing[2])
                radius = int(nod[3] / Spacing[0] / 2)

                if z not in nodules_datas:
                    nodules_datas[z] = []
                nodules_datas[z].append([x, y, radius])

            for i in nodules_datas:
                save_name = os.path.join(out_path, ct_name + "-" + str(i))

                with open(save_name + ".csv", "w", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(nodules_datas[i])

                imageidx = i
                if i == 0:
                    imageidx += 1
                if len(ct_sacn) == i + 1:
                    imageidx -= 1

                image = np.array([ct_sacn[imageidx - 1], ct_sacn[imageidx], ct_sacn[imageidx + 1]]).transpose(1, 2, 0)

                plt.imsave(save_name + ".png", image)
                print("deal %.4f (%4d in %4d)" % (idx / len(filenames), idx, len(filenames)), "write %5d" % i,
                      "in", ct_name, "with nodule(s) succeed!")
                nod_counter += 1


def delete_emptyct(out_path, out_num=0):
    filenames = os.listdir(out_path)

    for file in filenames:
        if os.path.splitext(file)[1] == ".csv":
            del file

    random.shuffle(filenames)

    del_num = len(filenames) - out_num
    counter = 0
    nod_num = 0

    del_list = []
    for i in range(len(filenames)):
        file = os.path.splitext(filenames[i])[0]

        if len(open(os.path.join(out_path, file + ".csv")).readlines()):
            del_list.append(i)
        else:
            nod_num += 1
            print(nod_num)

        counter += 1
        if counter == del_num:
            pass

    print("nodes:", nod_num)


def read_one(root):
    filenames = os.listdir(root)

    for i in filenames:
        if os.path.splitext(i)[1] == ".png":
            img = plt.imread(os.path.join(root, i))
            print(img)
            print(np.array(img).shape)
            break


def send_csv(root):
    filenames = os.listdir(root)

    t = []
    for file in filenames:
        ct_name = os.path.splitext(file)[0]
        if os.path.splitext(file)[1] == ".csv":
            if file == "annotations.csv":
                continue
            with open(root + "\\" + file, "r") as csv_file:
                reader = csv.reader(csv_file)
                for line in reader:
                    t.append([ct_name] + line)

    with open(root + "\\annotations.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(t)


def segmentation_image(root, out_root, begin=0, begin_file=0, test=1 / 6):
    filenames = os.listdir(root)
    test_begin = int(len(filenames) - len(filenames) * test)

    count = begin
    with open(os.path.join(out_root, "annotations.csv"), "w") as file:
        writer = csv.writer(file)

        for idx, filess in enumerate(filenames, 0):
            if idx < begin_file: continue
            for files in os.listdir(os.path.join(root, filess)):
                for file in os.listdir(os.path.join(root, filess, files)):
                    out_path = os.path.join(out_root, "train" if idx < test_begin else "test")
                    file = os.path.join(root, filess, files, file)

                    if not os.path.exists(out_path):
                        os.mkdir(out_path)
                    out_path_mask = os.path.join(out_path, "mask")
                    if not os.path.exists(out_path_mask):
                        os.mkdir(out_path_mask)
                    out_path_image = os.path.join(out_path, "image")
                    if not os.path.exists(out_path_image):
                        os.mkdir(out_path_image)

                    for ims, mks, p, sub in get_xml(file):
                        images, masks = cut_image(ims, mks, p)

                        for i, m in zip(images, masks):
                            plt.imsave(os.path.join(out_path_image, f"{count:0>6d}-image.png"), i, cmap="gray")
                            plt.imsave(os.path.join(out_path_mask, f"{count:0>6d}-mask.png"), m, cmap="gray")
                            writer.writerow([count, sub])
                            print(f"succeed save num {count:0>6d} from {filess}")
                            count += 1


def delect_error_data(root, door):
    path_mask = os.path.join(root, "mask")
    path_image = os.path.join(root, "image")

    count = 0

    names = os.listdir(path_mask)
    for file in names:
        file_id = file[:6]
        mask = plt.imread(os.path.join(path_mask, file))
        mask = mask.transpose(2, 0, 1)[0]

        mask[mask != 0] = 1
        num = np.sum(mask)

        if num < door or mask.shape != (64, 64):
            os.remove(os.path.join(path_mask, file_id + "-mask.png"))
            os.remove(os.path.join(path_image, file_id + "-image.png"))
            count += 1

    return count


def delete_train_test():
    door = 23
    n = delect_error_data(r"..\..\Medical Image\divided data\cut\train", door)
    print(f"succeed remove {n} train file!")
    n = delect_error_data(r"..\..\Medical Image\divided data\cut\test", door)
    print(f"succeed remove {n} test file!")


if __name__ == "__main__":
    # divide_ct(r"F:\Medical Image\Data",
    #           r"F:\Medical Image\annotations.csv",
    #           r"F:\Medical Image\divided data", 4)

    # delete_emptyct(r"F:\Medical Image\divided data")

    # read_one(r"F:\Medical Image\divided data\t")

    # send_csv(r"F:\Medical Image\divided data\train")
    # send_csv(r"F:\Medical Image\divided data\test")

    # divide_ct_3(r"F:\Medical Image\Data",
    #             r"F:\Medical Image\annotations.csv",
    #             r"F:\Medical Image\divided data", 4)

    # segmentation_image(r"..\..\Data_pro", r"..\..\Medical Image\divided data\cut", 0, 0)
    delete_train_test()
    pass
