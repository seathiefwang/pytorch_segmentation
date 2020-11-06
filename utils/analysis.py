import os
import cv2
import numpy as np

root = "../../rematch/"
filename = "train_all.txt"

class_weight = {1:0.07, 2:0.07, 3:0.07, 4:0.02, 7:0.07, 8:0.07, 9:0.07, 10:0.07, 11:0.07, 12:0.07, 13:0.07, 14:0.07, 15:0.07, 16:0.07, 17:0.07}

class_sum = {2: 1543114305, 7: 1311472725, 11: 2476695927, 13: 3064837353, 17: 473091372, 3: 2371671666, 9: 4478857437, 16: 221682972, 10: 247614603, 14: 1439428722, 8: 52628364, 12: 470929422, 1: 541517850, 15: 966983106, 4: 274176}
class_num = {2: 73904, 7: 28175, 11: 75679, 13: 73239, 17: 65397, 3: 64540, 9: 47947, 16: 7874, 10: 12158, 14: 44790, 8: 3101, 12: 15291, 1: 32271, 15: 20300, 4: 14}

with open(root + filename, 'r') as f:
    for line in f.readlines():
        label = cv2.imread(root+'label/'+line.strip()+'.png')

        key = np.unique(label)

        weight = 0
        for k in key:
            mask = label == k
            label_n = np.sum(mask)
            weight += class_weight[k] * (1 / class_num[k])
        print(weight)
        break




# src_dir = "../../rematch/label/"
# pixel_num = {}
# class_num = {}

# for i, filename in enumerate(os.listdir(src_dir)):
#     # print(i, filename)
#     label = cv2.imread(src_dir + filename)

#     key = np.unique(label)
#     for k in key:
#         mask = label == k
#         s = np.sum(mask)
#         if k in pixel_num:
#             pixel_num[k] += s
#             class_num[k] += 1
#         else:
#             pixel_num[k] = s
#             class_num[k] = 1
#     # break

# lbl = pixel_num.keys()
# res = np.asarray(list(pixel_num.values()))

# res_sum = np.sum(res) / 100

# print(sorted(lbl))
# print(pixel_num)
# print(class_num)
# print(res / res_sum)

# [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# {2: 1543114305, 7: 1311472725, 11: 2476695927, 13: 3064837353, 17: 473091372, 3: 2371671666, 9: 4478857437, 16: 221682972, 10: 247614603, 14: 1439428722, 8: 52628364, 12: 470929422, 1: 541517850, 15: 966983106, 4: 274176}
# [7.84868523e+00 6.67049522e+00 1.25971269e+01 1.55885689e+01
#  2.40626715e+00 1.20629459e+01 2.27806470e+01 1.12753790e+00
#  1.25943300e+00 7.32131308e+00 2.67681702e-01 2.39527090e+00
#  2.75430222e+00 4.91833041e+00 1.39453125e-03]