# -*- coding: utf-8 -*-
"""
@ Time ： 2020/3/2 11:33
@ Auth ： wangdx
@ File ：demo.py
@ IDE ：PyCharm
@ Function :
"""

import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import math
from utils.affga import AFFGA

# cls_dic = {1: 'cup', 2: 'notebook', 3: 'box', 4: 'pen', 5: 'screwdriver', 6: 'toothpaste', 7: 'tape', 8: 'wrench', 9: 'stapler', 10: 'knife', 11: 'pliers', 12: 'headset', 13: 'towel', 14: 'wrist developer', 15: 'mouse', 16: 'mobile phone', 17: 'umbrella', 18: 'toothbrush', 19: 'wallet', 20: 'banana', 21: 'paper', 22: 'charger', 23: 'remote controller', 24: 'glasses', 25: 'shaver', 26: 'card', 27: 'watch'}
cls_dic = {1: 'cup', 2: 'notebook',  3: 'pen', 4: 'screwdriver', 5: 'toothpaste', 6: 'tape', 7: 'wrench', 8: 'stapler', 9: 'knife', 10: 'pliers', 11: 'headset', 12: 'towel', 13: 'wrist developer', 14: 'mouse', 15: 'mobile phone', 16: 'umbrella', 17: 'toothbrush', 18: 'wallet', 19: 'banana', 20: 'charger', 21: 'remote controller', 22: 'glasses', 23: 'shaver', 24: 'card', 25: 'watch'}

def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi


def drawGrasps(img, grasps, mode):
    """
    绘制grasp
    file: img路径
    grasps: list()	元素是 [row, col, angle, width]
    mode: arrow / region
    """
    assert mode in ['arrow', 'region']

    num = len(grasps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 0)
    thickness = 2

    for i, grasp in enumerate(grasps):
        row, col, angle, width, point_class = grasp

        if mode == 'arrow':
            width = width / 2
            angle2 = calcAngle2(angle)
            k = math.tan(angle)
            if k == 0:
                dx = width
                dy = 0
            else:
                dx = k / abs(k) * width / pow(k ** 2 + 1, 0.5)
                dy = k * dx

            if angle < math.pi:
                cv2.arrowedLine(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 3, 8, 0, 0.5)
            else:
                cv2.arrowedLine(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 3, 8, 0, 0.5)

            if angle2 < math.pi:
                cv2.line(img, (col, row), (int(col + dx), int(row - dy)), (0, 0, 255), 3)
            else:
                cv2.line(img, (col, row), (int(col - dx), int(row + dy)), (0, 0, 255), 3)

            # if point_class == 1:
            #     text = 'SS'
            # elif point_class == 2:
            #     text = 'SL'
            # elif point_class == 3:
            #     text = 'LS'
            # elif point_class == 4:
            #     text = 'LL'
            text='88'
            if int(point_class) in cls_dic:
                text = cls_dic[point_class]
            cv2.putText(img, text, (int(col + dx), int(row - dy)), font, font_scale, color, thickness, cv2.LINE_AA)
            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            cv2.circle(img, (col, row), 2, (color_b, color_g, color_r), -1)

        else:
            color_b = 255 / num * i
            color_r = 0
            color_g = -255 / num * i + 255
            img[row, col] = [color_b, color_g, color_r]

    return img


def drawRect(img, rect):
    """
    绘制矩形
    rect: [x1, y1, x2, y2]
    """
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)


if __name__ == '__main__':
    # 模型路径
    model = 'output/models/240307_2048_train2/epoch_0044_iou_0.8835_class_0.8625_sort_0.8207_mAPg_0.7952_total_1004'
    input_path = '/home/plusmile/Desktop/affga-nets/dataset/VMRD_IMG'
    output_path = 'demo/output'

    # 运行设备
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    # 初始化
    affga = AFFGA(model, device=device_name)
    with torch.no_grad():
        for file in os.listdir(input_path):

            print('processing ', file)

            img_file = os.path.join(input_path, file)
            img = cv2.imread(img_file)

            # grasps, x1, y1 = affga.predict(img, device, mode='peak', thresh=0.3, peak_dist=30)  # 预测
            # im_rest = drawGrasps(img, grasps, mode='arrow')  # 绘制预测结果
            # rect = [x1, y1, x1 + 608, y1 + 608]
            # drawRect(im_rest, rect)
            im_rest = affga.maps(img, device)  # 预测
            im_rest = cv2.cvtColor(im_rest,cv2.COLOR_HSV2BGR)
            # 保存
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            save_file = os.path.join(output_path, file)
            cv2.imwrite(save_file, im_rest)

    print('FPS: ', affga.fps())
