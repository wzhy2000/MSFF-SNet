import torch
import numpy as np
from skimage.filters import gaussian


def post_process_output(able_pred, angle_pred, width_pred,class_pred):
    """
    :param able_pred:  (1, 2, 320, 320)      (as torch Tensors)
    :param angle_pred: (1, angle_k, 320, 320)     (as torch Tensors)
    """

    # 抓取置信度
    able_pred = able_pred.squeeze().cpu().numpy()    # (320, 320)
    able_pred = gaussian(able_pred, 1.0, preserve_range=True)

    # 抓取角
    angle_pred = np.argmax(angle_pred.cpu().numpy().squeeze(), 0)   # (320, 320)

    class_confidence = np.max(class_pred.cpu().numpy().squeeze(), 0)
    class_confidence = gaussian(class_confidence, 1.0, preserve_range=True)

    class_pred = np.argmax(class_pred.cpu().numpy().squeeze(), 0)

    # 抓取宽度
    width_pred = width_pred.squeeze().cpu().numpy() * 1000.  # (320, 320)
    width_pred = gaussian(width_pred, 1.0, preserve_range=True)

    return able_pred, angle_pred, width_pred, class_pred,class_confidence

