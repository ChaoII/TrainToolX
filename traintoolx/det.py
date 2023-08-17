
import sys
from . import cv
from .cv.models.utils.visualize import visualize_detection, draw_pr_curve
from .cv.models.utils.det_metrics.coco_utils import coco_error_analysis


message = 'Your script can be run normally only under PaddleX<2.0.0 ' + \
    'but the installed PaddleX version is greater than or equal to 2.0.0, ' + \
    'the solution is writen in the link {}, please refer to this link ro solve this issue.'.format(
        'https://github.com/PaddlePaddle/PaddleX/tree/develop/tutorials/train#%E7%89%88%E6%9C%AC%E5%8D%87%E7%BA%A7'
    )


def __getattr__(attr):
    if attr == 'transforms':

        print("\033[1;31;40m{}\033[0m".format(message).encode("utf-8")
              .decode("latin1"))
        sys.exit(-1)


visualize = visualize_detection
draw_pr_curve = draw_pr_curve
coco_error_analysis = coco_error_analysis

# detection
YOLOv3 = cv.models.YOLOv3
FasterRCNN = cv.models.FasterRCNN
PPYOLO = cv.models.PPYOLO
PPYOLOTiny = cv.models.PPYOLOTiny
PPYOLOv2 = cv.models.PPYOLOv2
PicoDet = cv.models.PicoDet

# instance segmentation
MaskRCNN = cv.models.MaskRCNN
