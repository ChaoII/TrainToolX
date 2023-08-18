from .operators import *
from .batch_operators import BatchRandomResize, BatchRandomResizeByShort, _BatchPadding, PadGT, PadRGT
from traintoolx.cv import transforms as T


def arrange_transforms(model_type, transforms, mode='train'):
    # 给transforms添加arrange操作
    if model_type == 'segmenter':
        if mode == 'eval':
            transforms.apply_im_only = True
        else:
            transforms.apply_im_only = False
        arrange_transform = ArrangeSegmenter(mode)
    elif model_type == 'classifier':
        arrange_transform = ArrangeClassifier(mode)
    elif model_type == 'detector':
        arrange_transform = ArrangeDetector(mode)
    else:
        raise Exception("Unrecognized model type: {}".format(model_type))
    transforms.arrange_outputs = arrange_transform


def build_transforms(transforms_info):
    transforms = list()
    for op_info in transforms_info:
        op_name = list(op_info.keys())[0]
        op_attr = op_info[op_name]
        if not hasattr(T, op_name):
            raise Exception("There's no transform named '{}'".format(op_name))
        transforms.append(getattr(T, op_name)(**op_attr))
    eval_transforms = T.Compose(transforms)
    return eval_transforms
