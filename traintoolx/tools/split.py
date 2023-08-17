

from .dataset_split import *
from traintoolx.utils import logging


def dataset_split(dataset_dir, dataset_format, val_value, test_value,
                  save_dir):
    logging.info("Dataset split starts...")
    if dataset_format == "coco":
        train_num, val_num, test_num = split_coco_dataset(
            dataset_dir, val_value, test_value, save_dir)
    elif dataset_format == "voc":
        train_num, val_num, test_num = split_voc_dataset(
            dataset_dir, val_value, test_value, save_dir)
    elif dataset_format == "seg":
        train_num, val_num, test_num = split_seg_dataset(
            dataset_dir, val_value, test_value, save_dir)
    elif dataset_format == "imagenet":
        train_num, val_num, test_num = split_imagenet_dataset(
            dataset_dir, val_value, test_value, save_dir)
    else:
        raise Exception("Dataset format {} is not supported.".format(
            dataset_format))
    logging.info("Dataset split done.")
    logging.info("Train samples: {}".format(train_num))
    logging.info("Eval samples: {}".format(val_num))
    logging.info("Test samples: {}".format(test_num))
    logging.info("Split files saved in {}".format(save_dir))
