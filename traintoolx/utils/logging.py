

import time
import os
import sys
import colorama
from colorama import init
import traintoolx
import paddle

init(autoreset=True)
levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}


def log(level=2, message="", use_color=False):
    if paddle.distributed.get_rank() == 0:
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if traintoolx.log_level >= level:
            if use_color:
                print("\033[1;31;40m{} [{}]\t{}\033[0m".format(
                    current_time, levels[level], message).encode("utf-8")
                      .decode("latin1"))
            else:
                print("{} [{}]\t{}".format(current_time, levels[
                    level], message).encode("utf-8").decode("latin1"))
            sys.stdout.flush()


def debug(message="", use_color=False):
    log(level=3, message=message, use_color=use_color)


def info(message="", use_color=False):
    log(level=2, message=message, use_color=use_color)


def warning(message="", use_color=True):
    log(level=1, message=message, use_color=use_color)


def error(message="", use_color=True, exit=True):
    log(level=0, message=message, use_color=use_color)
    if exit:
        sys.exit(-1)
