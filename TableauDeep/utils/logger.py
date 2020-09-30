# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/9/29

import logging


def get_logger(logger_name=None):
    if logger_name:
        return logging.getLogger(f"TableauDeep - {logger_name}")
    else:
        return logging.getLogger(f"TableauDeep")