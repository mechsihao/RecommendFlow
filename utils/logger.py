"""
创建人：李思浩（80302421）
创建时间：2022/02/13
功能描述：日志工具
"""

import logging
import sys
from os import makedirs
from os.path import dirname, exists
from logging.handlers import TimedRotatingFileHandler
import re

loggers = {}

log_path = f''  # 日志文件路径
LOG_ENABLED = True  # 是否开启日志
LOG_TO_CONSOLE = True  # 是否输出到控制台
LOG_TO_FILE = True  # 是否输出到文件

LOG_LEVEL = 'DEBUG'  # 日志级别
# 每条日志输出格式
LOG_FORMAT = '%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d - %(message)s'


def get_logger(name=None):
    """
    get logger by name
    :param name: name of logger
    :return: logger
    """
    global loggers

    if not name:
        name = __name__

    if loggers.get(name):
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # 输出到控制台
    if LOG_ENABLED and LOG_TO_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 输出到文件
    if LOG_ENABLED and LOG_TO_FILE:

        # interval 滚动周期，
        # when="MIDNIGHT", interval=1 表示每天0点为更新点，每天生成一个文件
        # backupCount  表示日志保存个数
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
        )
        # filename="mylog" suffix设置，会生成文件名为mylog.2020-02-25.log
        file_handler.suffix = "%Y-%m-%d.log"
        # extMatch是编译好正则表达式，用于匹配日志文件名后缀

        # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")

        # 定义日志输出格式
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        logger.addHandler(file_handler)

        # 如果路径不存在，创建日志文件文件夹
        log_dir = dirname(log_path)
        if not exists(log_dir):
            makedirs(log_dir)

        # 添加 FileHandler
        # file_handler = logging.FileHandler(log_path, encoding='utf-8')
        # file_handler.setLevel(level=LOG_LEVEL)
        # formatter = logging.Formatter(LOG_FORMAT)

    # 保存到全局 loggers
    loggers[name] = logger
    return logger
