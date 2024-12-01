from __future__ import annotations
import logging
import os
import re
import sys
import inspect
from typing import Any

from .platform import PLATFORM, OS_NAME, SYS_ARCH
from .resources import machine


# Global verbose
class Verbose(object):
    def __init__(self):
        self.__verbose = 10

    @property
    def val(self):
        """

        :return:
        """
        return self.__verbose

    @val.setter
    def val(self, val: int):
        """

        :param val:
        :return:
        """

        self.__verbose = val


__verbose = Verbose()
__exit_on_error = False
__framework_name__ = "master-thesis"


def set_exit_on_error(val: bool):
    global __exit_on_error
    __exit_on_error = val


def get_exit_on_error():
    return __exit_on_error


def get_global_verbose():
    return __verbose.val


def set_global_verbose(val: int):
    global __verbose
    __verbose.val = val


# Print iterations progress
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress_bar(iteration: int, total: int, prefix='', suffix='', decimals=1, length=20, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    assert isinstance(iteration, int)
    assert isinstance(total, int)
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


class Formatter(logging.Formatter):
    def __init__(self, msg):
        """

        :param msg:
        """
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        """

        :param record:
        :return:
        """

        record.msg = reformat_comment(record.msg, add_tab=True)

        return logging.Formatter.format(self, record)


class StreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        """

        :param record:
        :return:
        """
        super(StreamHandler, self).emit(record)

        if get_exit_on_error() and record.levelno == logging.ERROR:
            sys.exit(1)


def check_verbosity(kwargs: dict):
    """

    :param kwargs:
    :return:
    """
    verbose = None

    if "verbose" in kwargs.keys():
        verbose = kwargs.pop("verbose")

    if verbose is not None:
        if verbose > get_global_verbose():
            return False

    return True


# Custom logger class with multiple destinations
class Logger(logging.Logger):
    FORMAT = "[%(levelname)-7s %(asctime)s] %(message)s "

    def __init__(self, name, level=logging.INFO):
        """

        :param name:
        :param level:
        """
        logging.Logger.__init__(self, name, level)

        self.formatter = Formatter(self.FORMAT)
        console = StreamHandler()
        console.setFormatter(self.formatter)

        self.addHandler(console)

    def add_file_handler(self, filename: str, remove=False):
        if remove:
            if os.path.exists(filename) and os.path.isfile(filename):
                os.remove(filename)
        file_handler = logging.FileHandler(f"{filename}")
        file_handler.setFormatter(self.formatter)
        self.addHandler(file_handler)

    def info(self, msg, *args, **kwargs):
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        record = inspect.stack()[1]  # 0 represents this line
        # # 1 represents line at caller
        frame = record[0]
        info = inspect.getframeinfo(frame)
        stack_info = f"({os.path.basename(info.filename)}-{info.lineno})"
        msg = f"{msg} {stack_info}"
        if check_verbosity(kwargs):
            super(Logger, self).info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        record = inspect.stack()[1]  # 0 represents this line
        # # 1 represents line at caller
        frame = record[0]
        info = inspect.getframeinfo(frame)
        stack_info = f"({os.path.basename(info.filename)}-{info.lineno})"
        msg = f"{msg} {stack_info}"
        if check_verbosity(kwargs):
            super(Logger, self).warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        record = inspect.stack()[1]  # 0 represents this line
        # # 1 represents line at caller
        frame = record[0]
        info = inspect.getframeinfo(frame)
        stack_info = f"({os.path.basename(info.filename)}-{info.lineno})"
        msg = f"{msg} {stack_info}"
        if check_verbosity(kwargs):
            super(Logger, self).error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        record = inspect.stack()[1]  # 0 represents this line
        # # 1 represents line at caller
        frame = record[0]
        info = inspect.getframeinfo(frame)
        stack_info = f"({os.path.basename(info.filename)}-{info.lineno})"
        msg = f"{msg} {stack_info}"
        if check_verbosity(kwargs):
            super(Logger, self).debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """

        :param msg:
        :param args:
        :param kwargs:
        :return:
        """
        record = inspect.stack()[1]  # 0 represents this line
        # # 1 represents line at caller
        frame = record[0]
        info = inspect.getframeinfo(frame)
        stack_info = f"({os.path.basename(info.filename)}-{info.lineno})"
        msg = f"{msg} {stack_info}"
        if check_verbosity(kwargs):
            super(Logger, self).critical(msg, *args, **kwargs)

    def warn(
            self,
            *args,
            **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    def dashed_line(self, verbose=0):
        """

        :return:
        """
        if verbose <= get_global_verbose():
            print("-" * 40, file=sys.stderr)


logging.setLoggerClass(Logger)

# Global logger
__logger__ = logging.Logger(__framework_name__)
logger = logging.getLogger(__logger__.name)
logger.setLevel(logging.DEBUG)
logger: Logger


def reformat_comment(comment: str, add_tab=False):
    """

    :param comment:
    :param add_tab
    :return:
    """

    comment = re.sub(r'\s+', ' ', comment)
    sentence_len = 0
    tmp = []
    for word in comment.split(' '):

        if sentence_len >= 70:
            sentence_len = len(word)
            tmp.append(os.linesep)

        else:
            sentence_len += len(word)

        tmp.append(word)
    comment = " ".join(tmp)

    res = []
    more_than_one_line = False
    for line in comment.split(os.linesep):
        if add_tab and more_than_one_line:
            res.append("\t  {}".format(line.lstrip()))
        else:
            res.append(line.lstrip())
            more_than_one_line = True
    comment = "{}".format(os.linesep).join(res)

    return comment


# logger.dashed_line()
# logger.info("Platform : {}".format(PLATFORM.platform()))
# logger.info("OS Name : {}".format(OS_NAME))
# logger.info("Architecture : {}".format(SYS_ARCH))
# logger.info(f"Verbose : {get_global_verbose()}")
# logger.info("Machine : ")
# print(machine, file=sys.stderr)
# logger.dashed_line()

__all__ = ["logger", "get_global_verbose",
           "set_global_verbose",
           "set_exit_on_error",
           "progress_bar",
           "reformat_comment"]
