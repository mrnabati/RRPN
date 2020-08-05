################################################################################
## Date Created  : Sat Jun 15 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : July 11th, 2019                                            ##
## Copyright (c) 2019                                                         ##
################################################################################

import logging
import time

def initialize_logger(name, level='INFO'):
    ## Set up logger
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        ch = logging.StreamHandler()
        # if verbose:
        #     ch.setLevel(logging.DEBUG)
        # else:
        #     ch.setLevel(logging.INFO)
            
        formatter = ColorFormatter(fmt='%(filename)s:%(lineno)d %(levelname)s:: %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class ColorFormatter(logging.Formatter):
    DEBUG     = "\033[34;1mDEBUG\033[0;34;49m: "
    INFO     =  "\033[1;32mINFO\033[0;32;49m: "
    WARN     = "\033[1;33mWARNING\033[0;33;49m: "
    ERROR    = "\033[103;31mERROR\033[31;49m: "
    CRITICAL = "\033[41;97mCRITICAL\033[91;49;1m: "
    RESET = "\033[0m"

    def __init__(self, fmt='%(filename)s:%(lineno)d %(levelname)s:: %(message)s'):
        logging.Formatter.__init__(self, fmt=fmt, datefmt=None, style='%')
    
    def format(self, record):

        curTime = time.localtime()
        curTime = time.strftime("%m/%d/%Y %X", curTime)

        long_prefix = '[{time} {fileName}:{line}] '.format(
            time=curTime,
            fileName=record.filename,
            line=record.lineno
        )
        short_prefix = ''

        if record.levelno == logging.DEBUG:
            ret_s =  self.DEBUG + long_prefix + record.msg + self.RESET
        elif record.levelno == logging.INFO:
            ret_s = self.INFO + short_prefix +  record.msg + self.RESET
        elif record.levelno == logging.WARN:
            ret_s = self.WARN + short_prefix + record.msg + self.RESET
        elif record.levelno == logging.ERROR:
            ret_s = self.ERROR + long_prefix + record.msg + self.RESET
        elif record.levelno == logging.CRITICAL:
            ret_s = self.CRITICAL + long_prefix + record.msg + self.RESET
        
        return ret_s

def test_logger():
    logger = initialize_logger('test', level='DEBUG')
    logger.debug('This is debug')
    logger.info('This is info')
    logger.warning('This is warning')
    logger.error('This is error')
    logger.critical('This is critical')

if __name__ == '__main__':
    test_logger()