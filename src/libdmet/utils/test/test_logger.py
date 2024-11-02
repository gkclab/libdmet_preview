#! /usr/bin/env python

from libdmet.utils import logger as log

def test_logger():
    Level = log.Level
    log.result("Logger Levels: %s", Level)
    log.warning("Logger Levels: %s", Level)
    log.info("Logger Levels: %s", Level)
    log.debug(0, "Logger Levels: %s", Level)
    log.debug(1, "Logger Levels: %s", Level)
    log.debug(2, "Logger Levels: %s", Level)
    log.error("Logger Levels: %s", Level)
    log.fatal("Logger Levels: %s", Level)

if __name__ == "__main__":
    test_logger()
