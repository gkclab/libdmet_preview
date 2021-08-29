#! /usr/bin/env python

import sys
from datetime import datetime
import libdmet

"""
Logger
"""

DASH_LINE = "----------------------------------------------------------------------------- "

WARNING_LOGO = (" W    W    AA    RRRRR   N    N  II  N    N   GGGG  ",
                " W    W   A  A   R    R  NN   N  II  NN   N  G    G ",
                " W    W  A    A  R    R  N N  N  II  N N  N  G      ",
                " W WW W  AAAAAA  RRRRR   N  N N  II  N  N N  G  GGG ",
                " WW  WW  A    A  R   R   N   NN  II  N   NN  G    G ",
                " W    W  A    A  R    R  N    N  II  N    N   GGGG  ")

NOTE_LOGO = (" N    N   OOOO   TTTTTT  EEEEEE ",
             " NN   N  O    O    TT    E      ",
             " N N  N  O    O    TT    EEEEE  ",
             " N  N N  O    O    TT    EEEEE  ",
             " N   NN  O    O    TT    E      ",
             " N    N   OOOO     TT    EEEEEE ")

Level = dict(zip("FATAL ERR SECTION RESULT WARNING INFO DEBUG0 DEBUG1 DEBUG2".split(), range(9)))

stdout = sys.stdout
verbose = "INFO"
clock = True
INDENT = 16

def __verbose():
    return Level[verbose]

def fatal(msg, *args):
    if __verbose() >= Level['FATAL']:
        __clock()
        flush("  FATAL ", msg, *args, indent = clock * INDENT)

def fassert(cond, msg, *args):
    if not cond:
        fatal(msg, *args)
        raise Exception

def error(msg, *args):
    if __verbose() >= Level['ERR']:
        __clock()
        flush("  ERROR ", msg, *args, indent = clock * INDENT)

def eassert(cond, msg, *args):
    if not cond:
        error(msg, *args)
        raise Exception

def section(msg, *args):
    if __verbose() >= Level['SECTION']:
        __clock()
        flush("####### ", msg, *args, indent = clock * INDENT)

def result(msg, *args):
    if __verbose() >= Level['RESULT']:
        __clock()
        flush("******* ", msg, *args, indent = clock * INDENT)

def print_logo(logo):
    for i, line in enumerate(logo):
        __clock()
        flush("******* ", line, indent = clock * INDENT)

def print_dash_line():
    __clock()
    flush("******* ", " ", indent = clock * INDENT)
    __clock()
    flush("******* ", DASH_LINE, indent = clock * INDENT)
    __clock()
    flush("******* ", " ", indent = clock * INDENT)

def warning(msg, *args):
    if __verbose() >= Level['WARNING']:
        print_dash_line()
        print_logo(WARNING_LOGO) 
        __clock()
        flush("******* ", " ", indent = clock * INDENT)
        __clock()
        flush("WARNING ", msg, *args, indent = clock * INDENT)
        print_dash_line()

warn = warning

def note(msg, *args):
    if __verbose() >= Level['WARNING']:
        print_dash_line()
        #print_logo(NOTE_LOGO) 
        __clock()
        flush(" ", " ", indent = clock * INDENT)
        __clock()
        flush(" NOTE ", msg, *args, indent = clock * INDENT)
        print_dash_line()

def check(cond, msg, *args):
    if not cond:
        warning(msg, *args)

def info(msg, *args):
    if __verbose() >= Level['INFO']:
        __clock()
        flush("   INFO ", msg, *args, indent = clock * INDENT)

def debug(level, msg, *args):
    if __verbose() >= Level["DEBUG0"] + level:
        __clock()
        flush("  DEBUG " + "  " * level, msg, *args, indent = clock * INDENT)

def time():
    stdout.write(datetime.now().strftime("%y %b %d %H:%M:%S") + "\n")
    stdout.flush()

def flush(msgtype, msg, *args, **kwargs):
    indent = 0
    if len(msg) > 0:
      if "indent" in kwargs:
          indent = kwargs["indent"]

      __msg = (msg % args).split('\n')
      __msg = [msgtype + line for line in __msg]
      __msg = ("\n" + " " * indent) .join(__msg)
      stdout.write(__msg)

    stdout.write('\n')
    stdout.flush()

def __clock():
    if clock:
        stdout.write(datetime.now().strftime("%b %d %H:%M:%S") + " ")

section("%s", libdmet.__doc__)

# *********************************************************************
# logger wrapper for pyscf
# *********************************************************************

class flush_for_pyscf(object):
    def __init__(self, keywords):
        self.keywords = set(keywords)

    def addkey(self, key):
        self.keywords.add(key)

    def addkeys(self, keys):
        self.keywords.union(keys)

    def has_keyword(self, args):
        for arg in map(str, args):
            for key in self.keywords:
                if key in arg:
                    return True
        return False

    def __call__(self, object, *args):
        if self.has_keyword(args):
            result(*args)

if __name__ == "__main__":
    test()
