#! /usr/bin/env python

usage = """%prog Version of 12th April 2008

(c) Mark Johnson

usage: %prog [options]"""

import optparse
import sys

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    (options,args) = parser.parse_args()

    assert(len(args) == 2)
    inf1 = file(args[0], "rU")
    inf2 = file(args[1], "rU")
    lines1 = (line.strip() for line in inf1 if line != "" and line[0] != "#")
    lines2 = (line.strip() for line in inf2 if line != "" and line[0] != "#")
    for line1,line2 in zip(lines1,lines2):
        print line1, line2
