#! /usr/bin/env python

usage = """%prog Version of 10th October 2008

(c) Mark Johnson

Cuts selected fields from files and pastes them together into a single
line.

usage: %prog [options] [filename fieldno]*"""

import optparse, re, sys

def readfield(inf, fieldno, skip):
    lineno = 0
    for line in inf:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        lineno += 1
        if lineno < skip:
            continue
        if fieldno == 0:
            yield line
        else:
            yield line.split()[fieldno-1]
            
if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-s", "--skip", dest="skip", type="int", default=0,
                      help="number of lines to skip")
    
    (options,args) = parser.parse_args()
    if len(args) % 2 != 0:
        sys.stderr.write("Expected even number of arguments")
        sys.exit(1)

    fields = (readfield(file(args[2*i], "rU"), int(args[2*i+1]), options.skip) for i in xrange(len(args)/2))
    for f in zip(*fields):
        sys.stdout.write('\t'.join(f))
        sys.stdout.write('\n')
