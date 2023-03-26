#! /usr/bin/env python

usage = """%prog Version of 5th October 2008

(c) Mark Johnson

Maps br-syll.txt to the input required by py-cfg.

usage: %prog [options]"""

import optparse, re, sys
import lx
from signal import signal, SIGPIPE, SIG_DFL
# ignore the SIGPIPE signal and stop raise exception
signal(SIGPIPE,SIG_DFL) 

def file_brentformat(inf, outf, mapper):
    for line in inf:
        segs = (mapper.get(c, c) for c in line.strip()+' ')
        outf.write(' '.join(seg for seg in segs if seg))
        outf.write('\n')


if __name__ == "__main__":
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-s", "--syllable-boundaries", dest="syllable_boundaries",
                      help="map syllable boundaries to this value")
    parser.add_option("-w", "--word-boundaries", dest="word_boundaries",
                      help="map word boundaries to this value")
    (options,args) = parser.parse_args()
    mapper = {}
    mapper[' '] = options.word_boundaries
    mapper['\t'] = options.word_boundaries
    mapper['.'] = options.syllable_boundaries
    file_brentformat(sys.stdin, sys.stdout, mapper)

