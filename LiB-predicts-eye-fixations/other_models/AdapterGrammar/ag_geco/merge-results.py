#! /usr/bin/env python

usage = """%prog Version of 18th November 2008

(c) Mark Johnson

Extracts values from several results files and writes them into an R data file

usage: %prog [options] [filename fieldno]*"""

import optparse, re, sys

ofs = "\t"
outf = sys.stdout

def readfield(inf, fieldno, skip=0):
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

def process(inf, options, key):
    for lineno,field in enumerate(readfield(inf, options.field)):
        if options.lineno != 0:
            outf.write("%s%s"%(lineno*options.lineno,ofs))
        outf.write(field)
        if key:
            outf.write('%s%s'%(ofs,key))
        outf.write('\n')

            
if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-f", "--field", dest="field", type="int", default=2,
                      help="field number to extract")
    parser.add_option("-l", "--lineno", dest="lineno", type="int", default=10,
                      help="if non-zero, print this times line number")
    parser.add_option("-k", "--key_re", dest="key_re", default=r"_G([^_]+)_",
                      help="regex used to map filenames to plot keys")
    parser.add_option("-s", "--key_subst", dest="key_subst", default=r"\1",
                      help="substitution pattern used to map filenames to plot keys")    
    (options,args) = parser.parse_args()

    if options.lineno != 0:
        outf.write("epoch"+ofs)
    outf.write("fscore")
    if len(args) > 0:
        outf.write(ofs+"key")
    outf.write('\n')
    
    if len(args) > 0:
        key_re = re.compile(options.key_re)
        key_subst = options.key_subst
        for fname in args:
            if key_re and key_subst:
                mo = key_re.search(fname)
                if mo:
                    key = mo.expand(key_subst)
                else:
                    key = fname
            else:
                key = fname
            process(file(fname, "rU"), options, key)
    else:
        process(sys.stdin, options, None)
