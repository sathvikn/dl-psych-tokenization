#! /usr/bin/env python

usage = """%prog Version of 24th November 2008

(c) Mark Johnson

Reads the .score files produced by Sharon's program and writes one line per file

filename word-fscore syllable-fscore

usage: %prog [options]"""

import optparse, re, sys

score_rex = re.compile(r"""Results (?:[(]words[)] )?for (?P<filename>[^ \n:]+):
P [0-9.]+ R [0-9.]+ F (?P<wordf>[0-9.]+) BP [0-9.]+ BR [0-9.]+ BF [0-9.]+ LP [0-9.]+ LR [0-9.]+ LF [0-9.]+ 
Avg word length: [0-9.]+ [(]true[)], [0-9.]+ [(]found[)]
(?:Results [(]syllables[)] for (?P=filename):
P [0-9.]+ R [0-9.]+ F (?P<syllf>[0-9.]+) BP [0-9.]+ BR [0-9.]+ BF [0-9.]+ LP [0-9.]+ LR [0-9.]+ LF [0-9.]+ 
Avg word length: [0-9.]+ [(]true[)], [0-9.]+ [(]found[)])?""")

def extract_scores(txt):
    for mo in score_rex.finditer(txt):
        print mo.group('filename'), mo.group('wordf'), mo.group('syllf')

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    (options,args) = parser.parse_args()

    if len(args) == 0:
        extract_scores(sys.stdin.read())
    else:
        for fname in args:
            extract_scores(file(fname,"rU").read())
            


