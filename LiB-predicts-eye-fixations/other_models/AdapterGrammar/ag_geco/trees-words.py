#! /bin/env python

usage = """%prog -- map py-cfg parse trees to syllables

Version of 14th November, 2008

(c) Mark Johnson

usage: %prog [options]

"""

import lx, tb
import optparse, re, sys

word_rex=re.compile("Word")
filter_terminal_rex=re.compile("(?:[.0]|[#][WS])+")
    
def filter_rex(strg0):
    strg = ""
    pos = 0
    for it in filter_terminal_rex.finditer(strg0):
        strg += strg0[pos:it.start()]
        pos = it.end()
    strg += strg0[pos:]
    return strg

def simplified_terminal(t):
    if len(t) > 0 and t[0] == '\\':
        return t[1:]
    else:
        return t
    
def wordseqmatchiter(rex, words):
    """Enumerates subsequences of words that when joined with ' ' match rex.

    Yields tuples (matchflag,begin,end), where matchflag is either True or False,
    and (begin,end) identify the subsequence in words"""
    wordstr = ' '.join(words)
    nwords = len(words)
    index = 0
    pos = 0
    for mo in rex.finditer(wordstr):
        pos0 = pos
        pos = mo.start()
        index0 = index
        index += len(wordstr[pos0:pos].split())
        assert(index <= nwords)
        if index0 != index:
            yield (False, index0, index)
        pos0 = pos
        pos = mo.end()
        index0 = index
        index += len(wordstr[pos0:pos].split())
        assert(index <= nwords)
        yield (True, index0, index)
    index0 = index
    index += len(wordstr[pos:].split())
    if index0 != index:
        assert(index <= nwords)
        yield (False, index0, index)


def tree_string(tree):

    def visit(node, ancmatch, morphssofar, segssofar):
        """Does a preorder visit of the nodes in the tree"""
        if tb.is_terminal(node):
            t = filter_rex(simplified_terminal(node))
            if t != "":
                segssofar.append(t)
            return morphssofar,segssofar
        for child in tb.tree_children(node):
            morphssofar,segssofar = visit(child, ancmatch, morphssofar, segssofar)
        if word_rex.match(tb.tree_label(node)):
            if segssofar != []:
                morphssofar.append(''.join(segssofar))
                segssofar = []
            morphssofar.append(' ')
        return morphssofar,segssofar

    morphssofar,segssofar = visit(tree, False, [], [])
    assert(segssofar == [])
    return ''.join(morphssofar)
    
def read_write(inf, outf=sys.stdout, nskip=0):
    "Reads data from inf in tree format"
    for line in inf:
        line = line.strip()
        if len(line) > 0:
            if nskip <= 0:
                trees = tb.string_trees(line)
                trees.insert(0, 'ROOT')
                outf.write(tree_string(trees).strip())
                outf.write('\n')
        else:
            if nskip <= 0:
                outf.write('\n')
                outf.flush()
            nskip -= 1

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-n", "--nepochs", type="int", dest="nepochs", default=0, help="total number of epochs")
    parser.add_option("-s", "--skip", type="float", dest="skip", default=0, help="initial fraction of epochs to skip")
    parser.add_option("-r", "--rate", type="int", dest="rate", default=1, help="input provides samples every rate epochs")
    (options,args) = parser.parse_args()
    assert(len(args) <= 2)
    inf = sys.stdin
    outf = sys.stdout
    if len(args) >= 1:
        inf = file(args[0], "rU")
        if len(args) >= 2:
            outf = file(args[1], "w")
    nskip = int(options.skip*options.nepochs/options.rate)
    read_write(inf, outf, nskip)
