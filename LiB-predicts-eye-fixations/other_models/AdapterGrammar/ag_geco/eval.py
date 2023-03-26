#! /bin/env python

usage = """%prog -- evaluate word and syllable segmentation

Version of 4th May 2008

(c) Mark Johnson

usage: %prog [options]

"""

import lx, tb
import optparse, re, sys

def filter_rex(strg0, filter_terminal_rex):
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


def tree_morphemes(tree, score_cat_rex, filter_terminal_rex):

    def visit(node, ancmatch, morphssofar, segssofar):
        """Does a preorder visit of the nodes in the tree"""
        if tb.is_terminal(node):
            t = filter_rex(simplified_terminal(node), filter_terminal_rex)
            if t != "":
                segssofar.append(t)
        elif tb.is_preterminal(node):
            for child in tb.tree_children(node):
                morphssofar,segssofar = visit(child, True, morphssofar, segssofar)
        else:
            children = tb.tree_children(node)
            for matchflag, begin, end in wordseqmatchiter(score_cat_rex, map(tb.tree_label, children)):
                if matchflag:
                    if segssofar != []:
                        morphssofar.append(''.join(segssofar))
                        segssofar = []
                    for child in children[begin:end]:
                        morphssofar,segssofar = visit(child, True, morphssofar, segssofar)
                    if segssofar != []:
                        morphssofar.append(''.join(segssofar))
                        segssofar = []
                else:  # no match
                    for child in children[begin:end]:
                        morphssofar,segssofar = visit(child, True, morphssofar, segssofar)
                    if segssofar != [] and not ancmatch:
                        morphssofar.append(''.join(segssofar))
                        segssofar = []
        return morphssofar,segssofar

    morphssofar,segssofar = visit(tree, False, [], [])
    assert(segssofar == [])
    return morphssofar

def morphemes_stringpos(ms):
    stringpos = set()
    left = 0
    for m in ms:
        right = left+len(m)
        stringpos.add((left,right))
        left = right
    return stringpos
    
def read_data(inf, tree_flag, types_flag, score_cat_rex, filter_terminal_rex, morpheme_split_rex, debug_level=0):
    "Reads data from inf, either in tree format or in flat format"
    strippedlines = [line.strip() for line in inf]
    if tree_flag:
        lines0 = []
        for line in strippedlines:
            trees = tb.string_trees(line)
            trees.insert(0, 'ROOT')
            lines0.append(tree_morphemes(trees, score_cat_rex, filter_terminal_rex))
            if debug_level >= 10000:
                sys.stderr.write("# line = %s,\n# morphemes = %s\n"%(line, lines0[-1]))
    else:
        lines00 = ((filter_rex(morph, filter_terminal_rex) for morph in morpheme_split_rex.split(line)) for line in strippedlines)
        lines0 = [[morph for morph in line if morph != ""] for line in lines00]

    if types_flag:
        lines = []
        dejavu = set()
        for morphs in lines0:
            word = ''.join(morphs)
            if word not in dejavu:
                dejavu.add(word)
                lines.append(morphs)
    else:
        lines = lines0
        
    # print "# tree_flag =", tree_flag, "source =", strippedlines[-1], "line =", lines[-1]
    
    words = [''.join(morphs) for morphs in lines]
    stringpos = [morphemes_stringpos(morphs) for morphs in lines]
    return (words,stringpos)

PrecRecHeader = "# exact-match f-score precision recall";

class PrecRec:
    def __init__(self):
        self.test = 0
        self.gold = 0
        self.correct = 0
        self.n = 0
        self.n_exactmatch = 0
    def precision(self):
        return self.correct/(self.test+1e-100)
    def recall(self):
        return self.correct/(self.gold+1e-100)
    def fscore(self):
        return 2*self.correct/(self.test+self.gold+1e-100)
    def exact_match(self):
        return self.n_exactmatch/(self.n+1e-100)
    def update(self, testset, goldset):
        self.n += 1
        if testset == goldset:
            self.n_exactmatch += 1
        self.test += len(testset)
        self.gold += len(goldset)
        self.correct += len(testset & goldset)
    def __str__(self):
        return ("%.4g\t%.4g\t%.4g\t%.4g" % (self.exact_match(), self.fscore(), self.precision(), self.recall()))

def data_precrec(trainwords, goldwords):
    if len(trainwords) != len(goldwords):
        sys.stderr.write("## ** len(trainwords) = %s, len(goldwords) = %s\n" % (len(trainwords), len(goldwords)))
        sys.exit(1)
    pr = PrecRec()
    for (t,g) in zip(trainwords, goldwords):
        pr.update(t, g)
    return pr

def evaluate(options, trainwords, trainstringpos, goldwords, goldstringpos):
    
    if options.debug >= 1000:
        for (tw, tsps, gw, gsps) in zip(trainwords, trainstringpos, goldwords, goldstringpos):
            sys.stderr.write("Gold: ")
            for l,r in sorted(list(gsps)):
                sys.stderr.write(" %s"%gw[l:r])
            sys.stderr.write("\nTrain:")
            for l,r in sorted(list(tsps)):
                sys.stderr.write(" %s"%tw[l:r])
            sys.stderr.write("\n")
            
    if goldwords != trainwords:
        sys.stderr.write("## ** gold and train terminal words don't match (so results are bogus)\n")
        sys.stderr.write("## len(goldwords) = %s, len(trainwords) = %s\n" % (len(goldwords), len(trainwords)))
        for i in xrange(min(len(goldwords), len(trainwords))):
            if goldwords[i] != trainwords[i]:
                sys.stderr.write("# first difference at goldwords[%s] = %s\n# first difference at trainwords[%s] = %s\n"%
                                 (i,goldwords[i],i,trainwords[i]))
                break

    pr = data_precrec(trainstringpos, goldstringpos)
    if options.extra:
        pr += '\t'+options.extra
    print pr
    sys.stdout.flush()
             

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-g", "--gold", dest="goldfile", help="gold file")
    parser.add_option("-t", "--train", dest="trainfile", help="train file")
    parser.add_option("--gold-trees", dest="goldtree_flag", default=False,
                      action="store_true", help="gold data is in tree format")
    parser.add_option("--train-trees", dest="traintree_flag", default=False,
                      action="store_true", help="train data is in tree format")
    parser.add_option("-c", "--score-cat-re", dest="score_cat_re", default=r"$",
                      help="score categories in tree input that match this regex")
    parser.add_option("-f", "--filter-terminal-re", dest="filter_terminal_re", default=r"(?:[-]|(?:^[$]{3}$))",
                      help="filter substrings of terminals that match this regex")
    parser.add_option("-m", "--morpheme-split-re", dest="morpheme_split_re", default=r"[- \t]+",
                      help="regex used to split morphemes with non-tree input")
    parser.add_option("--types", dest="types_flag", default=False,
                      action="store_true", help="ignore multiple lines with same yield")
    parser.add_option("--extra", dest="extra", help="suffix to print at end of evaluation line")
    parser.add_option("-d", "--debug", dest="debug", help="print debugging information", default=0, type="int")
    (options,args) = parser.parse_args()
    
    if options.goldfile == options.trainfile:
        sys.stderr.write("## ** gold and train both read from same source\n")
        sys.exit(2)
    if options.goldfile:
        goldf = file(options.goldfile, "rU")
    else:
        goldf = sys.stdin
    if options.trainfile:
        trainf = file(options.trainfile, "rU")
    else:
        trainf = sys.stdin

    if options.debug > 0:
        sys.stderr.write('# score_cat_re = "%s"\n# filter_terminal_re = "%s"\n# morpheme_split_re = "%s"\n' % (options.score_cat_re, options.filter_terminal_re, options.morpheme_split_re))
        
    score_cat_rex = re.compile(options.score_cat_re)
    filter_terminal_rex = re.compile(options.filter_terminal_re)
    morpheme_split_rex = re.compile(options.morpheme_split_re)
    
    (goldwords,goldstringpos) = read_data(goldf, tree_flag=options.goldtree_flag, types_flag=options.types_flag,
                                          score_cat_rex=score_cat_rex, filter_terminal_rex=filter_terminal_rex,
                                          morpheme_split_rex=morpheme_split_rex)

    print PrecRecHeader
    sys.stdout.flush()
    
    trainlines = []
    for trainline in trainf:
        trainline = trainline.strip()
        if trainline != "":
            trainlines.append(trainline)
            continue

        (trainwords,trainstringpos) = read_data(trainlines, tree_flag=options.traintree_flag, types_flag=options.types_flag,
                                                score_cat_rex=score_cat_rex, filter_terminal_rex=filter_terminal_rex,
                                                morpheme_split_rex=morpheme_split_rex, debug_level=options.debug)

        evaluate(options, trainwords, trainstringpos, goldwords, goldstringpos)
        trainlines = []

    if trainlines != []:
        (trainwords,trainstringpos) = read_data(trainlines, tree_flag=options.traintree_flag, types_flag=options.types_flag,
                                                score_cat_rex=score_cat_rex, filter_terminal_rex=filter_terminal_rex,
                                                morpheme_split_rex=morpheme_split_rex, debug_level=options.debug)
        evaluate(options, trainwords, trainstringpos, goldwords, goldstringpos)


