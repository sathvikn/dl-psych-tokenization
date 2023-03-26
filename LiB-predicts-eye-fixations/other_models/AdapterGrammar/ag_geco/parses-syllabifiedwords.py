#! /usr/bin/env python

usage = """%prog -- convert parse trees into syllabified words

Version of 4th May 2008

(c) Mark Johnson

usage: %prog [options] < parse-trees.txt > syllabified-words.txt

This program is very simple.  It collects the terminal string, and
inserts a space ' ' at the right boundary of every Word category and
inserts a period '.' at the right boundary of Syllable category.

This program assumes that Syllable categories are always embedded
within Word categories.
"""

import lx, tb
import optparse, re, sys

def tree_syllabifiedwords(tree, word, syllable):
    """extracts all word and syllable categories from tree and
    returns a syllabifiedword string"""

    def visit(node):
        if tb.is_terminal(node):
            if len(node) > 0 and node[0] == '\\':
                node = node[1:]
            syllabifiedwords.append(node)
        else:
            for child in tb.tree_children(node):
                visit(child)
            if tb.tree_label(node) == syllable:
                syllabifiedwords.append('.')
            if tb.tree_label(node) == word:
                syllabifiedwords.append(' ')
                
    syllabifiedwords = []
    visit(tree)
    return ''.join(syllabifiedwords).rstrip()
    

if __name__ == '__main__':
    op = optparse.OptionParser(usage=usage)
    op.add_option("-s", "--syllable", dest="syllable", help="syllable category", default="Syllable")
    op.add_option("-w", "--word", dest="word", help="word category", default="Word")
    (options, args) = op.parse_args()

    if len(args) > 1:
        sys.stderr.write("Error: too many command-line arguments\n")
        sys.exit(2)

    if len(args) == 1:
        inf = file(args[0], "rU")
    else:
        inf = sys.stdin

    for line in inf:
        line = line.strip()
        if len(line) == 0:
            continue
        tree = tb.string_trees(line)
        tree.insert(0, 'ROOT')
        syllabifiedwords = tree_syllabifiedwords(tree, options.word, options.syllable)
        print syllabifiedwords
