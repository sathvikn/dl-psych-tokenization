#! /usr/bin/env python

usage = """%prog 

(c) Mark Johnson, 7th November 2008

Writes a PCFG that generates a bigram over a given set of terminals

usage: %prog [options] [terminals]"""

import optparse, re, sys

def nonterminal(options, term):
    return options.prefix+term+options.suffix

def rule(rule_prefix, parent, child1, child2=None):
    if rule_prefix == "":
        if child2:
            return parent+" --> "+child1+" "+child2
        else:
            return parent+" --> "+child1
    else:
        if child2:
            return rule_prefix+" "+parent+" --> "+child1+" "+child2
        else:
            return rule_prefix+" "+parent+" --> "+child1
        
if __name__ == '__main__':
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-s", "--start", dest="start", default="Start",
                      help="start nonterminal label")
    parser.add_option("-r", "--rule-prefix", dest="rule_prefix", default="",
                      help="prefix prepended to all rules")
    parser.add_option("-o", "--ordered-rule-prefix", dest="orule_prefix", default=None,
                      help="prefix prepended to all rules")
    parser.add_option("-p", "--prefix", dest="prefix",
                      help="prefix prepended to a terminal to make corresponding nonterminal")
    parser.add_option("-S", "--suffix", dest="suffix", default="",
                      help="suffix appended to a terminal to make corresponding nonterminal")
    (options,args) = parser.parse_args()

    if options.prefix == None and options.suffix == "":
        options.prefix = options.start+"_"

    if options.orule_prefix == None:
        options.orule_prefix = options.rule_prefix
        
    if len(args) > 0:
        terminals = args
    else:
        terminals = sys.stdin.read().split()

    for term in terminals:
        print rule(options.rule_prefix, options.start, nonterminal(options, term))

    for p1,parent in enumerate(terminals):
        print rule(options.rule_prefix, nonterminal(options, parent), parent)
        for p2,child in enumerate(terminals):
            print rule(options.orule_prefix if p1 < p2 else options.rule_prefix,
                       nonterminal(options, parent), parent, nonterminal(options, child))
    
    

