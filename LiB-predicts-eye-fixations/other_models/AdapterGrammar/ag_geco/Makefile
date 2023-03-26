################################################################################
#                                                                              #
#                      Makefile for GECO segmentation                          #
#                                                                              #
################################################################################

################################################################################
#                                                                              #
#               Based on the Makefile for morph/syllable-pyg                   #
#                                                                              #
#                  Mark Johnson, 14th November 2008                            #
#                                                                              #
################################################################################

################################################################################
#                                                                              #
#                            parameters                                        #
#                                                                              #
################################################################################

# PROGRAM is the name of the program that will be run
PROGRAM=./py-cfg

# OUTPUTDIR is the directory in which all files are produced
OUTPUTDIR=output

# OUTPUTPREFIX is the prefix on all files in OUTPUTDIR associated with this run
OUTPUTPREFIX=geco-en
#OUTPUTPREFIX=geco-nl

# GRAMMARS is a list of grammars we're going to try (without the .lt suffix)
#
GRAMMARS=ag-collocation
# GRAMMARS=ag-word

# GRAMMARS=colloc3-br-nsyllables colloc3-br-syllables unigram colloc
# GRAMMARS=colloc2-syllables colloc-syllables colloc3-syllablesIF colloc2 syllables
# GRAMMARS=colloc3-nsyllables colloc3-syllables unigram colloc
# GRAMMARS=unigram colloc colloc3-syllables colloc3-nsyllables
# GRAMMARS=words
# GRAMMARS=unigram colloc colloc3-syllables colloc3-nsyllables colloc3-esyllables
# GRAMMARS=colloc2-esyllables colloc-esyllables unigram-esyllables
# GRAMMARS=colloc-osyllables
# GRAMMARS=colloc3-syllablesIF-suppressed
# GRAMMARS=colloc-syllablesIF-hierarchical colloc3-syllablesIF-hierarchical
# GRAMMARS=colloc-syllablesIF-hierarchical colloc3-syllablesIF-hierarchical colloc-syllablesIF-hierarchical0 colloc3-syllablesIF-hierarchical0
# GRAMMARS=colloc-syllablesIF-suppressed colloc2-syllablesIF-suppressed colloc3-syllablesIF-suppressed colloc123-syllablesIF-suppressed
# GRAMMARS=colloc-syllablesIF-suppressed colloc-syllablesIF-hierarchical colloc3-syllablesIF-suppressed colloc3-syllablesIF-hierarchical
# GRAMMARS=colloc3-syllablesIF-hierarchical-biased2 colloc3-syllablesIF-hierarchical-biasedI2
# GRAMMARS=colloc3-syllablesIF-hierarchical-biased2
# GRAMMARS=colloc3-syllablesIF-biased2
# GRAMMARS=colloc3-syllables-biased2 colloc3-syllablesrb-biased2
# GRAMMARS=syllables colloc3-syllablesIF-biased2 colloc3-syllablesIF colloc3-syllables colloc2-syllablesIF-biased2 colloc2-syllablesIF colloc2-syllables colloc-syllablesIF-biased2 colloc-syllablesIF colloc-syllables unigram-syllablesIF-biased2 unigram-syllablesIF unigram-syllables 
# GRAMMARS=colloc3-syllablesIF-biased2-bigram colloc3-syllablesIF-biased2-c2bigram colloc3-syllablesIF-biased2-cbigram
# GRAMMARS=colloc3-syllablesIF-hierarchical0-biased2 colloc3-syllablesIF-hierarchical-biased2 colloc3-syllablesIF-hierarchical-biasedI2
# GRAMMARS=colloc3-syllables

# OUTS is a list of types of output files we're going to produce
OUTS=score
# OUTS=prs

# Each fold is a different run; to do 4 runs set FOLDS=0 1 2 3
# FOLDS=0 1 2 3
FOLDS=0

PYAS=1e-4
PYES=1
PYFS=1
PYBS=1e4
PYGS=10
PYHS=0.1

PYWS=1
PYTS=1
PYMS=0
# PYNS=2000
PYNS=1
PYRS=-1

PYD=D_
PYE=E_

# These variables aren't iterated over

# CORPUS is the corpus we're going to learn from
CORPUS=geco_sents_nl.txt

# rate at which model's output is evaluated
TRACEEVERY=10

EXEC=time

################################################################################
#                                                                              #
#                     everything below this should be generic                  #
#                                                                              #
################################################################################


# The list of files we will make

OUTPUTS=$(foreach out,$(OUTS),$(foreach e,$(PYES),$(foreach f,$(PYFS),$(foreach g,$(PYGS),$(foreach h,$(PYHS),$(foreach a,$(PYAS),$(foreach b,$(PYBS),$(foreach w,$(PYWS),$(foreach t,$(PYTS),$(foreach m,$(PYMS),$(foreach n,$(PYNS),$(foreach R,$(PYRS),$(foreach GRAMMAR,$(GRAMMARS),$(OUTPUTDIR)/$(OUTPUTPREFIX)_G$(GRAMMAR)_$(PYD)$(PYE)n$(n)_m$(m)_t$(t)_w$(w)_a$(a)_b$(b)_e$(e)_f$(f)_g$(g)_h$(h)_R$(R).$(out))))))))))))))

GRAMMARFILES=$(foreach g,$(GRAMMARS),$(g).lt)

TARGETS=$(PROGRAM) $(OUTPUTS)

.PHONY: top
top: $(TARGETS)

py-cfg:
	ln -s ~/research/py-cfg/py-cfg py-cfg

# PRSFILES=$(patsubst $(OUTPUTDIR)/$(OUTPUTPREFIX)_%.$(out),$(OUTPUTDIR)/$(OUTPUTPREFIX)_%.prs,$(OUTPUTS))
# .SECONDARY: $(PRSFILES)
# .PRECIOUS: $(PRSFILES)

.SECONDARY:
.DELETE_ON_ERROR:

getarg=$(patsubst $(1)%,%,$(filter $(1)%,$(filter-out $(OUTPUTDIR)/$(OUTPUTPREFIX),$(subst _, ,$(2)))))

keyword=$(patsubst $(1),-$(1),$(filter $(1),$(filter-out $(OUTPUTDIR)/$(OUTPUTPREFIX),$(subst _, ,$(2)))))

$(OUTPUTDIR)/$(OUTPUTPREFIX)_%.score: score_seg.prl $(CORPUS) $(OUTPUTDIR)/$(OUTPUTPREFIX)_%.avprs
	perl $^ > $@	

$(OUTPUTDIR)/$(OUTPUTPREFIX)_%.avprs: mbr.py $(foreach fold,$(FOLDS),$(OUTPUTDIR)/$(OUTPUTPREFIX)_%_$(fold).sws)
	python $^ > $@

$(OUTPUTDIR)/$(OUTPUTPREFIX)_%.sws  $(OUTPUTDIR)/$(OUTPUTPREFIX)_%.wlt: $(PROGRAM) $(GRAMMARFILES) $(CORPUS) br-syll-input.py trees-words.py eval.py
	mkdir -p $(OUTPUTDIR)
	echo "Starting $@"
	$(EXEC) python br-syll-input.py < $(CORPUS) | $(PROGRAM) -A $(basename $@).prs \
		-C -r $$RANDOM -d=10 -x $(TRACEEVERY) \
		-F $(basename $@).trace \
		-G $(basename $@).wlt \
		$(call keyword,D,$@) \
		$(call keyword,E,$@) \
        -a $(call getarg,a,$@) \
        -b $(call getarg,b,$@) \
        -e $(call getarg,e,$@) \
        -f $(call getarg,f,$@) \
        -g $(call getarg,g,$@) \
        -h $(call getarg,h,$@) \
        -w $(call getarg,w,$@) \
        -T $(call getarg,t,$@) \
        -m $(call getarg,m,$@) \
        -n $(call getarg,n,$@) \
        -R $(call getarg,R,$@) \
		-X "python eval.py --gold $(CORPUS) --train-trees --score-cat-re=\"Word\" --morpheme-split-re=\"[ \\t]+\" --filter-terminal-re=\"[.]+\" > $(basename $@).weval" \
		-X "python trees-words.py -n $(call getarg,n,$@) -r $(TRACEEVERY) -s 0.5 > $(basename $@).sws" \
		$(call getarg,G,$@).lt 

.PHONY:
data-clean:
	rm -fr *.gold *.wlt *.prs *.yld *.eval *.evalb *.trace

.PHONY: 
clean: data-clean
	rm -fr *.o *.d *~ core read-tree.cc *.out

.PHONY: real-clean
real-clean: clean 
	rm -fr $(TARGETS)

