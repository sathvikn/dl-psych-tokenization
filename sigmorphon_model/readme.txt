single best model from submission to SIG22 Task 2 (sentence-level), English model with POS features:
- sentence-level strategy (see paper): split sentences into single words and perform word-level segmentation
- after segmentation, glue segmented words back together to from original sentence
- i.e. test file must be tokenised (one word/token per line), see data/eng.sentence.preprocessed_split.test.tsv for an example
- shared task data is already tokenised, so have a look at this data if you work with custom data (I would assume a spacy tokenised would work fine, but I'd look at the tokenization of e.g. punctuation)
- I have added a python script (glue_words_task_II.py) that I used to form the original sentences
- The original data contained double whitespaces and this caused some problems with glue_words_task_II.py, so data/eng.sentence.corrected.test.tsv is a version of the test data with only single whitespaces
- also note that we used a different SINGLE segmentation token to decrease the complexity (↓), so check if this token is contained in your test data (if so, change it manually in the loaded vocabulary instance)

to run the model:
# output folder must exist ("." for current folder)
python predict.py --model-folder model --output PATH_TO_OUTPUT_DIR --test PATH_TO_FILE

If you want to reproduce the submitted results (paper only contains ensemble results):
# create predictions
python predict.py --model-folder model --test data/eng.sentence.preprocessed_split.test.tsv --output .
# replace segmentation token with original sequence
sed -i '' "s/↓/ @@/g" test_greedy.predictions
# create original sentences
python glue_words_task_II.py --full data/eng.sentence.corrected.test.tsv --split test_greedy.predictions --output .
# run official eval script
python evaluate.py --guess data/eng.sentence.test.gold.tsv --gold test_greedy.glued_predictions




