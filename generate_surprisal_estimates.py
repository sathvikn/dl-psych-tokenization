import argparse
import os
from typing import Dict, List

import kenlm
import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from tqdm import tqdm

def process_dundee_corpus(path: str, models: List[Dict]):
    averaged_times = [f for f in os.listdir(path) if f.endswith("_avg.txt")] # file name
    surprisal_rt = []
    for textfile in tqdm(averaged_times):
        current_sentence = []
        total_rt = 0
        file_lines = open(os.path.join(path, textfile), 'r').readlines()
        for line in file_lines:
            current_word, word_rt = line.split()
            current_sentence.append(current_word)
            total_rt += float(word_rt)
            if "." in current_word or "?" in current_word or "!" in current_word: # other conditions for end sentence, also condition to exclude things
                # TODO: up to index -1 is to exclude punctuation
                surprisal_rt.append(compute_sentence_surprisal(" ".join(current_sentence)[:-1], total_rt, models))
                current_sentence = []
                total_rt = 0 
    return pd.DataFrame(surprisal_rt)
    
def load_models(model_dir: str):
    # returns a kv pair of strings to kenLM Model objects for each model in the directory.
    models = []
    for model in os.listdir(model_dir):
        model_dict = {}
        model_dict['name'] = model.split(".arpa")[0]
        model_dict['lm'] = kenlm.Model(os.path.join(model_dir, model))
        if "bpe" in model_dict['name']:
            model_dict['tokenizer'] = AutoTokenizer.from_pretrained("gpt2") 
        models.append(model_dict)
    return models

def compute_sentence_surprisal(sentence: List[str], total_rt: float, models: List[Dict]):
    # for each model
    # tokenize sentence & compute surprisal
    sentence_rt_surprisals = {'sentence': sentence, 'total_rt': total_rt, 'word_count': len(sentence.split(" "))}
    for model_dict in models:
        processed_sentence = " ".join(sentence)
        model_name = model_dict['name']
        if 'tokenizer' in model_dict.keys(): # HACK, just works w/GPT2 tokenization now
            processed_sentence = "Ġ" + " ".join(model_dict['tokenizer'].tokenize(processed_sentence))
        # score is log prob (log 10)
        sentence_rt_surprisals[f'{model_name}_surprisal'] = model_dict['lm'].score(processed_sentence, bos = True, eos = True)
    return sentence_rt_surprisals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type = str, required = True, help = "directory w/input data")
    parser.add_argument("--models", type = str, required = True, help = "directory w/ngram models")
    parser.add_argument("--output", type = str, required = True, help = "output filepath")
    args = parser.parse_args()
    corpus_surprisals = pd.DataFrame()
    models = load_models(args.models)
    if "dundee" in args.corpus:
        corpus_surprisals = process_dundee_corpus(args.corpus, models)
    corpus_surprisals.to_csv(args.output)
