import argparse
import json
import re
from tqdm import tqdm
from typing import Dict

import kenlm
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from src.morph_segmenter import load_model_and_vocab, tokenize_sentence

def compute_surprisals(rt_data:pd.DataFrame, model: Dict):
    # rt_data has 4 columns, token_uid, token, rt, and transcript_id
    transcript_ids = rt_data['transcript_id'].unique()
    surprisals = []
    lookup_tbl = [] # mapping words to their morphological tokenizations
    print("transcript progress:")
    for tid in tqdm(transcript_ids): # keeping track of the unique transcript
        transcript_data = rt_data[rt_data['transcript_id'] == tid]
        sentences = sent_tokenize(" ".join(transcript_data['token']))
        transcript_surprisals = []
        for i in np.arange(len(sentences)):
            sent, tokens = process_sentence(sentences[i])
            if len(tokens):
                if 'bpe' in model['name']:
                    tokens = model['tokenizer'].tokenize(sent)
                    # adding word boundary characters so sentence-initial tokens don't get treated differently
                    sent = model['word_boundary'] + " ".join(tokens)
                    tokens[0] = model['word_boundary'] + tokens[0]
                elif 'morph' in model['name']:
                    tokens, token_mapping = tokenize_sentence(model['transducer'], model['vocab'], sent, model['word_boundary'])
                    sent = " ".join(tokens)
                    tokens = sent.split(" ")
                    lookup_tbl += token_mapping
                token_scores = [score for score in model['lm'].full_scores(sent, eos = False)]
                assert len(token_scores) == len(tokens)
                transcript_surprisals += [{"token": token, "transcript_id": tid, "sentence_id": i,
                "surprisal": convert_probability(result[0]), "oov": result[2]}
                for token, result in zip(tokens, token_scores)]
            else:
                print(sentences[i], tid)
        surprisals += transcript_surprisals
    if len(lookup_tbl):
        with open(model['lookup_path'], "w") as file:
            file.writelines(line + "\n" for line in lookup_tbl)
    return pd.DataFrame(surprisals)

def convert_probability(score: float):
    # the kenLM scores are in log base 10
    return -np.log2(10**score)

def process_sentence(sentence: str):
    sent = re.sub(r'[^\w\s]', '', sentence).lower().strip() # get rid of punctuation
    tokens = sent.split(" ")
    tokens = np.delete(tokens, [i for i in range(len(tokens)) if not tokens[i]])
    sent = " ".join(tokens)
    return sent, tokens

def load_model(model_path: str, model_config: Dict):
    # Create a dictionary with the loaded ngram model and metadata from the config json
    model_dict = {}
    model_dict['name'] = model_path.split(".arpa")[0].split("/")[-1]
    model_dict['lm'] = kenlm.Model(model_path)
    if "bpe" in model_dict['name']:
        model_dict['tokenizer'] = AutoTokenizer.from_pretrained("gpt2")
        model_dict['word_boundary'] = model_config['bpe']['word_boundary']
    elif "morph" in model_dict['name']:
        model_dict['transducer'], model_dict['vocab'] = load_model_and_vocab(model_config['transducer']['path'])
        model_dict['word_boundary'], model_dict['lookup_path'] = model_config['transducer']['word_boundary'], model_config['transducer']['lookup_tbl']
    return model_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, required = True, help = "path to CSV of RTs")
    parser.add_argument("--model", type = str, required = True, help = "path to .arpa file for 5gram LM")
    parser.add_argument("--output", type = str, required = True, help = "output filepath")
    args = parser.parse_args()
    corpus_surprisals = pd.DataFrame()
    model_config = json.load(open("model_config.json"))
    model = load_model(args.model, model_config)
    rt_df = pd.read_csv(args.data)
    corpus_surprisals = compute_surprisals(rt_df, model)
    corpus_surprisals.to_csv(args.output)