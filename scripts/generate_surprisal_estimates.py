import argparse
import os
import re
from typing import Dict, List

import kenlm
import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from tqdm import tqdm

def compute_surprisals(rt_data:pd.DataFrame, model: Dict):
    # rt_data has 4 columns, token_uid, token, rt, and transcript_id
    transcript_ids = rt_data['transcript_id'].unique()
    surprisals = []
    for tid in transcript_ids: # keeping track of the unique transcript
        transcript_data = rt_data[rt_data['transcript_id'] == tid]
        sentences = sent_tokenize(" ".join(transcript_data['token']))
        transcript_surprisals = []
        for i in np.arange(len(sentences)):
            sent, tokens = process_sentence(sentences[i])
            if len(tokens):
                if 'tokenizer' in model.keys(): # HACK, just works w/GPT2 tokenization now
                    tokens = model['tokenizer'].tokenize(sent)
                    # adding word boundary characters so sentence-initial tokens don't get treated differently
                    sent = "Ġ" + " ".join(tokens)
                    tokens[0] = "Ġ" + tokens[0]
                token_scores = [score for score in model['lm'].full_scores(sent, eos = False)]
                assert len(token_scores) == len(tokens)
                transcript_surprisals += [{"token": token, "transcript_id": tid, "sentence_id": i,
                "surprisal": convert_probability(result[0]), "oov": result[2]}
                # TODO convert token_score to suprisal
                for token, result in zip(tokens, token_scores)]
            else:
                print(sentences[i], tid)
        surprisals += transcript_surprisals
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

def load_model(model_path: str):
    model_dict = {}
    model_dict['name'] = model_path.split(".arpa")[0].split("/")[-1]
    model_dict['lm'] = kenlm.Model(model_path)
    if "bpe" in model_dict['name']:
        model_dict['tokenizer'] = AutoTokenizer.from_pretrained("gpt2") 
    return model_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, required = True, help = "path to CSV of RTs")
    parser.add_argument("--model", type = str, required = True, help = "path to .arpa file for 5gram LM")
    parser.add_argument("--output", type = str, required = True, help = "output filepath")
    args = parser.parse_args()
    corpus_surprisals = pd.DataFrame()
    model = load_model(args.model)
    rt_df = pd.read_csv(args.data)
    corpus_surprisals = compute_surprisals(rt_df, model)
    corpus_surprisals.to_csv(args.output)