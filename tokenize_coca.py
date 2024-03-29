import argparse
import json
import pickle
import sys
from typing import List

from transformers import AutoTokenizer

from src import morph_segmenter

def load_pickle(filename):
    with open(filename, "rb") as f:
        corpus = pickle.load(f)
    return corpus

def process_corpus(coca):
    coca_sents = []
    for doc in coca:
        for sentence in doc:
            coca_sents.append(("".join(sentence)).strip())
    return coca_sents

def write_outputs(processed_corpus: List[str], filename: str, model = None):
    with open(filename, "w") as f:
        for sentence in processed_corpus:
            f.write(sentence + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, required = True)
    parser.add_argument("--output", type = str, required = True)
    parser.add_argument("--model", type = str, required = False)
    args = parser.parse_args()

    corpus = process_corpus(load_pickle(args.input))
    config = json.load(open("model_config.json"))
    if args.model.upper() == "BPE":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        bpe_corpus = [config['bpe']['word_boundary'] + " ".join(tokenizer.tokenize(sentence)) for sentence in corpus] # figure out how to get the word boundary token from the library
        write_outputs(bpe_corpus, args.output)
    elif args.model == "transducer":
        transducer_results = morph_segmenter.tokenize_corpus(config['transducer']['path'], corpus, config['transducer']['word_boundary'])
        write_outputs(transducer_results, args.output)
    else:
        write_outputs(corpus, args.output)
