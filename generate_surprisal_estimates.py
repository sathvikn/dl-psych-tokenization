import argparse
import os

import pandas as pd


def process_dundee_corpus(path: str):
    averaged_times = [f for f in os.listdir("path") if f.endswith("_avg.txt")] # file name
    surprisal_rt = []
    for textfile in averaged_times:
        current_sentence = []
        for line in textfile.readlines():
            current_sentence.append(line)
            if "." in line: # other conditions for end sentence, also condition to exclude things
                surprisal_rt.append(compute_sentence_surprisal(" ".join(current_sentence)))
                current_sentence = []

                
    return pd.DataFrame(surprisal_rt)

    
def load_models():


def compute_sentence_surprisal(sentence: str, models: list):
    # for each model
    # tokenize sentence & compute surprisal

    return {sentence: , avg_rt:, model: , model: }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type = str, required = True)
    parser.add_argument("--output", type = str, required = True)
    args = parser.parse_args()


