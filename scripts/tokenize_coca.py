import argparse
import pickle
from transformers import AutoTokenizer

import importlib
import LiB
importlib.reload(LiB)

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

def train_model(processed_corpus):
    model = LiB.model
    coca_corpus = model.create_corpus(coca_sents)
    m = model(coca_corpus, lexicon_in=0.25, lexicon_out=0.0001, update_rate=0.2, life=3)
    m.logs['note'] = 0
    for epoch_id in range(0,501):
        m.run(epoch_id, article_length=200, test_interval=100);
        if epoch_id > 1 and epoch_id % 100 == 0:
            break

    print(f"Time cost for training: {sum(m.logs['time_cost'])//60} min {sum(m.logs['time_cost'])%60:.1f}s\n")

    print('Raw/LiB segmentation:')
    m.show_result(coca_corpus[:10])
    return m

def write_outputs(processed_corpus, filename, model = None):
    with open(filename, "w") as f:
        for sentence in processed_corpus:
            processed_sentence = sentence
            if model:
                processed_sentence = model.show_reading(sentence, return_chunks = True)
                f.write("|".join(processed_sentence) + "\n") # | is the delimiter for token boundaries
            else:
                f.write(processed_sentence + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, required = True)
    parser.add_argument("--output", type = str, required = True)
    parser.add_argument("--model", type = str, required = False)
    args = parser.parse_args()

    corpus = process_corpus(load_pickle(args.input))
    # this should prob be cleaned up: make a constant dictionary or something?
    if args.model == "LiB":
        lib_model = train_model(corpus)
        write_outputs(corpus, args.output, model = lib_model)
    elif args.model == "BPE":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        bpe_corpus = ["Ġ" + " ".join(tokenizer.tokenize(sentence)) for sentence in corpus] # figure out how to get the word boundary token from the library
        write_outputs(bpe_corpus, args.output)
    else:
        write_outputs(corpus, args.output)
