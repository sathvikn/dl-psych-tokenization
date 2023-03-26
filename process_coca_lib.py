import pickle
import importlib
import LiB
importlib.reload(LiB)
import argparse

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

def write_outputs(processed_corpus, model, filename):
    with open(filename, "w") as f:
        for sentence in processed_corpus:
            f.write(" ".join(model.show_reading(sentence, return_chunks = True)) + "\n")


if "__name__" == __main__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, required = True)
    parser.add_argumnt("--output", type = str, required = True)
    args = parser.parse_args()

    corpus = process_corpus(load_pickle(args.input))
    model = train_model(corpus)
    write_outputs(corpus, model, args.output)
