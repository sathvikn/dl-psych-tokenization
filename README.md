## Code for EMNLP 2023 Submission: Words, Subwords, and Morphemes: What Really Matters in the Surprisal-Reading Time Relationship?}

Run `conda create -n token --file environment.yml` to create a Conda environment with Python dependencies.

### Data Sources
The morphological transducer (renamed `neural_transducer` after cloning) was from https://github.com/slvnwhrl/il-reimplementation. Since the trained model was provided to us by the authors, we do not include it in the submission. If the paper is accepted, we will consult with them if we can provide the model when we release our code.

The following data were provided with the submission, our versions are available upon request. To replicate our implementation, make directories called `data` and `corpora`. In our directory `data`, `human_rts` came from  https://drive.google.com/file/d/1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8, provided in [Wilcox et al (2020)'s implementation](https://github.com/wilcoxeg/neural-networks-read-times/tree/master).
The reaction times from the Dundee and Natural Stories corpora were processed by one-off scripts in the `scripts` directory, and stored in `data/processed_rts`.
The version of COCA we used was from [Yang et al (2022)](https://www.frontiersin.org/articles/10.3389/frai.2022.731615/full), it was downloaded from https://osf.io/ydr7w/, and is in `corpora/public_coca_binary`.
KenLM was downloaded from https://github.com/kpu/kenlm, and built using the instructions under the "Compiling" section. Models were queried using a Python module, included with the rest of the dependencies.

### Running Experiments

To tokenize COCA, run `tokenize_coca.py`. If `transducer` or `BPE` are not provided as arguments to `model`, the corpus will be split based on orthographic words. Note: tokenizing the corpus according to the morphological transducer takes several hours on a personal computer.

Example:
`python tokenize_coca.py --input corpora/public_coca_binary --output corpora/public_coca_bpe.txt --model BPE`

To train a 5-gram model using KenLM, run `kenlm/bin/lmplz -o 5` and provide the input text and the path for the model file

Example:
`kenlm/build/bin/lmplz -o 5 <corpora/public_coca_bpe.txt >models/5gram_coca_bpe.arpa`

Finally, generate per-token surprisal estimates for a corpus of psycholinguistic results. These are stored in `data/surprisal_data`.

Example:
`python generate_surprisal_estimates.py --data data/processed_rts/dundee_rts.csv --model models/5gram_coca_bpe.arpa --output data/surprisal_data/dundee/bpe_surprisal.csv`

They will be combined with reading time data in `cleaning_exploration.ipynb` and written to `data/surprisal_data/`. That notebook also generates the bar graphs in the appendix, used for qualitative, exploratory purposes.
The predictive power analyses and visualizations are in `regression_analysis.ipynb`. Word frequency is used as a predictor in the regression models. They are in `data/word_freqs.txt`, and we query the n-gram model with orthographic tokenization for unigram probabilities as follows:

```
sed -n "/\\\\1-grams:/,/\\\\2-grams/p" models/5gram_baseline.arpa > word_freqs.txt
sed -i.backup '1d;$d' word_freqs.txt
```

### Citation
If you use any part of this implementation, please cite [our paper](https://aclanthology.org/2023.findings-emnlp.752/)

```
@inproceedings{nair-resnik-2023-words,
    title = "Words, Subwords, and Morphemes: What Really Matters in the Surprisal-Reading Time Relationship?",
    author = "Nair, Sathvik  and
      Resnik, Philip",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.752",
    doi = "10.18653/v1/2023.findings-emnlp.752",
    pages = "11251--11260",
    abstract = "An important assumption that comes with using LLMs on psycholinguistic data has gone unverified. LLM-based predictions are based on subword tokenization, not decomposition of words into morphemes. Does that matter? We carefully test this by comparing surprisal estimates using orthographic, morphological, and BPE tokenization against reading time data. Our results replicate previous findings and provide evidence that *in the aggregate*, predictions using BPE tokenization do not suffer relative to morphological and orthographic segmentation. However, a finer-grained analysis points to potential issues with relying on BPE-based tokenization, as well as providing promising results involving morphologically-aware surprisal estimates and suggesting a new method for evaluating morphological prediction.",
}
```
