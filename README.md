## Code for EMNLP 2023 Submission on Tokenization Techniques and Psychometric Predictive Power

### Data Sources
The morphological transducer (renamed `neural_transducer` after cloning) was from https://github.com/slvnwhrl/il-reimplementation [TODO: add note on trained model??]
In `data`, `human_rts` came from  https://drive.google.com/file/d/1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8, provided in the Wilcox et al (2020)'s implementation: https://github.com/wilcoxeg/neural-networks-read-times/tree/master.
The reaction times from the Dundee and Natural Stories corpora were processed by one-off scripts in the `scripts` directory, and stored in `data/processed_rts`.
The version of COCA we used was from [Yang et al (2022)](https://www.frontiersin.org/articles/10.3389/frai.2022.731615/full), it was downloaded from https://osf.io/ydr7w/, and is in `corpora/public_coca_binary`.
KenLM was downloaded from https://github.com/kpu/kenlm, and built using the instructions under the "Compiling" section. Models were queried using a Python module, included with the rest of the dependencies (is it???)

### Running Experiments

To tokenize COCA, run `tokenize_coca.py`. If `transducer` or `BPE` are not provided as arguments to `model`, the corpus will be split based on orthographic words. Note: tokenizing the corpus according to the morphological transducer takes several hours on a personal computer.

Example:
`python tokenize_coca.py --input corpora/public_coca_binary --output corpora/public_coca_bpe.txt --model BPE`

To train a 5-gram model using KenLM, run `kenlm/bin/lmplz -o 5` and provide the input text and the path for the model file

Example:
`kenlm/bin/lmplz/ -o 5 <corpora/public_coca_bpe.txt >models/5gram_coca_bpe.arpa`

Finally, generate per-token surprisal estimates for a corpus of psycholinguistic results. These are stored in `data/surprisal_data`.

They will be combined with reading time data in `cleaning_exploration.ipynb` and written to `data/surprisal_data/`. That notebook also generates the graphs in the appendix, used for qualitative, exploratory purposes.
The predictive power analyses and visualizations are in `regression_analysis.ipynb`.
