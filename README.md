# Sequence-to-sequence quotations attribution

This repository contains the source code for the paper "Fine-grained quotation detection and attribution in German news articles" by Fynn Petersen-Frey & Chris Biemann published at KONVENS 2024.
Implementation based on [Seq2seqCoref](https://github.com/WenzhengZhang/Seq2seqCoref).
 
## Setup

```
mamba env create -f env.yml --solver=libmamba
mamba activate seq2seq-quotations
```

## Dataset

Available at https://github.com/uhh-lt/german-news-quotation-attribution-2024


## Pre-trained models

* [mT5-large](https://ltdata1.informatik.uni-hamburg.de/quote/mt5-large.tar)

## Inference / Prediction

1. Download and untar a model (see Pre-trained models)
2. Place plain-text files to process in a folder
3. Create an empty directory for temporary files
4. Create an empty directory for final output
5. Activate the correct python environment (see setup)
6. Execute below command:

```
python predict.py <input-dir> <temp-processing-dir> <model-dir> <output-dir>
```

