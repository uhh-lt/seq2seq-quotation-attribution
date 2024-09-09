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


## Inference / Prediction

```
python predict.py <input-dir> <temp-processing-dir> <model-dir> <output-dir>
```

