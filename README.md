# TEXT AUTHOR DETECTION
## <b> Tech:</b> 
[![python](https://img.shields.io/badge/Python-black?style=for-the-badge&logo=Python)]()
[![jupyter](https://img.shields.io/badge/Jupyter-black?style=for-the-badge&logo=Jupyter)]()
[![Pytorch](https://img.shields.io/badge/Pytorch-black?style=for-the-badge&logo=Pytorch)](https://pytorch.org/)<br>
[![Lightning](https://img.shields.io/badge/Lightning-black?style=for-the-badge&logo=Lightning)](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
[![navec](https://img.shields.io/badge/Navec-black?style=for-the-badge&)](https://github.com/natasha/navec)
## <b>Approach:</b> 
This repository contains functionality for text classification based on <a src="https://github.com/natasha/navec">pre-trained word embeddings</a> for russian words.
The key idea is to build the multi-layer bidirectional LSTM (seq2one).


## <b> Structure</b>: 
```
├── source [training source] 
├── ...
├── model_package:
├── ├── embeddings.py             [Pretrained word embedding layer]
├── ├── classifiers.py            [Multi-layer bidirectional LSTM-classifier]
├── preprocessing: 
├── ├── extract_zipfiles.py       [extract zipfiles in correct form for launching]
├── ├── text.py                   [train-test processor raw data]
├── ├── utils.py                  [text dataframes]
├── ...
├── data_tools.py                 [torch data implementation]
├── utils.py                      [additional tools for evaluation and plotting]
├── 01_preprocessing.ipynb        [data processing]
├── 02_data_implementation.ipynb  [parse data to torch constructions]
├── 03_training_lstm.ipynb        [train LstmClassifier.ipynb]
└── 04_results_testing.ipynb      [test model with different inputs]
```
