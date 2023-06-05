from typing import Any, Iterable
from torch.utils.data import Dataset, DataLoader
import torch 
from sklearn.preprocessing import LabelEncoder

from numpy.random import randint
import numpy as np



class Vocab: 
    def __init__(self, data, target_col,
                 text_col, pad_token='<PAD>',
                 unk_token='<UNK>'):
        self.data = data 
        self.target_col = target_col
        self.text_col = text_col 
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.build(data[text_col])
        self.data, self.t_encoder = self.build_target_transform(data, col=target_col, t_encoder=None)
        
    def build(self,
              text_iterator:Iterable[Iterable]) -> None:
        self.tokens = set() 
        self.max_seq_len = 0 

        for token_sequence in text_iterator: 
            self.tokens.update(token_sequence)
            self.max_seq_len = self.max_seq_len if self.max_seq_len >= len(token_sequence) \
                                                else len(token_sequence)

        self.token2idx = {token:idx for idx, token in enumerate(self.tokens, 2)}
        self.token2idx[self.pad_token] = 0 
        self.token2idx[self.unk_token] = 1

        self.idx2token = {token:idx for idx, token in self.token2idx.items()}

    @staticmethod
    def build_target_transform(data, col, t_encoder=None):
        if t_encoder is None:
            t_encoder = LabelEncoder()
            data[col] = t_encoder.fit_transform(data[col])
            return data, t_encoder 
        else: 
            data[col] = t_encoder.transform(data[col])
            return data

    def __len__(self): 
        if hasattr(self, "token2idx"):
            return len(self.token2idx)

        else: 
            raise AttributeError("build Vocab before use len()")
        
    def __str__(self):
        random_idx = randint(0, len(self.token2idx))
        initial = f"Vocab(n_tokens={len(self.token2idx)}| max_seq_len={self.max_seq_len})"
        initial += f"\nVocab[{random_idx}]={self.idx2token[random_idx]}"
        return initial
    
    def __getitem__(self, key:str): 
        if key in self.token2idx:
            return self.token2idx.__getitem__(key)
        else:
            return self.token2idx[self.unk_token]
    

class TextDataset(Dataset): 

    def __init__(self, data, target_col, 
                 text_col,
                 vocab: Vocab,
                 noise_mask:bool=True): 
        
        self.X = data[text_col]
        self.gt = data[target_col]
        self.vocab = vocab 
        self.noise_mask = noise_mask

        self.__vectorize_word_index = np.vectorize(self.__word_index)

    def __getitem__(self, index):
        index_seq = self.__vectorize_word_index(self.X.iloc[index])
        non_standard_seq = torch.from_numpy(index_seq)
        X = torch.ones(self.vocab.max_seq_len) * self.vocab[self.vocab.pad_token]
        if len(non_standard_seq) <= self.vocab.max_seq_len:
            X[:len(non_standard_seq)] = non_standard_seq
        else:
            X[:self.vocab.max_seq_len] = non_standard_seq[:self.vocab.max_seq_len]

        return X.long(), \
               torch.Tensor([self.gt.iloc[index]]).long()
    
    def __len__(self): 
        return len(self.X)

    def __word_index(self, token, p=0.01): 
        if self.noise_mask:
            if np.random.random() < p:
                return self.vocab[self.vocab.unk_token]
            else: 
                return self.vocab[token]
        else:
            if token in self.vocab.token2idx: 
                return self.vocab[token]
            else: 
                return self.vocab[self.vocab.unk_token]
            


## utils for evaluation:   
from  preprocessing.text import TrainTestProcessor as processor 
import pandas as pd 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def texts_to_model_form(texts, 
                        vocab, 
                        window=5): 
    texts = list(map(lambda lst: " ".join(lst),
        [texts[s:s+window] for s in range(0, len(texts), window)]))
    clean_text_series = processor.base_text_preprocessing(pd.Series(texts), stopwords.words('russian'))
    data = pd.DataFrame({"sent": clean_text_series,
                     'author': 0})
    dataset = TextDataset(data, target_col='author',
                          text_col='sent',
                          vocab=vocab,
                          noise_mask=False)

    X, _ = next(iter(DataLoader(dataset, shuffle=True, batch_size=len(dataset))))
    return X 

def images_to_model_form(images, 
                         reader, 
                         vocab, 
                         window=5,
                         ) -> torch.Tensor:
    texts = list(map(lambda img: " ".join(map(lambda seq: seq[1], reader.readtext(img))), 
                    images))
    texts = list(map(lambda text_corpus: sent_tokenize(text_corpus), texts))
    texts = sum(texts, [])
    return texts_to_model_form(texts, vocab, window)
    


