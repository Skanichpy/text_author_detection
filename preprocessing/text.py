from . import utils as u 
import pandas as pd 
import math 

from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

from pymorphy2 import MorphAnalyzer

class TrainTestProcessor: 

    def __init__(self, train_data,
                 test_data,
                 target_col,
                 text_col,
                 stopwords):
         
        self.train_data = train_data
        self.test_data = test_data 
        self.stopwords = stopwords

        self.target_col = target_col
        self.text_col = text_col 

        self.analyzer = MorphAnalyzer()
        self.normalize_func = lambda word_seq: [self.analyzer.parse(word)[0][0] \
                                                for word in word_seq]
        

    @staticmethod
    def base_text_preprocessing(text_series:pd.Series, 
                                stopwords):
        
        text_series = text_series.map(lambda row: row.lower())
        text_series = text_series.str.replace(r'[^а-я\s]+', '', 
                                            regex=True)
        text_series = text_series.str.replace(r'[.,!-+)(}{]+', ' ', 
                                            regex=True)
        text_series = text_series.map(word_tokenize)
        text_series = text_series.map(lambda sent: list(
                        filter(lambda word: word not in stopwords and len(word) != 0,
                                                            sent)))
        return text_series 

    @staticmethod
    def filter_by_length(data, text_col, q): 
        if q < 0 or q > 1:
            raise ValueError('q must be in [0:1]')
        length_series = data[text_col].map(len)
        lower_t = math.ceil(length_series.quantile(q))
        sup_t  = math.floor(length_series.quantile(1-q)) 
        
        filter_condition = (length_series > lower_t) & (length_series < sup_t)
        return data[filter_condition]
    
    @staticmethod
    def rebalance_by_target(train_data, test_data,
                            test_size:float,
                            target_column,
                            text_column,
                            sampler=None): 
        
        data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
        new_train_data, new_test_data = train_test_split(data, test_size=test_size,
                                                         stratify=data[target_column]
                                                         )
        if sampler is not None:
            X, y = sampler.fit_resample(new_train_data[text_column].values.reshape(-1,1),
                                        new_train_data[target_column].values.reshape(-1,1))
            new_train_data = pd.concat([pd.DataFrame(X, columns=[text_column]), 
                                        pd.DataFrame(y, columns=[target_column])], axis=1)
        return new_train_data, new_test_data

    def transform(self, q:float, 
                  to_normal_form:bool=True): 

        train_text_series = self.train_data[self.text_col]
        test_text_series = self.test_data[self.text_col]

        self.train_data[self.text_col] = self.base_text_preprocessing(train_text_series,
                                                                      self.stopwords)
        self.test_data[self.text_col] = self.base_text_preprocessing(test_text_series,
                                                                     self.stopwords)
        
        self.train_data = self.filter_by_length(self.train_data,
                                                self.text_col,
                                                q)
        self.test_data = self.filter_by_length(self.test_data,
                                               self.text_col,
                                               q)
        self.train_data[self.text_col] = self.train_data[self.text_col].map(self.normalize_func)
        self.test_data[self.text_col] = self.test_data[self.text_col].map(self.normalize_func)

        return self.train_data, self.test_data
    
    