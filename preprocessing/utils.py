import pandas as pd 
import matplotlib.pyplot as plt 

from nltk.tokenize import sent_tokenize
N_SENTS_IN_SEQUENCE = 6

def buid_writes_dataframe(writes_test_path, writes_train_path,
                          window=N_SENTS_IN_SEQUENCE): 

    def to_dataframe_form(list_of_dicts):
        return pd.concat(list(map(lambda dct: pd.DataFrame(dct), 
                                  list_of_dicts))).reset_index(drop=True)
    train_data = []
    test_data = []
    
    for  test_tmp_path, train_tmp_path in zip(writes_test_path.iterdir(), 
                                             writes_train_path.iterdir()):

        with open(train_tmp_path, 'r', encoding='utf8') as fp:
            texts = sent_tokenize(fp.read(), language='russian')
            texts = [" ".join(texts[idx:idx+window]) for idx in range(0, len(texts)-window, window)]
            filtered_rows =  filter(lambda row: len(row) != 0,
                                    texts)
            train_data.append({"sent": list(filtered_rows),
                               "author": train_tmp_path.__fspath__() \
                                                       .split('\\')[-1].split('.')[0]} 
                                                       )
            
        with open(test_tmp_path, 'r', encoding='utf8') as fp:
            texts = sent_tokenize(fp.read(), language='russian')
            texts = [" ".join(texts[idx:idx+window]) for idx in range(0, len(texts)-window, window)]
            filtered_rows =  filter(lambda row: len(row) != 0,
                                     texts)
            test_data.append({"sent": list(filtered_rows),
                               "author": test_tmp_path.__fspath__() \
                                                       .split('\\')[-1].split('.')[0]} 
                                                       )
            
        
    return to_dataframe_form(train_data), to_dataframe_form(test_data)


def discrete_distr(dataframe:pd.DataFrame, 
                   column_name:str='author',
                   ax=None,
                   suffix:str="") -> None: 
    
    dataframe[column_name].value_counts().plot(kind='bar', ax=ax)
    plt.ylabel('Count')
    title = f"{column_name} distr."
    if ax is None:
        plt.grid()
        plt.title(title+suffix)
        
    else: 
        ax.set_title(title+suffix)
