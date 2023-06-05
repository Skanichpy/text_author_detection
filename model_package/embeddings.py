
import torch 
from torch import nn 
from navec import Navec
import numpy as np 

class NavecEmbedding(nn.Embedding): 
    def __init__(self, num_embeddings, embedding_dim, 
                vocab) -> None:
        padding_idx = vocab[vocab.pad_token] 
        
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx)
        self.n_pretrained = 0 
        self.not_matched_tokens = list() 
        self.build_navec_embeddings() 

        for token, idx in vocab.token2idx.items(): 
            if token != '<UNK>':
                vector = self.get_vector_by_key(token)
            else: 
                vector = self.navec['<unk>']
            vector = torch.from_numpy(vector)[:embedding_dim]
            
            with torch.no_grad():
                self.weight[idx, :] = nn.Parameter(vector) 

    def build_navec_embeddings(self):
        path2navec = "model_package/pretrained/navec_hudlit_v1_12B_500K_300d_100q.tar" 
        self.navec = Navec.load(path2navec)
            
        
    def get_vector_by_key(self, key): 
        if key in self.navec:   
            self.n_pretrained += 1         
            return self.navec[key]
        else: 
            self.not_matched_tokens.append(key)
            return np.random.randn((300))