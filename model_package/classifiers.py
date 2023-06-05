
from typing import Any
import pytorch_lightning as pl 

import torch
from torch import optim, nn 
import torch.nn.functional as F 

import torchmetrics as tm
from  . import embeddings as e


class LstmClassifier(pl.LightningModule): 
    def __init__(self, embedding_dim:int, 
                 hidden_size:int,
                 num_layers:int,
                 bidirectional:bool,
                 vocab,
                 rnn_dropout:float,
                 loss_fn=nn.CrossEntropyLoss()) -> None: 
        super().__init__()
        self.loss_fn = loss_fn
        self.emb = e.NavecEmbedding(num_embeddings=vocab.__len__(),
                                    embedding_dim=embedding_dim,
                                    vocab=vocab,
                                    )
        
        self.rnn = nn.LSTM(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=rnn_dropout,
                          bidirectional=bidirectional,
                          batch_first=True)
        
        self.scale_b = 2 if bidirectional else 1 
        flatten_dim = self.scale_b * num_layers * hidden_size
        self.num_classes = len(vocab.t_encoder.classes_)

        self.fc_classifier = nn.Sequential(
                        nn.BatchNorm1d(num_features=flatten_dim),
                        nn.Linear(flatten_dim, self.num_classes)
                        )
        # self.save_hyperparameters()

    def forward(self, x): 
        x_embedded = self.emb(x)
        _, (hidden, c0) = self.rnn(x_embedded)
        hidden = hidden * c0
        hidden = hidden.permute(1,0,2)
        hidden = nn.Flatten()(hidden)
        
        return self.fc_classifier(hidden)
    
    def training_step(self, batch, batch_idx): 
        x_batch, y_batch = batch
        logits = self(x_batch)

        loss = self.loss_fn(logits, y_batch.view(-1))
        self.log(name="train_loss", value=loss,
                 on_epoch=True, prog_bar=True)
        return loss 
    
    def validation_step(self, batch, batch_idx): 
        x, y = batch 
        acc, f1, aucroc  = self._shared_eval_step(batch, batch_idx)
        metrics = {'Acc.': acc, 'F1': f1, 'Auc.': aucroc}
        self.log(name="val_loss", value=self.loss_fn(self(x), y.view(-1)),
                 on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        
    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        x, _ = batch
        return self(x)

    def _shared_eval_step(self, batch, batch_idx): 
        x_batch, y_batch = batch
        y_batch = y_batch.view(-1).to(self.device)
        logits = self(x_batch).to(self.device)

        predictions = torch.argmax(logits, dim=1).view(-1)
        # if self.training:
        # predictions = torch.multinomial(nn.Softmax(dim=1)(logits), num_samples=1).view(-1)
        metrics = []

        acc = tm.Accuracy().to(device=self.device)(predictions, y_batch)
        f1 = tm.F1Score(average='weighted',
                        num_classes=self.num_classes)\
                    .to(device=self.device)(predictions, y_batch)
        auc = tm.AUROC(num_classes=self.num_classes)\
                    .to(device=self.device)(nn.Softmax(dim=1)(self(x_batch)), 
                                                              y_batch)

        return acc, f1, auc 

    def configure_optimizers(self):
        optimizer = optim.Adam([
                           *self.emb.parameters(),
                           *self.rnn.parameters(),
                           *self.fc_classifier.parameters()],
                           lr=3e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            factor=.5,
                                                            cooldown=0,
                                                            verbose=False)
        

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return [optimizer], [lr_scheduler_config]