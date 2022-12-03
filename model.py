import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# TODO
class IntentModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.
        self.optimizer = None # AdamW(self.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=0.01, betas=(0.9, 0.999))
        self.train_steps = int(11514 / args.batch_size * args.n_epochs)
        self.warmup_steps = int(self.train_steps * args.warmup_ratio)
        self.scheduler = None # get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.train_steps)
        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(p = args.drop_rate)
        self.classify = Classifier(args, target_size)
        
    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # transformer_check
        self.encoder.resize_token_embeddings(len(self.tokenizer))  
        
    def forward(self, inputs, targets):
        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """
        
        # task1
        outputs = self.encoder(**inputs)
        last_hidden_state = outputs[0]
        
        # task2
        cls_token = last_hidden_state[:, 0, :]
        cls_token_d = self.dropout(cls_token)
        
        # task3
        logits = self.classify(cls_token_d)
        
        return logits

  
class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit

# TODO
def LLRD(model, args):
    opt_parameters = []
    named_parameters = list(model.named_parameters()) 

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = args.learning_rate
    head_lr = args.head_lr
    lr = init_lr  
    
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "classify" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "classify" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}    
    opt_parameters.append(head_params)
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       
        
        lr *= args.lr_decay
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    return AdamW(opt_parameters, lr=init_lr, eps=args.adam_epsilon)

class CustomModel(IntentModel):
    def __init__(self, args, tokenizer, target_size):
        super().__init__(args, tokenizer, target_size)
        # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
    
    def forward(self, inputs, targets):
        return super().forward(inputs, targets)

# TODO
class SupConModel(IntentModel):
    def __init__(self, args, tokenizer, target_size, feat_dim=768):
        super().__init__(args, tokenizer, target_size)
        # task1: initialize a linear head layer
        self.head = nn.Linear(args.embed_dim, args.feat_dim)
        self.classifier = Classifier(args, target_size)

    def forward(self, inputs, targets):
        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the normalized output of the dropout layer to the linear head layer; return the embedding
        """

        outputs = self.encoder(**inputs)
        last_hidden_state = outputs[0]
        
        cls_token = last_hidden_state[:, 0, :]
        
        dropout_1 = self.dropout(cls_token)
        dropout_1 = self.head(dropout_1)
        dropout_1 = F.normalize(dropout_1, dim=1)
        dropout_1.unsqueeze_(1)
        dropout_2 = self.dropout(cls_token)
        dropout_2 = self.head(dropout_2)
        dropout_2 = F.normalize(dropout_2, dim=1)
        dropout_2.unsqueeze_(1)

        # merge [n,f] and [n,f] to [n,2,f]
        x = torch.cat([dropout_1, dropout_2], dim=1)

        return x
    
    def get_logits(self, inputs):
        with torch.no_grad():
            output = self.encoder(**inputs)
            last_hidden_state = output[0]

            cls_token = last_hidden_state[:, 0, :]

            dropout = self.dropout(cls_token)

            return dropout