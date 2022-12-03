import os, sys, pdb
import numpy as np
import random
import torch

import math
import sklearn.datasets
import pandas as pd
import umap
import umap.plot

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel, LLRD
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

device='cuda'

def baseline_train(args, model, datasets, tokenizer):
    # combines LogSoftmax() and NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            model.optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)

            loss.backward()

            model.optimizer.step()

            losses += loss.item()
    
        run_eval(args=args, model=model, datasets=datasets, tokenizer=tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
  
def custom_train(args, model, datasets, tokenizer):
    # combines LogSoftmax() and NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon) if not args.llrd else LLRD(model, args)
    if args.warmup:
        model.scheduler = get_linear_schedule_with_warmup(model.optimizer, num_warmup_steps=model.warmup_steps, num_training_steps=model.train_steps)
      
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            model.optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)

            loss.backward()

            model.optimizer.step()
            if args.warmup:
                model.scheduler.step()
            
            losses += loss.item()

        run_eval(args=args, model=model, datasets=datasets, tokenizer=tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    losses = 0
    cls = None
    total_labels = None
    criterion = nn.CrossEntropyLoss()
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)

        with torch.no_grad():
            loss = criterion(logits, labels)
            losses += loss.item()
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

        if split == 'test' and args.task == 'baseline':
            with torch.no_grad():
                outputs = model.encoder(**inputs)
                hidden_states = outputs[0]
                cls_token = hidden_states[:, 0, :]
                        
                if cls is None:
                    cls = cls_token
                else:
                    cls = torch.cat((cls, cls_token), 0)

                if total_labels is None:
                    total_labels = labels
                else:
                    total_labels = torch.cat((total_labels, labels), 0)

    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))
    print(f'{split} loss:', losses/len(dataloader))

    if split == 'test' and args.task == 'baseline':
        # only keep label 0-9
        cls = cls.cpu().numpy()
        total_labels = total_labels.cpu().numpy()
        index = np.where(total_labels < 10)
        labels = total_labels[index]
        tokens = cls[index]

        umap_embeddings = umap.UMAP(n_neighbors=10, n_components=2).fit(tokens)
        p = umap.plot.points(umap_embeddings, labels=labels)
        p.figure.savefig(os.path.join('./results', str(args.task) + "_umap_plot.png"))

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss

    criterion = SupConLoss(temperature=args.temperature)
    cross_entropy = nn.CrossEntropyLoss()

    # task1: load training split of the dataset
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    val_dataloader = get_dataloader(args, datasets['validation'], split='validation')
    test_dataloader = get_dataloader(args, datasets['test'], split='test')
    
    # task2: setup optimizer_scheduler in your model
    model.optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup:
        model.scheduler = get_linear_schedule_with_warmup(model.optimizer, num_warmup_steps=model.warmup_steps, num_training_steps=model.train_steps)

    # task3: write a training loop for SupConLoss function 
    for epoch_count in range(args.n_epochs):
        losses = 0
        ce_losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            model.optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            if args.loss_fc == 'simclr':
                loss = criterion(logits)
            else:
                loss = criterion(logits, labels)
            
            loss.backward()

            with torch.no_grad():
                logits = model.get_logits(inputs)
            pred = model.classifier(logits)
            ce_loss = cross_entropy(pred, labels)

            ce_loss.backward()

            model.optimizer.step()
            if args.warmup:
                model.scheduler.step()
            
            losses += loss.item()
            ce_losses += ce_loss.item()
        
        print('epoch', epoch_count, '| SupCon losses:', losses/len(train_dataloader))
        print('epoch', epoch_count, '| Cross Entropy losses:', ce_losses/len(train_dataloader))

        acc = 0
        for step, batch in progress_bar(enumerate(val_dataloader), total=len(val_dataloader)):
            with torch.no_grad():
                inputs, labels = prepare_inputs(batch, model)
                logits = model.get_logits(inputs)
                pred = model.classifier(logits)
            
            tem = (pred.argmax(1) == labels).float().sum()
            acc += tem.item()
        
        print('val acc:', acc/len(datasets['validation']), '|dataset split validation size:', len(datasets['validation']))

    acc = 0
    supcon_losses = 0
    cross_entropy_losses = 0
    cls = None
    total_labels = None
    for step, batch in progress_bar(enumerate(test_dataloader), total=len(test_dataloader)):
        with torch.no_grad():
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            pred = model.classifier(logits)
            if args.loss_fc == 'simclr':
                supcon_loss = criterion(logits)
            else:
                supcon_loss = criterion(logits, labels)
            
            logits = model.get_logits(inputs)
            pred = model.classifier(logits)
            cross_entropy_loss = cross_entropy(pred, labels)
            
            outputs = model.encoder(**inputs)
            hidden_states = outputs[0]
            cls_token = hidden_states[:, 0, :]
                        
            if cls is None:
                cls = cls_token
            else:
                cls = torch.cat((cls, cls_token), 0)

            if total_labels is None:
                total_labels = labels
            else:
                total_labels = torch.cat((total_labels, labels), 0)
        
        supcon_losses += supcon_loss.item()
        cross_entropy_losses += cross_entropy_loss.item()
        tem = (pred.argmax(1) == labels).float().sum()
        acc += tem.item()

    print('test acc:', acc/len(datasets['test']), '|dataset split test size:', len(datasets['test']))
    print('test supcon loss:', supcon_losses/len(test_dataloader))
    print('test cross entropy loss:', cross_entropy_losses/len(test_dataloader))

    # only keep label 0-9
    cls = cls.cpu().numpy()
    total_labels = total_labels.cpu().numpy()
    index = np.where(total_labels < 10)
    labels = total_labels[index]
    tokens = cls[index]

    umap_embeddings = umap.UMAP(n_neighbors=10, n_components=2).fit(tokens)
    p = umap.plot.points(umap_embeddings, labels=labels)
    p.figure.savefig(os.path.join('./results', str(args.loss_fc) + "_umap_plot.png"))


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))
    
    if args.task == 'baseline':
        model = IntentModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'custom': # you can have multiple custom task for different techniques
        model = CustomModel(args, tokenizer, target_size=60).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        custom_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'supcon':
        model = SupConModel(args, tokenizer, target_size=60).to(device)
        supcon_train(args, model, datasets, tokenizer)
