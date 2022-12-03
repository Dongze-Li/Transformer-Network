---
title: Experiment of Transformer Network with Amazon Massive Intent Dataset
author: Dongze Li, Xiaoyan He, Yunfan Long
Date: November 2022
---

# Experiment of Transformer Network with Amazon Massive Intent Dataset

Source code from CSE 151B Fall 2022 PA4

## Description

Transformers have recently demonstrated cutting-edge performance in a variety of tasks. In this paper, we investigate and employ a pre-trained BERT model to categorize user intent. BERT will be adjusted as a baseline model initially, followed by the use of various training methods learned from a blog to create a unique model, and ultimately, the training of models with contrastive losses. For training our model, we’ll use the Amazon Massive Intent dataset. This dataset includes 60 different user intent categories. With train/validation/test splits, Huggingface makes it simple to become familiar with the data samples and labels. For the classification tasks, the text serves as the model’s input and the intent label serves as its output (with the exception of contrastive learning at Task 5). We use accuracy, namely, the correct predictions over total number of samples, as our evaluation metrics. In our baseline model, we employed pre-trained BERT model as our encoder and implemented a classifier which will provide the final classification result. Our best final result of classification task which achieves test accuracy around 88.9711% for test set. After custom fine-tuning, we improve the test accuracy to 90.0471% by reducing batch size to 16 and adding both warm up and LLRD to get this improvement more than 1%. Finally, we explored contrastive learning by changing loss function to SimCLR and SupContrast. We found that with SimCLR, we get a test accuracy of 61.829% as our best result while with SupContrast we got a similar result (test accuracy 88.1977%) as our baseline model. This is not surprising because unlike SupContrast, SimCLR is not supervised, which means it can be hard for it to get high accuracy for a classification task, given the real result is not directly used to force training.

## Getting Started

### Major Dependencies

* `Python3`
* `PyTorch`
* `transformers`
* `datasets`
* `numpy`
* `matplotlib`
* `tqdm`
* `umap`
* `umap.plot`
* other basic packages

### Installs

* Everything should be done via internet connection when run `main.py` file.

### Files

* `argutils.py`: argument parser which contains all the arguments for the model
* `dataloaders.py`: data loader for the model
* `load.py`: load the data and tokenizer
* `loss.py`: special loss function for the model (SupCon and SimCLR)
* `model.py`: contains model
* `main.py`: main file to run the model
* `README.md`: this file
* `utils.py`: some utils for the model
* `run.sh`: bash file to run the model

### Executing Program

* Go to the correct directory where all files located
* In the terminal, run `python main.py` with or without arguments

* Hyper parameters for the model:
    * `--batch-size n`: batch size for training, default is n=16
    * `--learning-rate n`: learning rate for training, default is n=5e-5
    * `--hidden-dim n`: hidden dimension for linear layer, default is n=512
    * `--drop-rate n`: dropout rate for linear layer, default is n=0.1
    * `--feat-dim`: feature dimension for linear layer, default is n=768
    * `--adam-elipson n`: adam elipson for training, default is n=1e-3
    * `--n-epochs n`: number of epochs for training, default is n=10
    * `--max-len n`: max length for input, default is n=20
    * `--warmup`: turn on warmup for training
    * `--warm-ratio n`: warmup ratio for training, default is n=0.1
    * `--llrd`: turn on LLRD for training
    * `--lr-decay n`: learning rate decay for training, default is n=0.9
    * `head-lr n`: head learning rate for training, default is n=5e-5
    * `--loss-fc`: choose loss function for training, choices are `supcon` and `simclr`, default is `supcon`
    * `--temperature n`: temperature for loss function, default is n=0.07

* For different task you can use the following command:
    * `--task`: choose task for training, choices are `baseline`, `custom`, and `supcon`, default is `baseline`
    * `--do-train`: turn on training
    * `--do-eval`: turn on evaluation

* An example of running this file will be:
    * `python main.py --do-train`: run the model with default hyper parameters
* Or, for example
    * `python main.py --do-train --task custom --batch-size 32 --learning-rate 1e-4 --warmup --llrd`: run the model with custom task, batch size 32, learning rate 1e-4, with warmup and LLRD turned on.

## Help
* The run.sh has several default run commands inside, feel free to change them to run the model with different hyper parameters.
* Also, there are other arguments you can use to run the model, feel free to check them out in `arguments.py` file.
* To install umap, please refer to the repository(https://github.com/lmcinnes/umap)

## Authors
Contributor names and contacts info:

* Li, Dongze
    * dol005@ucsd.edu
* He, Xiaoyan
    * x6he@ucsd.edu
* Long, Yunfan
    * yulong@ucsd.edu

## Acknowledgments

We appreciate the help from the coruse website, Piazza, as well as TAs and Tutors' office hours. We also appreciate Professor [Garrison W. Cottrell](https://cseweb.ucsd.edu/~gary/) for his lectures and teachings.

This respository is refered and modified from this repository of the [paper](https://arxiv.org/abs/2109.03079).
