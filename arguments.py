import argparse
import os

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method",
                      choices=['baseline','custom','supcon'])

    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=0, type=int, 
                help="number of layers that are reinitialized. Count from last to first.")
    
    # Others
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=16, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=5e-5, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=512, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.1, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--feat-dim", default=768, type=int,
                help="The feature dimension of linear head.")
    parser.add_argument("--adam-epsilon", default=1e-3, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")
    parser.add_argument("--warmup", action="store_true",
                help="Whether to use warmup strategy for learning rate scheduler.")
    parser.add_argument("--warmup-ratio", default=0.1, type=float,
                help="Linear warmup over warmup_ratio.")
    parser.add_argument("--lr-decay", default=0.9, type=float,
                help="Learning rate decay ratio.")
    parser.add_argument("--llrd", action="store_true",
                help="Whether to use linear learning rate decay.")
    parser.add_argument("--head-lr", default=5e-5, type=float,
                help="Learning rate for head layer.")
    parser.add_argument("--loss-fc", default="supcon", type=str,
                help="Loss function for training", choices=['supcon', 'simclr'])
    parser.add_argument("--temperature", default=0.07, type=float,
                help="Temperature parameter for contrastive loss")

    args = parser.parse_args()
    
    return args
