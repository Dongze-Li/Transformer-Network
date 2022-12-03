from transformers import BertTokenizer

def load_data():
    from datasets import load_dataset
    dataset = load_dataset("mteb/amazon_massive_intent")
    print(dataset)
    return dataset

# TODO
def load_tokenizer(args):
    # task1: load bert tokenizer from pretrained "bert-base-uncased", you can also set truncation_side as "left" 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
    return tokenizer
