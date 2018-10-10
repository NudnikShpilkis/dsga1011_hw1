from random import shuffle

import spacy
import string
from tqdm import tqdm_notebook 

import pickle as pkl

import glob

import tokenize

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

##########Tokenizer##########

tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation

def lower_case_remove_punc(parsed):
    return [token.text.lower() for token in parsed if (token.text not in punctuations)]

def stop_word_remove(parsed):
    return [token.text.lower() for token in parsed if ((not token.is_stop) & (not token.is_punct) & (not token.is_space))]

def entity_keep(parsed):
    return [token.text.lower() for token in parsed if ((token.ent_type != "") & (not token.is_punct))]

def tokenize_dataset(dataset, save_tokens, ns=False, ek=False):
    token_dataset = []
    all_tokens = []

    for sample in tqdm_notebook(tokenizer.pipe(dataset, disable=['parser', 'ner'], batch_size=512, n_threads=1)):
        if ns:
            tokens = stop_word_remove(sample)
        elif ek:
            tokens = entity_keep(sample)
        else:
            tokens = lower_case_remove_punc(sample)
        token_dataset.append(tokens)
        if save_tokens:
            all_tokens += tokens

    return token_dataset, all_tokens
    
##########Vocab Builder##########

def build_vocab(all_tokens, max_vocab_size, ngram="ngram"):
    if ngram != "ngram":
        for i in range(len(all_tokens)):
            if i == 0:
                token_counter = Counter(all_tokens[i])
            else:
                token_counter.update(all_tokens[i])
    else:
        token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = 0 
    token2id['<unk>'] = 1
    return token2id, id2token

##########Token & Index##########

def token2index_dataset(tokens_data, token2id, id2token):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else 1 for token in tokens]
        indices_data.append(index_list)
    return indices_data

def token2index_bigram(tokens_data, token2id, id2token):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[(tokens[i], tokens[i+1])] if (tokens[i], tokens[i+1]) in token2id else 1 for i in range(len(tokens) - 1)]
        indices_data.append(index_list)
    return indices_data

def token2index_trigram(tokens_data, token2id, id2token):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[(tokens[i], tokens[i+1], tokens[i+2])] if (tokens[i], tokens[i+1], tokens[i+2]) in token2id else 1 for i in range(len(tokens) - 2)]
        indices_data.append(index_list)
    return indices_data

def token2index_tetragram(tokens_data, token2id, id2token):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[(tokens[i], tokens[i+1], tokens[i+2], tokens[i+3])] if (tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]) in token2id else 1 for i in range(len(tokens) - 3)]
        indices_data.append(index_list)
    return indices_data

##########DataLoader##########

class ReviewDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of review tokens 
        @param target_list: list of review targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))
        
        self.msl = 200

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:self.msl]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def review_collate_func(batch, max_sentence_length=200):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,max_sentence_length-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]
    
##########BagOfWords##########

class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,20)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = self.linear(out.float())
        return out
        
##########TestModel##########

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


##########Target Data########
train_targets = pkl.load(open("train_targets.p", "rb"))
val_targets = pkl.load(open("val_targets.p", "rb"))
test_targets = pkl.load(open("test_targets.p", "rb"))

##########Ablater##########

def ablate(train_data_tokens, all_train_tokens, val_data_tokens, ngrams, vocab_size, emb_dim, learning_rate, optim, weight_decay=0):
        
        
    if ngrams == "bigram":
        train_tokens = [[(train_data_tokens[j][i],  train_data_tokens[j][i + 1]) for i in range(len(train_data_tokens[j]) - 1)] for j in range(len(train_data_tokens))]
        token2id, id2token = build_vocab(all_train_tokens, vocab_size, "bigram")
        train_data_indices = token2index_bigram(train_data_tokens, token2id, id2token)
        val_data_indices = token2index_bigram(val_data_tokens, token2id, id2token)
    elif ngrams == "trigram":
        train_tokens = [[(train_data_tokens[j][i],  train_data_tokens[j][i+1], train_data_tokens[j][i+2]) for i in range(len(train_data_tokens[j]) - 2)] for j in range(len(train_data_tokens))]
        token2id, id2token = build_vocab(all_train_tokens, vocab_size, "trigram")
        train_data_indices = token2index_trigram(train_data_tokens, token2id, id2token)
        val_data_indices = token2index_trigram(val_data_tokens, token2id, id2token)
    elif ngrams == "tetragram":
        train_tokens = [[(train_data_tokens[j][i],  train_data_tokens[j][i+1], train_data_tokens[j][i+2], train_data_tokens[j][i+3]) for i in range(len(train_data_tokens[j]) - 3)] for j in range(len(train_data_tokens))]
        token2id, id2token = build_vocab(all_train_tokens, vocab_size, "tetragram")
        train_data_indices = token2index_tetragram(train_data_tokens, token2id, id2token)
        val_data_indices = token2index_tetragram(val_data_tokens, token2id, id2token)
    else:
        train_tokens = train_data_tokens.copy()
        token2id, id2token = build_vocab(all_train_tokens, vocab_size)
        train_data_indices = token2index_dataset(train_data_tokens, token2id, id2token)
        val_data_indices = token2index_dataset(val_data_tokens, token2id, id2token)
    
    BATCH_SIZE = 50
    train_dataset = ReviewDataset(train_data_indices, train_targets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=review_collate_func,
                                               shuffle=True)

    val_dataset = ReviewDataset(val_data_indices, val_targets)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=BATCH_SIZE,
                                               collate_fn=review_collate_func,
                                               shuffle=True)
    
    model = BagOfWords(len(id2token), emb_dim)
    
    num_epochs = 10

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()  
    if optim == "Adam":
        if type(learning_rate) == tuple:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[1], weight_decay = weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    else:
        if type(learning_rate) == tuple:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate[1], weight_decay = weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    if type(learning_rate) == tuple:
        if len(learning_rate) == 3:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning_rate[2])
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_accuracies = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    
    train_acc, val_acc = 0, 0

    for epoch in range(num_epochs):
        if type(learning_rate) == tuple:
            scheduler.step()
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()        
                
        train_accuracies[epoch] = test_model(train_loader, model)
        val_accuracies[epoch] = test_model(val_loader, model)
        
    return train_accuracies, val_accuracies

##########Test Model Accuracy##########
def test_final_model(train_data_tokens, all_train_tokens, test_data_tokens, vocab_size, emb_dim, learning_rate, optim, weight_decay):
    train_tokens = train_data_tokens.copy()
    token2id, id2token = build_vocab(all_train_tokens, vocab_size)
    train_data_indices = token2index_dataset(train_data_tokens, token2id, id2token)
    test_data_indices = token2index_dataset(test_data_tokens, token2id, id2token)

    BATCH_SIZE = 50
    train_dataset = ReviewDataset(train_data_indices, train_targets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=review_collate_func,
                                                   shuffle=True)

    test_dataset = ReviewDataset(test_data_indices, test_targets)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=review_collate_func,
                                                   shuffle=True)


    
    model = BagOfWords(len(id2token), emb_dim)
    
    num_epochs = 5

    if type(learning_rate) == tuple:
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning_rate[1])
    else:
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        if type(learning_rate) == tuple:
            scheduler.step()
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()        
    
    return test_model(test_loader, model)


###########Test Model Return Model##########
def test_final_model_model(train_data_tokens, all_train_tokens, test_data_tokens, vocab_size, emb_dim, learning_rate, optim, weight_decay):
    train_tokens = train_data_tokens.copy()
    token2id, id2token = build_vocab(all_train_tokens, vocab_size)
    train_data_indices = token2index_dataset(train_data_tokens, token2id, id2token)
    test_data_indices = token2index_dataset(test_data_tokens, token2id, id2token)

    BATCH_SIZE = 50
    train_dataset = ReviewDataset(train_data_indices, train_targets)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=review_collate_func,
                                                   shuffle=True)

    test_dataset = ReviewDataset(test_data_indices, test_targets)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=review_collate_func,
                                                   shuffle=True)

    model = BagOfWords(len(id2token), emb_dim)
    
    num_epochs = 5

    if type(learning_rate) == tuple:
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=learning_rate[1])
    else:
        criterion = torch.nn.CrossEntropyLoss()  
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        if type(learning_rate) == tuple:
            scheduler.step()
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()        
    
    return model
    
