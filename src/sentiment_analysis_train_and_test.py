import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset

from file_processing import file_path_train, file_path_test, remove_old_files_if_existed, generate_new_files
from lstm_model_define import train_on_gpu, SentimentLSTM


#
# Padding / Truncating the remaining data
#

def pad_features(reviews_int, seq_length):
    
    # Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

#
# Tokenize train dataset 
#

def tokenize_train_dataset(reviews, labels, seq_length=240, batch_size=50, split_frac=0.9):
    
    # Data Processing — convert to lower case 
    reviews = reviews.lower()
     
    # Data Processing — remove punctuation
    all_text = "".join([c for c in reviews if c not in punctuation])
    
    # Data Processing — create list of reviews and labels
    labels_split = labels.split("\n")
    reviews_split = all_text.split("\n")
    
    # Tokenize — Create Vocab to Int mapping dictionary
    all_text2 = " ".join(reviews_split)
    words = all_text2.split() # create a list of words
    count_words = Counter(words) # Count all the words using Counter Method
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    
    # Tokenize — Encode the words
    reviews_int = []
    for review in reviews_split:
        r = [vocab_to_int[w] for w in review.split()]
        reviews_int.append(r)
    
    # Tokenize — Encode the labels
    encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
    encoded_labels = np.array(encoded_labels)
    
    # Analyze Reviews Length
    reviews_len = [len(x) for x in reviews_int]
    
    # Removing Outliers — Getting rid of extremely long or short reviews
    reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
    encoded_labels = [ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]
    
    features = pad_features(reviews_int, seq_length)    
    
    # Train dataset setup
    len_feat = features.shape[0]
    
    train_x = features[0:int(split_frac*len_feat)]
    train_y = encoded_labels[0:int(split_frac*len_feat)]
    
    valid_x = features[int(split_frac*len_feat):]
    valid_y = encoded_labels[int(split_frac*len_feat):]
    
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(np.array(train_x)), torch.from_numpy(np.array(train_y)))
    valid_data = TensorDataset(torch.from_numpy(np.array(valid_x)), torch.from_numpy(np.array(valid_y)))
    
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    
    return train_loader, valid_loader, vocab_to_int

#
# Tokenize test dataset 
#

def tokenize_test_dataset(reviews, labels, seq_length=240, batch_size=50):
    
    # Data Processing — convert to lower case 
    reviews = reviews.lower()
     
    # Data Processing — remove punctuation
    all_text = "".join([c for c in reviews if c not in punctuation])
    
    # Data Processing — create list of reviews and labels
    labels_split = labels.split("\n")
    reviews_split = all_text.split("\n")
    
    # Tokenize — Create Vocab to Int mapping dictionary
    all_text2 = " ".join(reviews_split)
    words = all_text2.split() # create a list of words
    count_words = Counter(words) # Count all the words using Counter Method
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    
    # Tokenize — Encode the words
    reviews_int = []
    for review in reviews_split:
        r = [vocab_to_int[w] for w in review.split()]
        reviews_int.append(r)
    
    # Tokenize — Encode the labels
    encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
    encoded_labels = np.array(encoded_labels)
    
    # Analyze Reviews Length
    reviews_len = [len(x) for x in reviews_int]
    
    # Removing Outliers — Getting rid of extremely long or short reviews
    reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
    encoded_labels = [ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]
    
    features = pad_features(reviews_int, seq_length)    
    
    # Train dataset setup
    test_x = features
    test_y = encoded_labels
    
    # create Tensor datasets
    test_data = TensorDataset(torch.from_numpy(np.array(test_x)), torch.from_numpy(np.array(test_y)))
    
    # make sure to SHUFFLE your data
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
    return test_loader

#
# Train the Network
#

def train(train_loader, valid_loader, vocab_to_int, batch_size):
    
    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    
    net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    
    # loss and optimization functions
    lr = 0.001
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # training params
    epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
    counter = 0
    print_every = 100
    clip = 5 # gradient clipping
    
    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()
    
    net.train()
    
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
    
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
    
            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
    
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
    
            # zero accumulated gradients
            net.zero_grad()
    
            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = net(inputs, h)
    
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
    
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
    
                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()
    
                    inputs = inputs.type(torch.LongTensor)
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
    
                    val_losses.append(val_loss.item())
    
                net.train()
                
                print("\nEpoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                
                
    return net, criterion
#
# Testing
#

def test(net, criterion, test_loader, batch_size):
    # Get test data loss and accuracy
    
    test_losses = [] # track loss
    num_correct = 0
    
    # init hidden state
    h = net.init_hidden(batch_size)
    
    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:
    
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
    
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        
        # get predicted outputs
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)
        
        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        
        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer
        
        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
    
    # -- stats! -- ##
    # avg test loss
    print("\nTest loss: {:.3f}".format(np.mean(test_losses)))
    
    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


#
# Main function
#

if __name__ == "__main__":
    
    # Define parameters
    seq_length = 240
    batch_size = 50
    split_frac = 0.9
    
    # Process files
    remove_old_files_if_existed()
    generate_new_files()
    
    # Read train and test datasets from txt files
    with open(file_path_train + "train_reviews.txt", "r") as f:
        train_reviews = f.read()
    with open(file_path_train + "train_labels.txt", "r") as f:
        train_labels = f.read()
    
    with open(file_path_test + "test_reviews.txt", "r") as f:
        test_reviews = f.read()
    with open(file_path_test + "test_labels.txt", "r") as f:
        test_labels = f.read()
        
    # Tokenize datasets
    train_loader, valid_loader, train_vocab_to_int = tokenize_train_dataset(train_reviews, train_labels, seq_length, batch_size, split_frac)
    test_loader = tokenize_test_dataset(test_reviews, test_labels, seq_length, batch_size)
    
    # Train lstm model
    net, criterion = train(train_loader, valid_loader, train_vocab_to_int, batch_size)
    
    # Test using test dataset
    test(net, criterion, test_loader, batch_size)
    
    print("\nTHE END")

    