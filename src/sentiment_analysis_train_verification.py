import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset

from file_processing import file_path_train
from lstm_model_define import train_on_gpu, SentimentLSTM


#
# Read data from text files
#

with open(file_path_train + "train_reviews.txt", "r") as f:
    reviews = f.read()
with open(file_path_train + "train_labels.txt", "r") as f:
    labels = f.read()

#
# Data Processing — convert to lower case
#

reviews = reviews.lower()

#    
# Data Processing — remove punctuation
#

all_text = "".join([c for c in reviews if c not in punctuation])

#
# Data Processing — create list of reviews and labels
#

labels_split = labels.split("\n")
reviews_split = all_text.split("\n")
#print("\nNumber of reviews :", len(reviews_split))

#
# Tokenize — Create Vocab to Int mapping dictionary
#

all_text2 = " ".join(reviews_split)
words = all_text2.split() # create a list of words
count_words = Counter(words) # Count all the words using Counter Method
total_words = len(words)
sorted_words = count_words.most_common(total_words)
#print(count_words)

vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
#print(vocab_to_int)

#
# Tokenize — Encode the words
#

reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
#print (reviews_int[0:3])

#
# Tokenize — Encode the labels
#

encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
encoded_labels = np.array(encoded_labels)
#print(encoded_labels)

#
# Analyze Reviews Length
#

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
#plt.show()
pd.Series(reviews_len).describe()

#
# Removing Outliers — Getting rid of extremely long or short reviews
#

reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
encoded_labels = [ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]

#
# Padding / Truncating the remaining data
#

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
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

seq_length = 240
features = pad_features(reviews_int, seq_length)    
#print(features[:10,:])

#
# Training, Validation, Test Dataset Split
#

split_frac = 0.8
len_feat = features.shape[0]
#print(len_feat)

train_x = features[0:int(split_frac*len_feat)]
train_y = encoded_labels[0:int(split_frac*len_feat)]

remaining_x = features[int(split_frac*len_feat):]
remaining_y = encoded_labels[int(split_frac*len_feat):]

valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]

test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

#
# Dataloaders and Batching
#

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(np.array(train_x)), torch.from_numpy(np.array(train_y)))
valid_data = TensorDataset(torch.from_numpy(np.array(valid_x)), torch.from_numpy(np.array(valid_y)))
test_data = TensorDataset(torch.from_numpy(np.array(test_x)), torch.from_numpy(np.array(test_y)))

# dataloaders
batch_size = 50

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
#dataiter = iter(train_loader)
#sample_x, sample_y = dataiter.next()
#print('Sample input size: ', sample_x.size()) # batch_size, seq_length
#print('Sample input: \n', sample_x)
#print()
#print('Sample label size: ', sample_y.size()) # batch_size
#print('Sample label: \n', sample_y)

#
# Training the Network
#

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
#print(net)

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
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            
#
# Testing
#

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
print("\nTest loss for 10 percent internal data: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy for 10 percent internal data: {:.3f}".format(test_acc))

#
# Testing on User-generated Reviews
#

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = "".join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints

# test code and generate tokenized review
#test_ints = tokenize_review(test_review_neg)
#print(test_ints)

# test sequence padding
#seq_length = 240 
#features = pad_features(test_ints, seq_length)
#print(features)

# test conversion to tensor and pass into your model
#feature_tensor = torch.from_numpy(features)
#print(feature_tensor.size())

def predict(net, test_review, sequence_length=240):
    
    net.eval()
    
    # tokenize review
    test_ints = tokenize_review(test_review)
    
    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    
    # get the output from the model
    output, h = net(feature_tensor, h)
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('\nPrediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")

#
# Prediction
#

test_review_pos = "This movie had the best acting and the dialogue was so good. I loved it."
test_review_neg = "This movie sucks, to be honest, dude, I'll never watch this movie again."

seq_length = 240 # good to use the length that was trained on

predict(net, test_review_pos, seq_length)