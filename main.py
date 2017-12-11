import numpy as np
import urllib
import time
import re

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_bible(url='http://www.godrules.net/downloads/web_utf8_FINAL.txt'):
    my_txt = urllib.urlopen(url).read()

    str_bible = re.sub('web.*', '', my_txt)
    str_bible = re.sub('<br>', '', str_bible)
    str_bible = str_bible.replace('\n\n', ' ')
    str_bible = str_bible.lower()

    return str_bible


my_str = get_bible()

vocab = sorted(set(my_str))
count = {elem: my_str.count(elem) for elem in vocab}

print my_str[:1000]
print count


def sample_generator(my_str, sample_len, balanced=False):
    """Retruns a sample of the dataset (my_str) with lenght <sample_len>"""
    count = {elem: my_str.count(elem) for elem in vocab}

    while True:
        permutations = list(np.random.permutation(len(my_str) - sample_len))
        while len(permutations) > 1:
            rand_idx = permutations.pop()
            sample = my_str[rand_idx: rand_idx + sample_len]
            if balanced is True:
                # yield a sample if statistically acurate
                # 2000 should be chosen depending on dataset
                y = sample[-1]
                if np.random.randint(count[y]) < 2000:
                    yield sample
            else:
                # Always yield a sample
                yield sample


def batch_generator(batch_size, n_items):
    """Return a random sample"""
    s_gen = sample_generator(n_items)
    while True:
        # Generate a batch
        batch = []
        for i in range(batch_size):
            batch.append(s_gen.next())
        yield batch


def encode_sample(sample, one_hot_dimention=0):
    sample_encoded = []  # np.ndarray(len(sample), dtype='float')
    for idx_sample in range(len(sample)):
        sample_encoded.append(vocab.index(sample[idx_sample]))

    _X = torch.LongTensor(sample_encoded)
    X = autograd.Variable(_X)
    return X.cuda()


def encode_batch(batch, one_hot_dimention=0):
    """Takes a batch of string as input and encode it to a numerical
    batch"""
    batch_new = np.ndarray((len(batch), len(batch[0])), dtype='int')
    for i in range(len(batch)):
        for j in range(len(batch[0])):
            batch_new[i][j] = vocab.index(batch[i][j])

    if one_hot_dimention != 0:
        # Build a one-hot representation
        batch_one_hot = np.zeros((batch_new.shape + (one_hot_dimention,)))
        for i in range(batch_new.shape[0]):
            for j in range(batch_new.shape[1]):
                batch_one_hot[i, j, batch_new[i, j]] = 1
        return batch_one_hot[:, :-1, :], batch_one_hot[:, -1, :]

    return batch_new[:, :-1], batch_new[:, -1]


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.conv1 = CausalConv1d(embedding_dim, embedding_dim, 2)
        self.conv2 = CausalConv1d(embedding_dim, embedding_dim, 2, stride=2)
        self.conv3 = CausalConv1d(embedding_dim, embedding_dim, 2, stride=4)
        self.conv4 = CausalConv1d(embedding_dim, embedding_dim, 2, stride=8)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)
        self.hidden2 = self.init_hidden()

        # The linear layer that predicts next input
        self.hidden2pred = nn.Linear(hidden_dim, tagset_size)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        conv_1 = F.selu(self.conv1(embeds))
        conv_2 = conv_1.add(F.selu(self.conv2(conv_1)))
        conv_3 = conv_2.add(F.selu(self.conv3(conv_2)))
        conv_4 = conv_3.add(F.selu(self.conv4(conv_3)))
        lstm_out, self.hidden = self.lstm(conv_4.view(len(sentence), 1, -1), self.hidden)
        pred_space = self.hidden2pred(lstm_out.view(len(sentence), -1))
        pred_scores = F.log_softmax(pred_space)
        return pred_scores

    # def forward(self, sentence):
    #     embeds = self.word_embeddings(sentence)
    #     lstm_out,  self.hidden  = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
    #     lstm_out2, self.hidden2 = self.lstm2(lstm_out.view(len(sentence), 1, -1), self.hidden2)
    #     pred_space = self.hidden2pred(lstm_out2.view(len(sentence), -1))
    #     pred_scores = F.log_softmax(pred_space)
    #     return pred_scores


sentense_size = 200
HIDDEN_DIM = 64

model = LSTMTagger(sentense_size,
                   HIDDEN_DIM,
                   vocab_size=len(vocab),
                   tagset_size=len(vocab))
model.cuda()
loss_function = nn.NLLLoss().cuda()

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
gen = sample_generator(my_str, sentense_size + 1)
inputs = encode_sample(gen.next()[: -1])
pred = model(inputs)
print(pred)

np_loss = 0
lr = 0.002
t_start = time.time()
optimizer = optim.SGD(model.parameters(), lr=lr)
for iter in range(1000000):
    sample = gen.next()
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Variables of word indices.
    all_sentense = encode_sample(sample)
    sentence_in = all_sentense[: -1]
    targets = all_sentense[1:]

    # Step 3. Run our forward pass.
    pred = model(sentence_in)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(pred[10:], targets[10:])
    np_loss += loss.cpu().data.numpy()[0]
    loss.backward(retain_graph=True)
    optimizer.step()

    if iter % 200 == 0:  # Test function
        t_total = time.time() - t_start
        t_start = time.time()
        sample = 'i love the green banana that is growing in my garden of love that our holly god gave as a tribute to mankind'
        sample = sample[: sentense_size]
        for _ in range(100):
            inputs = encode_sample(sample[-sentense_size:])
            pred = model(inputs)
            pred = pred.cpu().data.numpy()[-1]
            pred = vocab[pred.argmax()]
            sample += pred

        print iter, sample[50:].replace('\n', '_'), np_loss / 200, t_total, lr

        np_loss = 0
        lr *= 0.99
        optimizer = optim.SGD(model.parameters(), lr=lr)
