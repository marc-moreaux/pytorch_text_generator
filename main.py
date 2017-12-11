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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier(m.weight.data)
        xavier(m.bias.data)


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

    def __init__(self, sentense_len, hidden_dim, vocab_size, n_layers=5):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.word_embeddings = nn.Embedding(sentense_len, vocab_size)

        # Initialize the conv layers
        self.conv = []
        self.conv.append(CausalConv1d(vocab_size, hidden_dim, 2).cuda())
        for i in range(n_layers - 2):
            self.conv.append(CausalConv1d(hidden_dim,
                                          hidden_dim,
                                          kernel_size=2,
                                          dilation=2 ** (i + 1)).cuda())
        self.conv.append(CausalConv1d(hidden_dim, vocab_size, 2, 2 ** n_layers).cuda())

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        _conv = F.selu(self.conv[0](embeds))
        for i in range(self.n_layers - 2):
            _conv = _conv.add(F.selu(self.conv[i + 1](_conv)))

        _conv = F.selu(self.conv[self.n_layers-1](_conv))
        pred_scores = F.log_softmax(_conv.view(_conv.data.shape[:2]))
        return pred_scores


sentense_size = 200
HIDDEN_DIM = 64

model = LSTMTagger(sentense_size,
                   HIDDEN_DIM,
                   vocab_size=len(vocab))
model.apply(weight_init)
model.cuda()
loss_function = nn.NLLLoss().cuda()

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
gen = sample_generator(my_str, sentense_size + 1)
inputs = encode_sample(gen.next()[: -1])
pred = model(inputs)
print(pred)

np_loss = 0
lr = 0.5
t_start = time.time()
optimizer = optim.SGD(model.parameters(), lr=lr)
for iter in range(1000000):
    model.zero_grad()

    # Get next sample
    sample = gen.next()
    all_sentense = encode_sample(sample)
    sentence_in = all_sentense[: -1]
    targets = all_sentense[1:]

    # Predict
    pred = model(sentence_in)
    loss = loss_function(pred[10:], targets[10:])
    np_loss += loss.cpu().data.numpy()[0]

    # Update
    loss.backward(retain_graph=True)
    optimizer.step()

    # Maybe show stg
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
