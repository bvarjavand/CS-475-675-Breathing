import torch
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

import glob
import pandas as pd
import numpy as np
X = []
y = []

files = glob.glob("./test/*.pkl")
for fname in files:
    df = pd.read_pickle(fname)
    Xtemp = df["Volume"]  # , "from_thresh", "diff"
    ytemp = df["Label"]
    X.append(Xtemp.values)
    y.append(ytemp.values)

# X = np.stack(X, axis=0) BROKE since seqs are diff lengths
# y = np.stack(y, axis=0)

vectorized_seqs = X
print(X[0].shape)

# pad with 0
seq_lengths = LongTensor(list(map(len, vectorized_seqs)))
print(seq_lengths)
seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
print(seq_tensor.shape)
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = LongTensor(seq)

# sort
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]

packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
print(packed_input)
print(packed_input.data.shape)

# packed_output, (ht, ct) = lstm(packed_input)