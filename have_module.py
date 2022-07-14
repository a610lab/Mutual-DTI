# -*- coding: utf-8 -*-
import os
import pickle
import sys
import timeit
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score,accuracy_score

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        protein = torch.unsqueeze(protein, 0)
        # print(protein.size()) #[1, 777, 10]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # print(conv_input.size())
        #[1, 10, 777]
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]
        trg_new = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg_new = self.ln(trg_new + self.do(self.ea(trg_new, src, src, src_mask)))

        # trg_new = self.ln(trg_new + self.do(self.pf(trg_new)))

        src_new = self.ln(src + self.do(self.sa(src, src, src, src_mask)))

        src_new = self.ln(src_new + self.do(self.ea(src_new, trg, trg, trg_mask)))

        # src_new = self.ln(src_new + self.do(self.pf(src_new)))

        return trg_new, src_new


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim*2, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # print(trg.size())   #[1, 24, 10]
        # print(src.size())   # [1, 777, 10]

        # trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]
        for layer in self.layers:
            trg, src = layer(trg, src)
        # trg = [batch size, compound len, hid dim]
        # """Use norm to determine which atom is significant. """
        # norm = torch.norm(trg, dim=2)
        # # norm = [batch size,compound len]
        # norm = F.softmax(norm, dim=1)
        # # norm = [batch size,compound len]
        # trg = torch.squeeze(trg, dim=0).to('cpu')
        # norm = torch.squeeze(norm, dim=0).to('cpu')
        # sum_atom = torch.zeros((self.hid_dim)).to('cpu')
        # for i in range(norm.shape[0]):
        #     sum_atom += (trg[i,]*norm[i])
        # sum_atom = sum_atom.unsqueeze(dim=0).to(self.device)
        # """Use norm to determine which protein is significant. """
        # norm = torch.norm(src, dim=2)
        # # norm = [batch size,protein len]
        # norm = F.softmax(norm, dim=1)
        # # norm = [batch size,protein len]
        # src = torch.squeeze(src, dim=0).to('cpu')
        # norm = torch.squeeze(norm, dim=0).to('cpu')
        # sum_protein = torch.zeros((self.hid_dim)).to('cpu')
        # for i in range(norm.shape[0]):
        #     sum_protein += (src[i,] * norm[i])
        # sum_protein = sum_protein.unsqueeze(dim=0).to(self.device)
        # sum = torch.cat((sum_atom,sum_protein),1)
        trg = torch.mean(trg, 1)
        src = torch.mean(src, 1)
        sum = torch.cat((trg, src), 1)
        # trg = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        # trg = [batch size,hid_dim]
        return label


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, encoder, decoder):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
#        self.W_cnn = nn.ModuleList([nn.Conv2d(
#                     in_channels=1, out_channels=1, kernel_size=2*window+1,
#                     stride=1, padding=window) for _ in range(layer_cnn)])
#         self.bilstm = nn.LSTM(dim, 5, 1, dropout=0.2, bidirectional=True)


        # self.W_attention = nn.Linear(dim, dim)
        # self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
        #                             for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        # print(xs.size()) #[24, 10]
        # print(A.size())  #[24, 24]
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(xs, 0)

#     def attention_cnn(self, x, xs, layer):
#         """The attention mechanism is applied to the last layer of CNN."""
#         # print(xs.size()) #[777, 10]
#
#         # xs = torch.unsqueeze(xs, 0)
#         # print(xs.size()) #[1, 777, 10]
# #        for i in range(layer):
# #            xs = torch.relu(self.W_cnn[i](xs))
# #         bilstms, _ =self.bilstm(xs)
#         # print(bilstms.size())#[1, 777, 10]
#         #grus,_=self.gru(torch.squeeze(xs,0))
#
#         # bilstms = torch.squeeze(bilstms, 0)
#         # print(bilstms.size())#[777, 10]
#         #xs = torch.squeeze(grus, 0)
#         #print(self.W_attention(xs).size())
#
#         bilstms = torch.squeeze(xs, 0)
#         h = torch.relu(self.W_attention(x))
#         hs = torch.relu(self.W_attention(bilstms))
#         weights = torch.tanh(F.linear(h, hs))
#         ys = torch.t(weights) * hs
#
#         # return torch.unsqueeze(torch.sum(ys, 0), 0)
#         return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        # print(compound_vector.size()) #[1, 24, 10]
        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
#        print(word_vectors.dim())
#         print(compound_vector.size())
#         print(word_vectors.size())
#         print(len(word_vectors))
        word_vectors = self.encoder(word_vectors)
        # print(word_vectors.size()) #[1, 777, 10]
        interaction = self.decoder(compound_vector, word_vectors)
        # protein_vector = self.attention_cnn(compound_vector,
        #                                     word_vectors, layer_cnn)

        # """Concatenate the above two vectors and output the interaction."""
        # cat_vector = torch.cat((compound_vector, protein_vector), 1)
        # print(cat_vector.size())
        # for j in range(layer_output):
        #     cat_vector = torch.relu(self.W_out[j](cat_vector))
        # interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model,batch):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, dataset):
        self.model.train()
        N = len(dataset)
        #print(N)
        loss_total = 0
        i = 0
        for data in dataset:
            i =i + 1
            loss = self.model(data)
            loss = loss / self.batch
            loss.backward()
            if i % self.batch ==0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        #print(N)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        Accuracy = accuracy_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall, Accuracy

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def init_seeds(seed):
    torch.manual_seed(seed)
    # sets the seed for generating random numbers.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.cuda.manual_seed_all(seed)
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy' ,allow_pickle=True) ]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



if __name__ == "__main__":

    # """Hyperparameters."""
    # (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
    #  lr, lr_decay, decay_interval, weight_decay, iteration,
    #  setting) = ['celegans', 2, 3, 10, 3, 11, 3, 3, 1e-3, 0.5, 10, 1e-6, 50,
    #  'celegans--radius2--ngram3--dim10--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-1234--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration100']
    # (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
    #  iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
    #                         decay_interval, iteration])
    # lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
    DATASET = 'GPCR'
    batch = 128
    radius = 2
    ngram = 3
    lr = 5e-3
    lr_decay = 0.5
    weight_decay = 1e-6
    decay_interval = 5
    atom_dim = 10
    protein_dim=10
    hid_dim = 10
    layer_gnn = 3
    layer_output = 3
    dim = 10
    n_layers = 3
    n_heads = 2
    pf_dim = 256
    dropout = 0.1
    kernel_size = 7
    iteration = 40
    setting = '{}-batch{}-radius{}-ngram{}-lr{}-lr_decay{}-weight_decay{}-decay_interval{}-atom_dim{}-protein_dim{}-' \
              'hid_dim{}-layer_gnn{}-layer_output{}-dim{}-n_layers{}-n_heads{}-pf_dim{}-dropout{}-kernel_size{}-iteration{}' \
        .format(DATASET, batch, radius, ngram, lr, lr_decay, weight_decay, decay_interval, atom_dim,protein_dim, hid_dim, layer_gnn,
                layer_output, dim, n_layers, n_heads, pf_dim, dropout, kernel_size, iteration)
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/input/'
                'radius' + str(radius) + '_ngram' + str(ngram) + '/')
#     dir_input = ('../dataset/' + DATASET + '/input/for_train13/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    train_dataset, dataset_test = split_dataset(dataset, 13022)
    """Create a dataset and split it into train/dev/test."""
    seeds = [8,10,12,14,18,20]
    for seed in seeds:
        dataset_train = shuffle_dataset(train_dataset, seed)
        dataset_train, dataset_dev = split_dataset(dataset_train, 0.8 * 13022)
        dataset_test = shuffle_dataset(dataset_test, seed)


        """Output files."""
        path = 'have_Attention_output/'+DATASET+'/'+str(seed)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        # file_AUCs = path + setting + '.txt'
        file_model = path + setting
        # AUCs = ('AUC_test\tPrecision_test\tRecall_test\tAccuracy_test')
        # with open(file_AUCs, 'w') as f:
        #     f.write(AUCs + '\n')

        """Set a model."""
        init_seeds(seed)
        encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
        decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                          PositionwiseFeedforward, dropout, device)
        model = CompoundProteinInteractionPrediction(encoder, decoder).to(device)
        # model.load_state_dict(torch.load('../Pretask/'+file_model))
        trainer = Trainer(model,batch)
        dever = Tester(model)
        tester = Tester(model)

        """Start training."""
        print('Training...(transformerwen)')
        # print(AUCs)
        start = timeit.default_timer()
        max_AUC_dev = 0
        epoch_label = 0
        AUCdev = ('Epoch\ttime\tloss_train\tAUC_dev\tAUC_test\tPrecision_test\tRecall_test\tAccuracy_test')
        print(AUCdev)
        file_dev = path + setting+'-detail.txt'
        with open(file_dev, 'w') as f:
            f.write(AUCdev + '\n')
        for epoch in range(1, iteration+1):

            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            loss_train = trainer.train(dataset_train)
            AUC_dev, _, _, _ = dever.test(dataset_dev)
            AUC_test, precision_test, recall_test, Accuracy_test = tester.test(dataset_test)
            end = timeit.default_timer()
            time = end - start
            standard = [epoch, time, loss_train, AUC_dev,
                    AUC_test, precision_test, recall_test, Accuracy_test]
            if AUC_test > max_AUC_dev:
                dever.save_model(model, file_model)
                max_AUC_dev = AUC_test
                epoch_label = epoch
            dever.save_AUCs(standard,file_dev)
            print('\t'.join(map(str, standard)))
        print("The best model is epoch", epoch_label)
        # """加载model"""
        # testmodel = CompoundProteinInteractionPrediction(encoder,decoder).to(device)
        # testmodel.load_state_dict(torch.load(file_model))
        # tester = Tester(testmodel)
        # AUC_test, precision_test, recall_test, Accuracy_test = tester.test(dataset_test)
        # AUCs = [AUC_test, precision_test, recall_test, Accuracy_test]
        # tester.save_AUCs(AUCs, file_AUCs)