from mxnet import autograd, nd, gluon, init
from mxnet.gluon import rnn, nn
import numpy as np
import mxnet as mx
import random

########### Model ##########


class Encoder(nn.Block):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, 
                 num_layers, dropout_p = 0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        
        self.input_dense = nn.Dense(units = self.enc_hid_dim, 
                                    in_units = self.input_dim,
                                    flatten = False,
                                    activation='tanh')
        self.lstm = rnn.LSTM(hidden_size=self.enc_hid_dim, 
                             num_layers=self.num_layers,
                            bidirectional=True)
        self.output_dense = nn.Dense(units = self.dec_hid_dim, 
                                     in_units = self.enc_hid_dim * 2,
                                     flatten = False,
                                     activation='tanh')
        self.dropout = nn.Dropout(self.dropout_p)
        
    def forward(self, input_seq):
        embedded = self.dropout(self.input_dense(input_seq))
        
        embedded = embedded.swapaxes(0, 1)
        ini_state = self.lstm.begin_state(batch_size = embedded.shape[1],
                                         ctx = embedded.context)
        outputs, out_states = self.lstm(embedded, ini_state)
        
        hidden = out_states[0]
        cell = out_states[1]
        
        hidden = nd.concat(hidden[0:self.num_layers, :, :], 
                           hidden[self.num_layers:, :, :], 
                           dim = 2)
        cell = nd.concat(cell[0:self.num_layers, :, :], 
                         cell[self.num_layers:, :, :], 
                         dim = 2)
        
        hidden = self.output_dense(hidden)
        cell = self.output_dense(cell)
        
        return outputs, [hidden, cell]



class Global_Attention(nn.Block):
    def __init__(self, units, dropout, **kwargs):
        super(Global_Attention, self).__init__(**kwargs)
        
        self.w_k = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.w_q = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)
        
    def masked_softmax(self, X, valid_length):
        # X: 3-D tensor, valid_length: 1-D or 2-D tensor
        if valid_length is None:
            return nd.softmax(X)
        else:
            shape = X.shape
            if valid_length.ndim == 1:
                valid_length = valid_length.repeat(shape[1], axis=0)

            # Fill masked elements with a large negative, whose exp is 0
            X = nd.where(valid_length, 
                         X.reshape(-1, shape[-1]),
                         -1e6 * nd.ones_like(X.reshape(-1, shape[-1])))
        return nd.softmax(X).reshape(shape)
        
    def forward(self, query, key, value, valid_length):
        query, key = self.w_k(query), self.w_q(key)
        # Expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast
        features = nd.expand_dims(query, axis=2) + nd.expand_dims(key, axis=1)
        scores = nd.squeeze(self.v(features), axis=-1)
        attention_weights = self.dropout(self.masked_softmax(scores, valid_length))
        
        return nd.batch_dot(attention_weights, value)

class Decoder(nn.Block):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, 
                 num_layers, dropout_p = 0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.attention = Global_Attention(self.dec_hid_dim, 
                                   dropout_p)
        
        self.input_dense = nn.Dense(units = self.dec_hid_dim, 
                                    in_units = self.output_dim,
                                    flatten = False,
                                    activation='tanh')
        self.lstm = rnn.LSTM(hidden_size=self.dec_hid_dim,
                             num_layers=self.num_layers)
        self.out = nn.Dense(units = self.output_dim,
                            in_units = self.enc_hid_dim * 2 + self.dec_hid_dim,
                            flatten = False)
        self.dropout = nn.Dropout(self.dropout_p)
        
    def init_state(self, src_seq, enc_outputs, enc_valid_len, len_before):
        outputs, hidden_state = enc_outputs
        
        # Find the last value before the target sequence
        init_output = src_seq[nd.arange(src_seq.shape[0]), 
                              len_before - 1, 
                              :]
        
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.swapaxes(0, 1), 
                hidden_state, 
                enc_valid_len,
                init_output)
    
    def forward(self, trg_seq, state, teacher_forcing = 0.5):
        enc_outputs, hidden_state, enc_valid_len, init_out = state
        
        trg_seq = trg_seq.swapaxes(0, 1)
            
        outputs_list = []
        
        out = init_out
        for t, dec_input in enumerate(trg_seq):
            lstm_input = self.dropout(self.input_dense(out))
        
            # query shape: (batch_size, 1, hidden_size)
            query = nd.expand_dims(hidden_state[0][-1], axis=1)
            # Context has same shape as query
            context = self.attention(query, 
                                     enc_outputs, 
                                     enc_outputs, 
                                     enc_valid_len)
            # Concatenate on the feature dimension
            lstm_input = nd.concat(context, 
                                  nd.expand_dims(lstm_input, axis=1), 
                                  dim=-1)
            # Reshape dec_input to (1, batch_size, embed_size + hidden_size)
            out, hidden_state = self.lstm(lstm_input.swapaxes(0, 1),
                                         hidden_state)
            
            out = self.out(nd.concat(context, out.swapaxes(0, 1), dim=-1))
            
            outputs_list.append(out.swapaxes(0, 1))
            
            teacher = random.random() < teacher_forcing
            
            out = (dec_input if teacher else out.swapaxes(0, 1)[0])
        
        outputs = outputs_list[0]
        for output_one in outputs_list[1:]:
            outputs = nd.concat(outputs, output_one, dim = 0)
        outputs = outputs.swapaxes(0, 1)
        
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_len]


class Seq2Seq(nn.Block):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def create_mask(self, src):
        
        mask = np.prod((src != 0).asnumpy(), axis = 2)
        return nd.array(mask)
        
    def forward(self, src, trg, before_len, 
                teacher_forcing = 0.5):
        
        encoder_outputs = self.encoder(src)
        
        masking_array = self.create_mask(src)
        ini_state = self.decoder.init_state(src, 
                                encoder_outputs,
                                 masking_array,
                                before_len)
        out, state = self.decoder(trg, ini_state,
                            teacher_forcing)
        
        return out

