import random, math, os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd

from SSIM.models.SSIM_model import Encoder,Decoder,Seq2Seq

from SSIM.utils.early_stopping import EarlyStopping

def data_preprocessing(data):
    
    # Normalization
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data).astype(np.float32)
    
    return train_test_split(norm_data,
                           test_size = 0.2,
                           shuffle = False)

def eval_val(model, val_data, loss_train):
    val_loss = 0.0
    n_counter = 0
    
    for x,y,before_len in val_data:
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        before_len = before_len.as_in_context(ctx)
        with autograd.predict_mode():
            out = model(x, y, before_len)
            val_loss += loss_train(out,y).mean().asscalar()
            n_counter += 1
            
    return val_loss / n_counter

def train_s2s(model, data_iters, test_iters, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                           'adam', {'learning_rate': lr})
    loss_train = gluon.loss.L1Loss()
    
    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, num_epochs + 1):
        for i, (seq, trg, before_len) in enumerate(data_iters):
            seq = seq.as_in_context(ctx)
            trg = trg.as_in_context(ctx)
            before_len = before_len.as_in_context(ctx)
            with autograd.record():
                trg_hat = model(seq, trg, before_len)
                loss = loss_train(trg_hat, trg)
                
            loss.backward()
            
            grads = [i.grad(ctx) for i in model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, CLIP)
            
            trainer.step(BATCH_SIZE)
            
        train_epoch_loss = eval_val(model, data_iters, loss_train)
        val_epoch_loss = eval_val(model, test_iters, loss_train)
        
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_epoch_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if epoch%NMONITOR == 0:
            print("epoch::{}, train_loss::{}, val_loss::{}".format(epoch,
                         train_epoch_loss, val_epoch_loss))
            

if __name__ == "__main__":
    
    # Data
    data_generator = time_data_gen(steps_num=50)
    
    time_series_data = data_generator.random_walking_gen()
    #time_series_data = 
    #src_data, trg_data = data_generator.data_prep()
    
    data_tuple = data_preprocessing(time_series_data)
    
    src_data, trg_data, before_len_data = data_generator.sliding(data_tuple[0])
    src_test, trg_test, before_len_test = data_generator.sliding(data_tuple[1])
    
    # Model hyperparameter
    INPUT_DIM = src_data.shape[-1]
    OUTPUT_DIM = trg_data.shape[-1]
    ENC_HID_DIM = 20
    DEC_HID_DIM = 20
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    NUM_LAYER = 2
    LR = 0.001 # learning rate
    CLIP = 1
    EPOCH = 50
    BATCH_SIZE = 100
    NMONITOR = 10
    
    # Model
    enc = Encoder(input_dim = INPUT_DIM, 
                  enc_hid_dim = ENC_HID_DIM, 
                  dec_hid_dim = DEC_HID_DIM, 
                  num_layers = NUM_LAYER,
                  dropout_p = ENC_DROPOUT)
    dec = Decoder(output_dim = OUTPUT_DIM,
                  enc_hid_dim = ENC_HID_DIM, 
                  dec_hid_dim = DEC_HID_DIM, 
                  num_layers = NUM_LAYER,
                  dropout_p = DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec)
    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    
    train_data = gluon.data.DataLoader(
                       gluon.data.ArrayDataset(src_data, trg_data, before_len_data),
                       batch_size=BATCH_SIZE, shuffle=False)
    test_data = gluon.data.DataLoader(
                       gluon.data.ArrayDataset(src_test, trg_test, before_len_test),
                       batch_size=BATCH_SIZE, shuffle=False)
    
    train_s2s(model, train_data, test_data, LR, EPOCH, ctx)
