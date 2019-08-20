import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random, math, os, time

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd

from SSIM.models.SSIM_model import Global_Attention,Encoder,Decoder,Seq2Seq

from SSIM.utils.early_stopping import EarlyStopping
from SSIM.utils.prepare_PM25 import test_pm25_single_station
from SSIM.utils.support import *



# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train(model, optimizer, criterion, X_train, y_train, x_len, x_before_len):
    # model.train()

    iter_per_epoch = int(np.ceil(X_train.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)

    n_iter = 0

    perm_idx = np.random.permutation(X_train.shape[0])

    # train for each batch

    for t_i in range(0, X_train.shape[0], BATCH_SIZE):
        batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

        x_train_batch = np.take(X_train, batch_idx, axis=0)
        y_train_batch = np.take(y_train, batch_idx, axis=0)
        x_len_batch = np.take(x_len, batch_idx, axis=0)
        x_before_len_batch = np.take(x_before_len, batch_idx, axis=0)

        loss = train_iteration(model, optimizer, criterion, CLIP, x_train_batch, y_train_batch, x_len_batch,
                               x_before_len_batch)

        iter_losses[t_i // BATCH_SIZE] = loss

        n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)])


def train_iteration(model, optimizer, criterion, clip, X_train, y_train, x_len, x_before_len):
    model.train()
    optimizer.zero_grad()

    X_train = np.transpose(X_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    X_train_tensor = numpy_to_tvar(X_train)
    y_train_tensor = numpy_to_tvar(y_train)
    x_train_len_tensor = numpy_to_tvar(x_len)
    x_train_before_len_tensor = numpy_to_tvar(x_before_len)

    output = model(X_train_tensor, y_train_tensor, x_train_len_tensor, x_train_before_len_tensor)

    output = output.view(-1)

    y_train_tensor = y_train_tensor.view(-1)

    loss = criterion(output, y_train_tensor)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    # loss_meter.add(loss.item())

    return loss.item()


### evaluate

def evaluate(model, criterion, X_test, y_test, x_len, x_before_len):
    # model.eval()

    epoch_loss = 0
    iter_per_epoch = int(np.ceil(X_test.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    perm_idx = np.random.permutation(X_test.shape[0])

    n_iter = 0

    with torch.no_grad():
        for t_i in range(0, X_test.shape[0], BATCH_SIZE):
            batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

            x_test_batch = np.take(X_test, batch_idx, axis=0)
            y_test_batch = np.take(y_test, batch_idx, axis=0)
            x_len_batch = np.take(x_len, batch_idx, axis=0)
            x_before_len_batch = np.take(x_before_len, batch_idx, axis=0)

            loss = evaluate_iteration(model, criterion, x_test_batch, y_test_batch, x_len_batch, x_before_len_batch)
            iter_losses[t_i // BATCH_SIZE] = loss

            n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)])


def evaluate_iteration(model, criterion, x_test, y_test, x_len, x_before_len):
    model.eval()

    x_test = np.transpose(x_test, [1, 0, 2])
    y_test = np.transpose(y_test, [1, 0, 2])

    x_test_tensor = numpy_to_tvar(x_test)
    y_test_tensor = numpy_to_tvar(y_test)
    x_test_len_tensor = numpy_to_tvar(x_len)
    x_test_before_len_tensor = numpy_to_tvar(x_before_len)

    output = model(x_test_tensor, y_test_tensor, x_test_len_tensor, x_test_before_len_tensor, 0)

    output = output.view(-1)
    y_test_tensor = y_test_tensor.view(-1)

    loss = criterion(output, y_test_tensor)

    # test_loss_meter.add(loss.item())

    # plot_result(output, y_test_tensor)
    # show_attention(x_test_tensor,output,decoder_attn)

    return loss.item()


if __name__ == "__main__":

    # model hyperparameters
    INPUT_DIM = 11
    OUTPUT_DIM = 1
    ENC_HID_DIM = 20
    DEC_HID_DIM = 20
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ECN_Layers = 2
    DEC_Layers = 2
    LR = 0.001  # learning rate
    CLIP = 1
    EPOCHS = 50
    BATCH_SIZE = 100

    # Data
    sampling_params = {
        'dim_in': 11,
        'output_length': 5,
        'min_before': 8,
        'max_before': 10,
        'min_after': 8,
        'max_after': 10,
        'test_size': 0.2
    }

    ## Different test data

    (x_train, y_train, x_train_len, x_train_before_len), (
    x_test, y_test, x_test_len, x_test_before_len) = test_pm25_single_station()

    # Model
    glob_attn = Global_Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_Layers, DEC_Layers, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_Layers, DEC_DROPOUT, glob_attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.MSELoss()

    # # visulization visdom
    # vis = Visualizer(env='attention')
    # loss_meter = meter.AverageValueMeter()
    # test_loss_meter = meter.AverageValueMeter()

    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):

        train_epoch_losses = np.zeros(EPOCHS)
        evaluate_epoch_losses = np.zeros(EPOCHS)
        # loss_meter.reset()

        print('Epoch:', epoch, 'LR:', scheduler.get_lr())

        start_time = time.time()
        train_loss = train(model, optimizer, criterion, x_train, y_train, x_train_len, x_train_before_len)
        valid_loss = evaluate(model, criterion, x_test, y_test, x_test_len, x_test_before_len)
        end_time = time.time()

        scheduler.step()

        # # visulization
        # vis.plot_many_stack({'train_loss': loss_meter.value()[0], 'test_loss': test_loss_meter.value()[0]})

        train_epoch_losses[epoch] = train_loss
        evaluate_epoch_losses[epoch] = valid_loss

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # # prediction
    #
    # model.load_state_dict(torch.load('checkpoint.pt'))
    #
    # test_loss = evaluate(model, criterion, X_test, y_test)
    #
    # plt.show()
    #
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
