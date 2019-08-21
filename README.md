# SSIM Model

This is the SSIM model for *[SSIM—A Deep Learning Approach for Recovering Missing Time Series Sensor Data](https://ieeexplore.ieee.org/document/8681112)*

Considering the dataset we are using in the paper is not public available, we use a different open dataset for demo. 

The original PM2.5 data can be download from: [PM2.5](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

The Pytorch implementation has not been fully tested.
Bugs may be fixed later.

***

Code structure:

/checkpoints ------- store trained model

/data        ------- data set

/model       ------- SSIM model: encoder, decoder, attention

/utils

/prepare_PM2.5 ------------ prepare train/test for PM2.5 data. 2010-2013 for train, 2014 for test

/VLSM --------------- VLSM algorithm to generate variable length samples (with 0 pad)
     
