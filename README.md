# BERT for document classification

Pure Python TF-IDF text classifier

## Prerequisites

torch, pandas, configargparse, sklearn

## Usage

Navigate to /examples/ and edit the config file newstest_test.ini for values: batch size, bert batch size, epochs, CUDA device ID, checkpoint interval, eval interval (still have to automate that). 

Remove the 'cuda' and make it 'device cpu' if you want to run on CPU locally.

Run Train.py. Before running Predict.py you will need to change the name of the folder (line 15) to the correct directory in the results folder which contains the checkpoint saved after training is done (still need to automate that, folder name is currently dependent on epoch no).

## Issues

F1 metrics during training do not match with testing and are too optimistic. Have to debug.
