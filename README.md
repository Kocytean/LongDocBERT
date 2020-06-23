# BERT for document classification

BERT for classifying arbitrarily long input text (limited to 10200 tokens) 

## Prerequisites

torch, transformers, pandas, configargparse, sklearn

## Usage

Navigate to /examples/ and edit the config file conf.ini for values: batch size, bert batch size, epochs, CUDA device ID, checkpoint interval, eval interval (still have to automate that). 

Remove the 'cuda' and make it 'device cpu' if you want to run on CPU locally.

Run Train.py. Before running Predict.py you will need to change the name of the folder (line 15) to the correct directory in the results folder which contains the checkpoint saved after training is done (still need to automate that, folder name is currently dependent on epoch no).

