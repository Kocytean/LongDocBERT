import pandas as pd
import glob

def load(mode = 'TRAIN'):
    folder_name = './TrainingData/' if 'TRAIN' in mode else './TestingData/'
    all_files = glob.glob(folder_name+ '*.csv')
    data = pd.concat([pd.read_csv(file) for file in all_files]).sample(frac=1)
    labels = data.iloc[:,0]
    texts = data.iloc[:,1].to_list()
    labelset = labels.unique()
    return labels.to_list(), texts, sorted(labelset)

def generate(mode = 'TRAIN'):
    labels, texts, _ = load(mode)
    for i, (text, label) in enumerate(zip(texts, labels)):
        yield i,text, label
