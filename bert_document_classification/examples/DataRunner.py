import pandas as pd
import glob
# def load_newstest( random_state=42 , categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']):

#     remove=('headers', 'footers')

#     data_train = fetch_20newsgroups(subset='train', categories=categories,
#                                     shuffle=True, random_state=random_state,
#                                     remove=remove)

#     data_test = fetch_20newsgroups(subset='test', categories=categories,
#                                    shuffle=True, random_state=random_state,
#                                    remove=remove)

#     target_names = data_train.target_names
#     print(  'labels for newstest dataset:', target_names )

#     return data_train, data_test, target_names

# def generator_newstest( data, target_names   ):
#     """
#     Yields a generator of id, doc, label tuples.
#     :param dict of newstest data , target_names/labels:
#     :return:
#     """
#     ids=[]
#     documents=[]
#     labels=[]
#     for index, (text, nr_label) in enumerate(zip( data.data, data.target )):
#         ids.append( index )
#         documents.append( text )
#         labels.append( target_names[ nr_label  ]   )

#     for id, text, label in zip(ids,documents,labels):
#         yield (id,text,label)

def load(mode = 'TRAIN'):
    folder_name = './TrainingData/' if 'TRAIN' in mode else './TestingData/'
    all_files = glob.glob(folder_name+ '*.csv')
    data = pd.concat([pd.read_csv(file) for file in all_files]).sample(frac=1)
    labels = data.iloc[:,0]
    texts = data.iloc[:,1].to_list()
    labelset = labels.unique()
    return labels.to_list(), texts, sorted(labelset)

def generate(mode = 'TRAIN'):
    labels, texts, _ = load()
    for i, (text, label) in enumerate(zip(texts, labels)):
        yield i,text, label