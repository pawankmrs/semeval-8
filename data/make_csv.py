import pandas as pd
import sklearn



all_classes = list(set([l.strip() for l in open('test-trg.txt')] + [l.strip() for l in open('train-trg.txt')]))
name2id = {cl:i for i,cl in enumerate(all_classes)}

print( name2id)
train_df = pd.DataFrame({'text': [l.strip() for l in open('train-src.txt')], 'label': [name2id[l.strip()] for l in open('train-trg.txt')]})
eval_df = pd.DataFrame({'text': [l.strip() for l in open('test-src.txt')], 'label': [name2id[l.strip()] for l in open('test-trg.txt')]})

#train_df = pd.DataFrame({'text': [l.strip() for l in open('train-src.txt')], 'label': [l.strip() for l in open('train-trg.txt')]})
#eval_df = pd.DataFrame({'text': [l.strip() for l in open('test-src.txt')], 'label': [l.strip() for l in open('test-trg.txt')]})


train_df.to_csv('train.csv', index=False)
eval_df.to_csv('test.csv', index=False)

