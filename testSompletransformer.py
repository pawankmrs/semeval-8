import pandas as pd
import sklearn


datapath='.'


all_classes = list(set([l.strip() for l in open('test-trg.txt')] + [l.strip() for l in open('train-trg.txt')]))
name2id = {cl:i for i,cl in enumerate(all_classes)}

train_df = pd.DataFrame({'text': [l.strip() for l in open('train-src.txt')], 'label': [name2id[l.strip()] for l in open('train-trg.txt')]})

eval_df = pd.DataFrame({'text': [l.strip() for l in open('test-src.txt')], 'label': [name2id[l.strip()] for l in open('test-trg.txt')]})


from simpletransformers.classification import ClassificationModel, ClassificationArgs

model_args = ClassificationArgs(num_train_epochs=5)
model_args.n_gpu=3
model_args.train_batch_size=64

# Create a TransformerModel
#model = ClassificationModel('roberta', 'roberta-base', num_labels=len(all_classes), args=model_args)

# Train the model
#model.train_model(train_df)

model = ClassificationModel('roberta', 'outputs/checkpoint-625-epoch-5')

#f1_score = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average='macro')

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=sklearn.metrics.accuracy_score)


print(result)
