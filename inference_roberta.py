from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from sklearn.metrics import f1_score
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import os

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
    

label2id = {
    'Ineffective' : 2,
    'Effective' : 1,
    'Adequate' : 0
}
id2label = {
    id : label for label, id  in label2id.items()
}

def compute_score(y_true, y_pred):
    scores = f1_score(y_true, y_pred, average=None)
    for i, score in enumerate(scores):
        print(f"f-score for class '{id2label[i]}' = {score:.4f}")




def preprocess(data_path, random_state, train_size):
    roberta = RobertaModel.from_pretrained(
    'checkpoints',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data-bin'
).cuda()
    roberta.eval()  
    
    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )

    df = pd.read_csv(Path(data_path) / 'train.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['discourse_text'], df['discourse_effectiveness'],
                                                        stratify=df['discourse_effectiveness'],
                                                        test_size=0.1,
                                                        random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)


    samples = list(df_test[['discourse_text', 'discourse_effectiveness']].values)
    
    y_true, y_pred = [], []
    for sample in tqdm(samples):
        text = sample[0].replace("\n", "")
        label = str(label2id[sample[1]])
        tokens = roberta.encode(text)[:512]
        # pred = label_fn(roberta.predict('discourse_head', tokens).argmax().item())
        pred = roberta.predict('discourse_head', tokens).detach().cpu().numpy()[0]

        
        y_true.append(label)
        y_pred.append(softmax(pred))

    print(log_loss(y_true, y_pred))
    
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--random_state', default=42)
    parser.add_argument('--train_size', default=0.8)
    args = parser.parse_args()
    preprocess(args.data_path, args.random_state, args.train_size)
