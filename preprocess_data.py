from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import os
from sklearn.model_selection import train_test_split


label2id = {
    'Ineffective' : 2,
    'Effective' : 1,
    'Adequate' : 0
}
id2label = {
    id : label for label, id  in label2id.items()
}



def preprocess(data_path, random_state, train_size):
    df = pd.read_csv(Path(data_path) / 'train.csv')

    X_train, X_test, y_train, y_test = train_test_split(df['discourse_text'], df['discourse_effectiveness'],
                                                        stratify=df['discourse_effectiveness'],
                                                        test_size=0.1,
                                                        random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    for split in ['train', 'test']:
        samples = df_train if  split == 'train' else df_test
        samples = list(samples[['discourse_text', 'discourse_effectiveness']].values)
    
        out_fname = 'train' if split == 'train' else 'dev'

        with open(os.path.join(data_path, out_fname + '.input0'), 'w') as f1, open(os.path.join(data_path, out_fname + '.label'), 'w') as f2:
            for sample in samples:
                f1.write(sample[0].replace("\n", "") + "\n")
                f2.write(str(label2id[sample[1]]) + "\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--random_state', default=42)
    parser.add_argument('--train_size', default=0.8)
    args = parser.parse_args()
    preprocess(args.data_path, args.random_state, args.train_size)