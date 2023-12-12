from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import os


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
    df_shuffled = df.sample(len(df), random_state=random_state) 
 
    num_train_samples =  int(len(df_shuffled)*train_size)
    for split in ['train', 'test']:

        samples = df_shuffled[:num_train_samples] if  split == 'train' else df_shuffled[num_train_samples:]
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