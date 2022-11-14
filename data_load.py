import pandas as pd
import typing as tp

def load_data(path:str) -> pd.DataFrame:
    data = pd.read_csv(path,sep='\t')

    return data

train_data = load_data('data/arguments-training.tsv')
labels = load_data('data/labels-training.tsv')