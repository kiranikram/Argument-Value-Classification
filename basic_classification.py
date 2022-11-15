import cohere
import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA

from data_load import load_data

api_key = 'NuYO6K6yfYM2dMJbiUZ0qKHYd15KA8tmUjmnuivJ' # Paste your API key here. Remember to not share it publicly 
co = cohere.Client(api_key)

df = load_data('data/arguments-training.tsv')

df.drop(columns=['Argument ID','Conclusion','Stance'],inplace=True)

df_small = df.sample(frac=0.2, random_state=30)

def get_embeddings(texts,model='medium'):
  output = co.embed(
                model=model,
                texts=texts)
  return output.embeddings

df_small['premise_embeds'] = get_embeddings(df_small['Premise'].tolist())