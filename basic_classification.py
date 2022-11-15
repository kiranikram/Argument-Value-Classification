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

df_small = df.sample(frac=0.12, random_state=30)
df_small.reset_index(drop=True,inplace=True)

def get_embeddings(texts,model='medium'):
  output = co.embed(
                model=model,
                texts=texts)
  return output.embeddings

df_small['premise_embeds'] = get_embeddings(df_small['Premise'].tolist())

def get_pc(arr,n):
  pca = PCA(n_components=n)
  embeds_transform = pca.fit_transform(arr)
  return embeds_transform

embeds = np.array(df_small['premise_embeds'].tolist())
embeds_pc = get_pc(embeds,10)

sample = 9
source = pd.DataFrame(embeds_pc)[:sample]
source = pd.concat([source,df_small['Premise']], axis=1)
source = source.dropna()
source = source.melt(id_vars=['Premise'])

chart = alt.Chart(source).mark_rect().encode(
    x=alt.X('variable:N', title="Embedding"),
    y=alt.Y('Premise:N', title='',axis=alt.Axis(labelLimit=500)),
    color=alt.Color('value:Q', title="Value", scale=alt.Scale(
                range=["#917EF3", "#000000"]))
)

result = chart.configure(background='#ffffff'
        ).properties(
        width=700,
        height=400,
        title='Embeddings with 10 dimensions'
       ).configure_axis(
      labelFontSize=15,
      titleFontSize=12)

# Show the plot
result

def generate_chart(df,xcol,ycol,lbl='on',color='basic',title=''):
  chart = alt.Chart(df).mark_circle(size=500).encode(
    x=
    alt.X(xcol,
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labels=False, ticks=False, domain=False)
    ),

    y=
    alt.Y(ycol,
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labels=False, ticks=False, domain=False)
    ),
    
    color= alt.value('#333293') if color == 'basic' else color,
    tooltip=['Premise']
    )

  if lbl == 'on':
    text = chart.mark_text(align='left', baseline='middle',dx=15, size=13,color='black').encode(text='Premise', color= alt.value('black'))
  else:
    text = chart.mark_text(align='left', baseline='middle',dx=10).encode()

  result = (chart + text).configure(background="#FDF7F0"
        ).properties(
        width=800,
        height=500,
        title=title
       ).configure_legend(
  orient='bottom', titleFontSize=18,labelFontSize=18)
        
  return result

embeds_pc2 = get_pc(embeds,2)

# Add the principal components to dataframe
df_pc2 = pd.concat([df_small, pd.DataFrame(embeds_pc2)], axis=1)
df_pc2 = df_pc2.dropna()

# Plot the 2D embeddings on a chart
df_pc2.columns = df_pc2.columns.astype(str)
generate_chart(df_pc2.iloc[:sample],'0','1',title='2D Embeddings')

