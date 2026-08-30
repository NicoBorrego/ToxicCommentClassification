import re

import pandas as pd
import classifier
from sklearn.feature_extraction.text import TfidfVectorizer

cls = classifier.Classifier()

df_twitter = pd.read_csv("../resources/twitter/FinalBalancedDataset.csv")

#print(df_twitter.head())

df_twitter = df_twitter.dropna(subset=['tweet'])
toxic = df_twitter[df_twitter['Toxicity'] == 1].sample(n=100, random_state=42)
non_toxic = df_twitter[df_twitter['Toxicity'] == 0].sample(n=100, random_state=42)

# 3. Unir ambas muestras
sample_df = pd.concat([toxic, non_toxic])

# 4. Desordenar las filas (shuffle) para que no estén los 100 primeros juntos
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

for index, row in sample_df.iterrows():
    prediction = cls.predict(row['tweet'])
    sample_df.at[index, 'toxicity_classification'] = prediction["is_toxic"]

print("Toxicity Distribution:\n")
print(sample_df['toxicity_classification'].value_counts())
print("-" * 50)

print("\nClassification examples:\n")
print(sample_df[['tweet', 'toxicity_classification']].sample(10, random_state=42))
print("-" * 50)

