# Toxic Comment Classification

## What is about?

This repo is focused on learning how to develop, optimize and evaluate a supervised machine learning model to treat Natural Language Processing and classify a toxic or non-toxic comment provided by the user. I refer toxic as an insult only.

Once the model is selected, lets proceed with an external validation with [Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset) from Kaggle.

Also, you can try the model on my own [portfolio](https://bold-block-32cb.nicolasborregogonzalez.workers.dev/), with an API deployed with Docker on Render.

## Selected model

After trying some classification model as logistic regresion, random forests, etc. I decided to use LinearSVC from SVM as explained on the repo Notebook.

## External validation (validation.py)

Once I treated the NaN values, I made a sample of 100 toxic and 100 non-toxic tweets and the results were:

Toxicity Distribution:

```
toxicity_classification
False    120
True      80
Name: count, dtype: int64
--------------------------------------------------
                                                 tweet toxicity_classification  confidence
95   #oitnb â ¤ who's   !! #moi #me ð    #tgif #nom...                   False      0.9311
15   the greetings are so incredible @user thank yo...                   False      0.9311
30                            Sgp still in bitches dms                    True      0.9311
158  @KimberlyyAye @Leslieeeixta I think I did my j...                    True      0.9311
128  ' August Alsina Deserved that &#1043323;&#1043...                   False      0.9311
115  @user when saying @user surname keep saying as...                   False      0.9311
69   since i finished most of my assignments i'll c...                   False      0.9311
170  I tell bitches I got that I.T.S ( IKE TURNER S...                    True      0.9311
174  #theshallows weak attempt to cash in on the ja...                   False      0.9311
45   wonderful dinner â ¤ï¸  #dinner #yum #thegaffo...                   False      0.9311
```
There are some toxic tweets in the dataset that are not scored as toxic by this model cause it could contains numbers as other simbols. 
